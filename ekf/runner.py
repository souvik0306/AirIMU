"""Bounded feature caching and batched GPU heads with parallel CPU ESKF replay."""

from collections import OrderedDict
from time import perf_counter

import torch

from .data import prepare_flight
from .eskf import DifferentiableESKF
from .experiment import CovarianceStatistics, covariance_statistics
from .parallel import FlightReplayPool, replay_velocity


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class FlightTrainer:
    def __init__(self, wrapper, optimizer, *, workers=0, batch_size=1, feature_cache_mb=512,
                 replay_backend="eager"):
        self.wrapper = wrapper
        self.optimizer = optimizer
        self.parameters = [p for p in wrapper.network.parameters() if p.requires_grad]
        self.device = next(wrapper.network.parameters()).device
        self.batch_size = batch_size
        self.replay_backend = replay_backend
        self.cpu_filter = DifferentiableESKF(gravity=wrapper.ekf.gravity.cpu().tolist(),
                                             height_updates=wrapper.ekf.height_updates)
        self.features = OrderedDict()
        self.feature_bytes = 0
        self.feature_limit = int(feature_cache_mb * 1024**2)
        self.cache_hits = 0
        self.cache_misses = 0
        self.optimizer_steps = 0
        self.pool = FlightReplayPool(workers, gravity=self.cpu_filter.gravity.tolist(),
                                     backend=replay_backend,
                                     height_updates=self.cpu_filter.height_updates) if workers else None

    def _features(self, records, cutoffs):
        result = []
        for record in records:
            cutoff = cutoffs[record["id"]]
            key = (record["id"], cutoff)
            cached = self.features.pop(key, None)
            if cached is None:
                self.cache_misses += 1
                flight = prepare_flight(record, cutoff, self.cpu_filter)
                cached = self.wrapper.extract_features(flight["raw_acceleration"], flight["raw_gyroscope"])
                cached = {name: value.cpu() if torch.is_tensor(value) else value
                          for name, value in cached.items()}
                size = sum(value.numel()*value.element_size() for value in cached.values() if torch.is_tensor(value))
                if size <= self.feature_limit:
                    while self.feature_bytes + size > self.feature_limit and self.features:
                        _, old = self.features.popitem(last=False)
                        self.feature_bytes -= sum(value.numel()*value.element_size()
                                                  for value in old.values() if torch.is_tensor(value))
                    self.features[key] = cached
                    self.feature_bytes += size
            else:
                self.cache_hits += 1
                self.features[key] = cached
            result.append(cached)
        return result

    def run_split(self, records, cutoffs, *, epoch, phase, epoch_start, log):
        training = phase == "train"
        self.wrapper.train(training)
        losses = []
        aggregate = CovarianceStatistics()
        batches = [records[start:start+self.batch_size] for start in range(0, len(records), self.batch_size)]
        cached = None
        for batch_index, batch in enumerate(batches):
            started = perf_counter()
            for index, record in enumerate(batch):
                print(f"epoch={epoch} phase={phase} flight={record['id']} "
                      f"progress={batch_index*self.batch_size+index+1}/{len(records)} status=started", flush=True)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            if cached is None:
                cached = self._features(batch, cutoffs)
            prepared = perf_counter()
            with torch.set_grad_enabled(training):
                variances = self.wrapper.predict_variance_batch(cached)
            synchronize(self.device)
            decoded = perf_counter()
            if self.pool:
                futures = [self.pool.submit(record, cutoffs[record["id"]], variance, backward=training)
                           for record, variance in zip(batch, variances)]
                # Frozen feature extraction for the next batch can overlap CPU
                # replay; trainable heads are not evaluated until Adam updates.
                cached = self._features(batches[batch_index+1], cutoffs) if (
                    batch_index+1 < len(batches)) else None
                outputs = [future.result() for future in futures]
            else:
                outputs = [replay_velocity(self.wrapper.ekf,
                           prepare_flight(record, cutoffs[record["id"]], self.cpu_filter),
                           variance, backward=training, backend=self.replay_backend)
                           for record, variance in zip(batch, variances)]
                cached = None
            replayed = perf_counter()
            if training:
                # Exact chain rule: workers compute dL/dvariance at the current
                # head weights, then one batched backward computes the mean dL/dw.
                gradients = [torch.as_tensor(output["gradient"], device=variance.device,
                                             dtype=variance.dtype) / len(batch)
                             for output, variance in zip(outputs, variances)]
                torch.autograd.backward(variances, gradients)
                checks = [torch.isfinite(p.grad).all() for p in self.parameters if p.grad is not None]
                if not checks or not bool(torch.stack(checks).all()):
                    raise RuntimeError("Missing or non-finite covariance-head gradients.")
                synchronize(self.device)
                backwarded = perf_counter()
                self.optimizer.step()
                synchronize(self.device)
                self.optimizer_steps += 1
            else:
                backwarded = perf_counter()
            finished = perf_counter()
            for index, (record, variance, output) in enumerate(zip(batch, variances, outputs)):
                losses.append(output["loss"])
                covariance = covariance_statistics(variance)
                aggregate.update(covariance)
                log({"epoch": epoch, "phase": phase, "flight": record["id"],
                     "flight_index": batch_index*self.batch_size+index+1, "flight_count": len(records),
                     "batch_index": batch_index+1, "batch_size": len(batch),
                     f"{phase}_xy_mse": output["loss"], "covariance": covariance,
                     "timing_seconds": output["timing_seconds"],
                     "worker_pid": output.get("worker_pid"),
                     "worker_peak_memory_mb": output.get("worker_peak_memory_mb"),
                     "height_update_count": output["height_update_count"],
                     "epoch_elapsed_seconds": perf_counter()-epoch_start})
            replay_cores = sum(output["timing_seconds"].get("cpu", 0.) for output in outputs) / max(replayed-decoded, 1e-9)
            gpu_peak = torch.cuda.max_memory_allocated(self.device)/1024**3 if self.device.type == "cuda" else 0.
            print(f"epoch={epoch} phase={phase} batch={batch_index+1}/{len(batches)} "
                  f"flights={len(batch)} prepare_s={prepared-started:.3f} heads_s={decoded-prepared:.3f} "
                  f"replay_wall_s={replayed-decoded:.3f} heads_backward_s={backwarded-replayed:.3f} "
                  f"optimizer_s={finished-backwarded:.3f} replay_cpu_cores={replay_cores:.2f} "
                  f"gpu_peak_gb={gpu_peak:.2f} feature_cache_mb={self.feature_bytes/1024**2:.1f}", flush=True)
            # Prevent graphs from the previous batch surviving the next replay.
            del variances, outputs, variance, output
            if training:
                del gradients
        return losses, aggregate.summary()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.pool:
            self.pool.close()
