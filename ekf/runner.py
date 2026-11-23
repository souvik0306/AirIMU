"""Fresh trainable GRUs or cached frozen features with parallel CPU ESKF replay."""

from collections import OrderedDict
import math
import resource
from time import perf_counter

import torch

from .data import prepare_flight
from .eskf import DifferentiableESKF
from .experiment import CovarianceStatistics, covariance_statistics
from .parallel import FlightReplayPool, replay_velocity
from .gradients import (component_gradient_diagnostics, gradient_norm as parameter_gradient_norm,
                        summarize_batches, clipping_groups, clip_gradients, parameter_update_diagnostics)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class FlightTrainer:
    def __init__(self, wrapper, optimizer, *, workers=0, batch_size=1, feature_cache_mb=512,
                 replay_backend="eager", loss_options=None, gradient_clip=1., correction_weight=0.,
                 gradient_clipping="global", gru2_gradient_clip=5., head_gradient_clip=1.,
                 max_relative_update=None, cnn_gradient_clip=1., gru1_gradient_clip=1.,
                 require_encoder_gradients=False, adapter_weight=0.):
        self.wrapper = wrapper
        if not math.isfinite(adapter_weight) or adapter_weight < 0:
            raise ValueError('adapter_weight must be finite and nonnegative.')
        if adapter_weight and wrapper.configuration != 'Adapter':
            raise ValueError('Adapter regularization requires adapter-only training.')
        self.adapter_weight = adapter_weight
        self.optimizer = optimizer
        self.parameters = [p for p in wrapper.network.parameters() if p.requires_grad]
        self.device = next(wrapper.network.parameters()).device
        self.batch_size = batch_size
        self.replay_backend = replay_backend
        self.loss_options = loss_options or {}
        self.gradient_clip = gradient_clip
        self.gradient_clipping = gradient_clipping
        self.clipping_groups = clipping_groups(optimizer.param_groups, mode=gradient_clipping,
            global_limit=gradient_clip, gru2_limit=gru2_gradient_clip, head_limit=head_gradient_clip,
            cnn_limit=cnn_gradient_clip, gru1_limit=gru1_gradient_clip)
        self.require_encoder_gradients = require_encoder_gradients
        if {id(p) for group in self.clipping_groups for p in group['params']} != {id(p) for p in self.parameters}:
            raise ValueError('Optimizer/clipping sets must cover every trainable network parameter exactly once.')
        if max_relative_update is not None and (not math.isfinite(max_relative_update) or max_relative_update <= 0):
            raise ValueError('Maximum relative update must be positive and finite.')
        self.max_relative_update = max_relative_update
        self.correction_weight = correction_weight
        self.cpu_filter = DifferentiableESKF(gravity=wrapper.ekf.gravity.cpu().tolist(),
                                             height_updates=wrapper.ekf.height_updates)
        self.features = OrderedDict()
        self.feature_bytes = 0
        self.feature_limit = int(feature_cache_mb * 1024**2) if wrapper.cacheable_features else 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.optimizer_steps = 0
        self.last_split_diagnostics = {}
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
                if self.wrapper.cacheable_features:
                    cached = {name: value.cpu() if torch.is_tensor(value) else value
                              for name, value in cached.items()}
                size = sum(value.numel()*value.element_size() for value in cached.values() if torch.is_tensor(value))
                if self.wrapper.cacheable_features and size <= self.feature_limit:
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

    def run_split(self, records, cutoffs, *, epoch, phase, epoch_start, log, log_batch=None):
        training = phase == "train"
        self.wrapper.train(training)
        losses = []
        aggregate = CovarianceStatistics()
        distribution_samples=[] if self.loss_options.get('orientation_smoothing_window',0.) else None
        batches = [records[start:start+self.batch_size] for start in range(0, len(records), self.batch_size)]
        cached = None
        gradient_records = []
        quality_totals = {}
        for batch_index, batch in enumerate(batches):
            started = perf_counter()
            for index, record in enumerate(batch):
                print(f"epoch={epoch} phase={phase} flight={record['id']} "
                      f"progress={batch_index*self.batch_size+index+1}/{len(records)} status=started", flush=True)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            if cached is None:
                with torch.set_grad_enabled(training):
                    cached = self._features(batch, cutoffs)
            corrections = [item["correction"] - item["reference_correction"] for item in cached]
            with torch.set_grad_enabled(training):
                scales = torch.cat((self.wrapper.network.acc_std.expand(3), self.wrapper.network.gyro_std.expand(3)))
                consistency = torch.stack([(delta/scales.to(delta)).square().mean() for delta in corrections]).mean()
            correction_rmse = [delta.detach().square().mean(0).sqrt().cpu().tolist() for delta in corrections]
            prepared = perf_counter()
            with torch.set_grad_enabled(training):
                if self.wrapper.configuration == 'Adapter':
                    variances, adapter_deltas = self.wrapper.predict_variance_batch(cached, return_adapter_deltas=True)
                    adapter_penalties = [delta.square().mean() for delta in adapter_deltas]
                    adapter_penalty = torch.stack(adapter_penalties).mean()
                else:
                    variances = self.wrapper.predict_variance_batch(cached)
                    adapter_deltas = adapter_penalties = None
            if any(not bool(torch.isfinite(q).all()) or bool((q <= 0).any()) for q in variances):
                raise RuntimeError('Optimization instability: non-finite or non-positive predicted variance.')
            synchronize(self.device)
            decoded = perf_counter()
            if self.pool:
                futures = [self.pool.submit(record, cutoffs[record["id"]], variance, backward=training,
                                           loss_options=self.loss_options)
                           for record, variance in zip(batch, variances)]
                # Frozen feature extraction for the next batch can overlap CPU
                # replay; trainable heads are not evaluated until Adam updates.
                cached = self._features(batches[batch_index+1], cutoffs) if (
                    self.wrapper.cacheable_features and batch_index+1 < len(batches)) else None
                outputs = [future.result() for future in futures]
            else:
                outputs = [replay_velocity(self.wrapper.ekf,
                           prepare_flight(record, cutoffs[record["id"]], self.cpu_filter),
                           variance, backward=training, backend=self.replay_backend, loss_options=self.loss_options)
                           for record, variance in zip(batch, variances)]
                cached = None
            replayed = perf_counter()
            if any(not math.isfinite(output[key]) for output in outputs
                   for key in ('loss','total_loss','gyro_calibration_loss','smoothness_loss')):
                raise RuntimeError('Optimization instability: non-finite replay loss.')
            gradient_diagnostics = {}
            if training:
                gradient_diagnostics = component_gradient_diagnostics(
                    variances, outputs, self.parameters, self.optimizer.param_groups,
                    self.loss_options.get("gyro_weight", 0))
                if adapter_deltas is not None:
                    regularizer_grads = torch.autograd.grad(self.adapter_weight*adapter_penalty,
                        self.parameters, retain_graph=True, allow_unused=True)
                    gradient_diagnostics['gradient_norm_adapter_regularization'] = parameter_gradient_norm(regularizer_grads)
                # Exact chain rule: workers compute dL/dvariance at the current
                # head weights, then one batched backward computes the mean dL/dw.
                gradients = [torch.as_tensor(output["gradient"], device=variance.device,
                                             dtype=variance.dtype) / len(batch)
                             for output, variance in zip(outputs, variances)]
                backward_tensors, backward_gradients = list(variances), list(gradients)
                if adapter_deltas is not None and self.adapter_weight:
                    backward_tensors.append(adapter_penalty)
                    backward_gradients.append(torch.full_like(adapter_penalty, self.adapter_weight))
                if self.correction_weight and consistency.requires_grad:
                    backward_tensors.append(consistency)
                    backward_gradients.append(torch.full_like(consistency, self.correction_weight))
                torch.autograd.backward(backward_tensors, backward_gradients)
                checks = [torch.isfinite(p.grad).all() for p in self.parameters if p.grad is not None]
                if not checks or not bool(torch.stack(checks).all()):
                    raise RuntimeError("Missing or non-finite covariance-head gradients.")
                if self.wrapper.configuration == 'Adapter' and (
                        any(p.grad is None for p in self.parameters)
                        or not any(bool(p.grad.ne(0).any()) for p in self.parameters)):
                    raise RuntimeError('Missing or all-zero adapter gradients.')
                for group_index,group in enumerate(self.optimizer.param_groups):
                    gradient_diagnostics['gradient_norms_by_group'][group.get('name',f'group_{group_index}')]['total'] = parameter_gradient_norm(
                        p.grad for p in group['params'])
                    if self.require_encoder_gradients and group.get('name') in ('cnn','gru1','gru2'):
                        if any(p.grad is None for p in group['params']) or not any(bool(p.grad.ne(0).any()) for p in group['params']):
                            raise RuntimeError(f"Missing/nonzero-gradient check failed for {group['name']}.")
                gradient_norm = parameter_gradient_norm(p.grad for p in self.parameters)
                clipped = clip_gradients(self.clipping_groups)
                for group_index,group in enumerate(self.optimizer.param_groups):
                    gradient_diagnostics['gradient_norms_by_group'][group.get('name',f'group_{group_index}')]['total_after_clip'] = parameter_gradient_norm(
                        p.grad for p in group['params'])
                gradient_diagnostics.update(gradient_norm_before_clip=gradient_norm,
                    gradient_norm_after_clip=parameter_gradient_norm(p.grad for p in self.parameters),
                    gradient_was_clipped=any(row['was_clipped'] for row in clipped.values()),
                    gradient_clip_scale=clipped['global']['scale'] if self.gradient_clipping == 'global' else None,
                    gradient_clipping_groups=clipped)
                before_update = {id(p):p.detach().clone() for p in self.parameters}
                synchronize(self.device)
                backwarded = perf_counter()
                self.optimizer.step()
                gradient_diagnostics['parameter_updates_by_group'] = parameter_update_diagnostics(
                    self.optimizer.param_groups, before_update, self.max_relative_update)
                del before_update
                gradient_records.append(gradient_diagnostics)
                synchronize(self.device)
                self.optimizer_steps += 1
            else:
                backwarded = perf_counter()
            finished = perf_counter()
            for index, (record, variance, output) in enumerate(zip(batch, variances, outputs)):
                losses.append(output["loss"])
                covariance = covariance_statistics(variance)
                if distribution_samples is not None:distribution_samples.append(variance.detach().cpu())
                aggregate.update(covariance)
                quality = output.get('gyro_reference_quality') or {}
                for key in ('samples', 'valid_samples', 'invalid_samples', 'boundary_windows',
                            'gap_windows', 'invalid_quaternion_windows', 'orientation_jump_windows',
                            'angular_rate_windows', 'invalid_quaternions', 'orientation_jump_intervals',
                            'angular_rate_intervals'):
                    quality_totals[key] = quality_totals.get(key, 0)+quality.get(key, 0)
                log({"epoch": epoch, "phase": phase, "flight": record["id"],
                     "flight_index": batch_index*self.batch_size+index+1, "flight_count": len(records),
                     "batch_index": batch_index+1, "batch_size": len(batch),
                     f"{phase}_xy_mse": output["loss"], "covariance": covariance,
                     f"{phase}_xy_rmse": math.sqrt(output["loss"]),
                     "total_loss": output["total_loss"] + self.correction_weight*float(consistency.detach())
                         + (self.adapter_weight*float(adapter_penalties[index].detach()) if adapter_penalties is not None else 0.),
                     **({'adapter_delta_mse':float(adapter_penalties[index].detach()),
                         'adapter_scale_mean_xyz_acc_gyro':adapter_deltas[index].detach().exp().mean(0).cpu().tolist(),
                         'adapter_delta_max_abs':float(adapter_deltas[index].detach().abs().max())}
                         if adapter_deltas is not None else {}),
                     "gyro_calibration_loss": output["gyro_calibration_loss"],
                     "smoothness_loss": output["smoothness_loss"],
                     "gyro_diagnostics": output["gyro_diagnostics"],
                     "gyro_reference_quality": quality,
                     'gyro_reference_comparison':output.get('gyro_reference_comparison'),
                     "correction_drift_rmse": correction_rmse[index],
                     "correction_consistency_loss": float(consistency.detach()),
                     "gradient_norm_before_clip": gradient_norm if training else None,
                     **gradient_diagnostics,
                     "main_peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
                     "gpu_peak_memory_mb": torch.cuda.max_memory_allocated(self.device)/1024**2 if self.device.type == "cuda" else 0.,
                     "timing_seconds": output["timing_seconds"],
                     "worker_pid": output.get("worker_pid"),
                     "worker_peak_memory_mb": output.get("worker_peak_memory_mb"),
                     "height_update_count": output["height_update_count"],
                     "epoch_elapsed_seconds": perf_counter()-epoch_start})
            replay_cores = sum(output["timing_seconds"].get("cpu", 0.) for output in outputs) / max(replayed-decoded, 1e-9)
            gpu_peak = torch.cuda.max_memory_allocated(self.device)/1024**3 if self.device.type == "cuda" else 0.
            if log_batch is not None:
                log_batch({'epoch':epoch, 'phase':phase, 'batch_index':batch_index+1,
                    'batch_size':len(batch), 'flights':[record['id'] for record in batch],
                    **gradient_diagnostics,
                    'prepare_seconds':prepared-started, 'replay_seconds':replayed-decoded,
                    'network_backward_seconds':backwarded-replayed,
                    'optimizer_seconds':finished-backwarded})
            gradient_text = (f" grad_total={gradient_norm:.6g} "
                f"grad_vel={gradient_diagnostics['gradient_norm_velocity']:.6g} "
                f"grad_gyro={gradient_diagnostics['gradient_norm_gyro_calibration']:.6g} "
                f"grad_gyro_raw={gradient_diagnostics['gradient_norm_gyro_calibration_unweighted']:.6g}" if training else '')
            if training:
                gradient_text += ''.join(f" clip_{name}={value['norm_before']:.6g}->{value['norm_after']:.6g}"
                    for name,value in gradient_diagnostics['gradient_clipping_groups'].items())
                gradient_text += ''.join(f" update_{name}={value['l2']:.6g} rel_{name}={value['relative_l2']:.6g}"
                    for name,value in gradient_diagnostics['parameter_updates_by_group'].items())
            print(f"epoch={epoch} phase={phase} batch={batch_index+1}/{len(batches)} "
                  f"flights={len(batch)} prepare_s={prepared-started:.3f} heads_s={decoded-prepared:.3f} "
                  f"replay_wall_s={replayed-decoded:.3f} heads_backward_s={backwarded-replayed:.3f} "
                  f"optimizer_s={finished-backwarded:.3f} replay_cpu_cores={replay_cores:.2f} "
                  f"gpu_peak_gb={gpu_peak:.2f} feature_cache_mb={self.feature_bytes/1024**2:.1f}"
                  f"{gradient_text}", flush=True)
            # Prevent graphs from the previous batch surviving the next replay.
            del variances, outputs, variance, output
            del corrections, consistency
            if adapter_deltas is not None:
                del adapter_deltas, adapter_penalties, adapter_penalty
                if training:
                    del regularizer_grads
            if training:
                del gradients, backward_tensors, backward_gradients
        self.last_split_diagnostics = {'gradients':summarize_batches(gradient_records),
                                       'gyro_reference_quality':quality_totals}
        statistics=aggregate.summary()
        if distribution_samples:
            samples=torch.cat(distribution_samples)
            quantiles=samples.quantile(torch.tensor([.5,.95],dtype=samples.dtype),dim=0).tolist()
            for sensor,offset in (('acc_cov',0),('gyro_cov',3)):
                statistics[sensor]['median']=quantiles[0][offset:offset+3]
                statistics[sensor]['p95']=quantiles[1][offset:offset+3]
        return losses, statistics

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.pool:
            self.pool.close()
