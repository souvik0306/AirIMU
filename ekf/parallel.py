"""CPU flight replay workers return exact variance gradients to GPU covariance heads.

Only detached CPU arrays cross process boundaries. The main process owns CUDA,
the trainable network, and Adam. Workers retain the complete flight graph until
its velocity gradient has been computed; no state or gradient is truncated.
"""

from concurrent.futures import ProcessPoolExecutor
from collections import OrderedDict
import multiprocessing
import os
import resource
from time import perf_counter, process_time

import numpy as np
import torch

from .eskf import DifferentiableESKF
from .data import prepare_flight
from .training import horizontal_velocity_loss, velocity_evaluation_mask


_worker_filter = None
_worker_backend = "eager"
_worker_flights = OrderedDict()


def _initialize_worker(gravity, backend="eager", height_updates=False):
    global _worker_filter, _worker_backend
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _worker_filter = DifferentiableESKF(gravity=gravity, height_updates=height_updates)
    _worker_backend = backend


def replay_velocity(ekf, flight, variance, *, backward, backend="eager"):
    """Pure full-flight loss/VJP shared by serial checks and worker processes."""
    with torch.set_grad_enabled(backward):
        variance = variance.detach().requires_grad_(backward)
        started = perf_counter()
        if backend == "compiled":
            from .compiled import velocity_sequence
            velocity = velocity_sequence(ekf, flight, variance)
        elif backend == "eager":
            result = ekf.run_sequence(
                {"acc": flight["raw_acceleration"], "gyro": flight["raw_gyroscope"]},
                flight["imu_time"], variance, flight["vicon_events"], flight["cutoff_timestamp"],
                flight["initial_state"], flight["initial_covariance"], return_covariance=False)
            velocity = result["velocity"]
        else:
            raise ValueError(f"Unknown replay backend: {backend}")
        times = flight["imu_time"].to(ekf.gravity.device)
        ground_truth = flight["vicon_velocity_gt"].to(ekf.gravity.device)
        mask = velocity_evaluation_mask(times, flight["cutoff_timestamp"])
        loss = horizontal_velocity_loss(velocity, ground_truth, mask)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite full-flight ESKF velocity loss.")
        forwarded = perf_counter()
        gradient = torch.autograd.grad(loss, variance)[0] if backward else None
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("Non-finite full-flight ESKF variance gradient.")
        finished = perf_counter()
        return {"loss": float(loss.detach()), "gradient": gradient,
                "height_update_count": sum(event.observation_type == "height"
                                           for event in flight["vicon_events"]),
                "timing_seconds": {"forward": forwarded-started,
                                   "backward": finished-forwarded}}


def _replay_record(record, cutoff, variance, backward):
    started = perf_counter()
    cpu_started = process_time()
    key = (record["id"], cutoff)
    flight = _worker_flights.pop(key, None)
    if flight is None:
        flight = prepare_flight(record, cutoff, _worker_filter)
    _worker_flights[key] = flight
    # Bounded independently in each worker; never cache trainable tensors.
    while len(_worker_flights) > 16:
        _worker_flights.popitem(last=False)
    prepared = perf_counter()
    # NumPy uses one contiguous IPC buffer instead of per-observation tensor
    # handles. The large observation/state graph is constructed inside a worker.
    result = replay_velocity(_worker_filter, flight, torch.from_numpy(variance), backward=backward,
                             backend=_worker_backend)
    if result["gradient"] is not None:
        result["gradient"] = result["gradient"].numpy()
    result["timing_seconds"]["prepare"] = prepared-started
    result["timing_seconds"]["total"] = perf_counter()-started
    result["timing_seconds"]["cpu"] = process_time()-cpu_started
    result["worker_pid"] = os.getpid()
    result["worker_peak_memory_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
    return result


class FlightReplayPool:
    def __init__(self, workers, *, gravity, backend="eager", height_updates=False):
        # Spawn prevents inherited CUDA/BLAS state from deadlocking workers.
        self.executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker, initargs=(gravity, backend, height_updates))

    def submit(self, record, cutoff, variance, *, backward=True):
        if variance.device.type != "cpu":
            raise ValueError("Parallel replay requires --ekf-device cpu.")
        values = np.array(variance.detach().numpy(), copy=True, order="C")
        return self.executor.submit(_replay_record, record, cutoff, values, backward)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
