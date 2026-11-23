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
from .regularization import (gyro_reference, gyro_calibration_loss,
                             log_variance_smoothness, gyro_diagnostics, compare_gyro_references)


_worker_filter = None
_worker_backend = "eager"
_worker_flights = OrderedDict()


def _initialize_worker(gravity, backend="eager", height_updates=False):
    global _worker_filter, _worker_backend
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _worker_filter = DifferentiableESKF(gravity=gravity, height_updates=height_updates)
    _worker_backend = backend


def replay_velocity(ekf, flight, variance, *, backward, backend="eager", loss_options=None):
    """Pure full-flight loss/VJP shared by serial checks and worker processes."""
    with torch.set_grad_enabled(backward):
        variance = variance.detach().requires_grad_(backward)
        options = loss_options or {}
        calibration = "vicon_quaternion_gt" in flight
        started = perf_counter()
        if backend == "compiled":
            from .compiled import velocity_sequence
            replay = velocity_sequence(ekf, flight, variance, return_gyro_bias=calibration)
            velocity, bias = replay if calibration else (replay, None)
        elif backend == "eager":
            result = ekf.run_sequence(
                {"acc": flight["raw_acceleration"], "gyro": flight["raw_gyroscope"]},
                flight["imu_time"], variance, flight["vicon_events"], flight["cutoff_timestamp"],
                flight["initial_state"], flight["initial_covariance"], return_covariance=False)
            velocity = result["velocity"]
            bias = result["gyroscope_bias"]
        else:
            raise ValueError(f"Unknown replay backend: {backend}")
        times = flight["imu_time"].to(ekf.gravity.device)
        ground_truth = flight["vicon_velocity_gt"].to(ekf.gravity.device)
        mask = velocity_evaluation_mask(times, flight["cutoff_timestamp"])
        loss = horizontal_velocity_loss(velocity, ground_truth, mask)
        gyro_loss = loss.new_zeros(())
        diagnostic_args = {}
        reference_quality = None
        reference_comparison = None
        if calibration:
            key = (options.get("calibration_window", .05), options.get("orientation_std", .001),
                   options.get("calibration_max_step_deg", 45.), options.get("calibration_max_rate", 20.),
                   options.get('orientation_smoothing_window',0.))
            if flight.get("_gyro_reference_key") != key:
                flight["_gyro_reference"] = gyro_reference(
                    flight["imu_time"], flight["vicon_quaternion_gt"],
                    orientation_time=flight.get("vicon_time_gt"),
                    window_seconds=key[0], orientation_std=key[1],
                    max_step_degrees=key[2], max_angular_rate=key[3],smoothing_window_seconds=key[4])
                flight["_gyro_reference_key"] = key
            reference_quality = flight["_gyro_reference"]["quality"]
            if bool(flight["_gyro_reference"]["mask"].any()):
                gyro_loss, residual, effective_q, valid = gyro_calibration_loss(
                    variance, flight["raw_gyroscope"], bias, flight["_gyro_reference"])
                diagnostic_args = dict(residual=residual, effective_q=effective_q, mask=valid)
            reference_comparison=compare_gyro_references(variance,flight['raw_gyroscope'],bias,flight['_gyro_reference'])
        elif options.get("gyro_weight", 0):
            raise ValueError("Gyro calibration requires aligned Vicon orientations.")
        smooth = log_variance_smoothness(variance, times)
        total = loss + options.get("gyro_weight", 0)*gyro_loss + options.get("smoothness_weight", 0)*smooth
        if not bool(torch.isfinite(total)):
            raise RuntimeError("Non-finite full-flight ESKF velocity loss.")
        forwarded = perf_counter()
        gradient = velocity_gradient = gyro_gradient = None
        if backward:
            # Only velocity requires EKF backpropagation. Calibration's bias is
            # detached, so its direct variance graph can be differentiated
            # independently without replaying or retaining the filter graph.
            velocity_gradient = torch.autograd.grad(loss, variance)[0]
            gyro_gradient = (torch.autograd.grad(gyro_loss, variance)[0]
                             if gyro_loss.requires_grad else torch.zeros_like(variance))
            gradient = velocity_gradient + options.get("gyro_weight", 0)*gyro_gradient
            if options.get("smoothness_weight", 0):
                gradient = gradient + options["smoothness_weight"]*torch.autograd.grad(smooth, variance)[0]
            if not all(bool(torch.isfinite(g).all()) for g in (gradient, velocity_gradient, gyro_gradient)):
                raise RuntimeError("Non-finite full-flight ESKF variance gradient.")
        finished = perf_counter()
        return {"loss": float(loss.detach()), "gradient": gradient,
                "velocity_gradient": velocity_gradient, "gyro_calibration_gradient": gyro_gradient,
                "total_loss": float(total.detach()), "gyro_calibration_loss": float(gyro_loss.detach()),
                "smoothness_loss": float(smooth.detach()),
                "gyro_diagnostics": gyro_diagnostics(variance, times, flight["raw_gyroscope"],
                    floor=options.get("gyro_floor", 1e-8), **diagnostic_args),
                "gyro_reference_quality": reference_quality,
                'gyro_reference_comparison':reference_comparison,
                "height_update_count": sum(event.observation_type == "height"
                                           for event in flight["vicon_events"]),
                "timing_seconds": {"forward": forwarded-started,
                                   "backward": finished-forwarded}}


def _replay_record(record, cutoff, variance, backward, loss_options):
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
                             backend=_worker_backend, loss_options=loss_options)
    for name in ("gradient", "velocity_gradient", "gyro_calibration_gradient"):
        if result[name] is not None:
            result[name] = result[name].numpy()
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

    def submit(self, record, cutoff, variance, *, backward=True, loss_options=None):
        if variance.device.type != "cpu":
            raise ValueError("Parallel replay requires --ekf-device cpu.")
        values = np.array(variance.detach().numpy(), copy=True, order="C")
        return self.executor.submit(_replay_record, record, cutoff, values, backward, loss_options)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
