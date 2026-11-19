"""Compiled, exact aligned-timestamp ESKF replay for covariance-head training.

The reference filter handles arbitrary asynchronous event schedules. TLab's
aligned rows allow prediction and the three sequential updates to be compiled
as one step. The state and covariance remain differentiable across every step.
"""

from functools import lru_cache
import math

import torch

from .eskf import (quaternion_exp, quaternion_matrix, quaternion_multiply,
                  normalize_quaternion, right_jacobian, skew)


def _predict(x, P, acceleration, gyroscope, variance, dt, gravity, bias_density):
    position, velocity, q, ba, bg = x.split((3, 3, 4, 3, 3))
    rotation = quaternion_matrix(q)
    phi = (gyroscope-bg)*dt
    jr = right_jacobian(phi)
    increment = quaternion_exp(phi)
    # R(conjugate(q)) == R(q).T. Avoid concatenating a scalar and a negated
    # slice before normalization: PyTorch 2.5's CUDA inference compiler emits
    # an invalid Triton scalar broadcast for that expression.
    inverse_rotation = quaternion_matrix(increment).T
    attitude = -rotation @ skew(acceleration-ba)
    eye = torch.eye(3, dtype=x.dtype, device=x.device)
    zero = torch.zeros_like(eye)
    dt2 = dt.square()
    F = torch.cat((
        torch.cat((eye, eye*dt, .5*attitude*dt2, -.5*rotation*dt2, zero), 1),
        torch.cat((zero, eye, attitude*dt, -rotation*dt, zero), 1),
        torch.cat((zero, zero, inverse_rotation, zero, -jr*dt), 1),
        torch.cat((zero, zero, zero, eye, zero), 1),
        torch.cat((zero, zero, zero, zero, eye), 1)), 0)
    G = torch.cat((
        torch.cat((-.5*rotation*dt, zero, zero, zero), 1),
        torch.cat((-rotation, zero, zero, zero), 1),
        torch.cat((zero, -jr, zero, zero), 1),
        torch.cat((zero, zero, eye, zero), 1),
        torch.cat((zero, zero, zero, eye), 1)), 0)
    diagonal = torch.cat((variance*dt2, bias_density.square()*dt))
    predicted = F @ P @ F.T + (G*diagonal[None, :]) @ G.T
    world = rotation @ (acceleration-ba) + gravity
    next_x = torch.cat((position + velocity*dt + .5*world*dt2,
                        velocity + world*dt,
                        normalize_quaternion(quaternion_multiply(q, increment)), ba, bg))
    return next_x, .5*(predicted+predicted.T)


def _update(x, P, residual, H, R, jitter, eye15):
    S = H @ P @ H.T + R
    S = .5*(S+S.T)
    # The checked CUDA Cholesky synchronizes the host at every update. Collect
    # factorization status on-device and check it once at the end of the flight.
    L, info = torch.linalg.cholesky_ex(
        S+jitter*torch.eye(S.shape[0], dtype=S.dtype, device=S.device), check_errors=False)
    K = torch.cholesky_solve((P @ H.T).T, L).T
    delta = K @ residual
    position, velocity, q, ba, bg = x.split((3, 3, 4, 3, 3))
    next_x = torch.cat((position+delta[:3], velocity+delta[3:6],
                       normalize_quaternion(quaternion_multiply(q, quaternion_exp(delta[6:9]))),
                       ba+delta[9:12], bg+delta[12:15]))
    A = eye15-K @ H
    updated = A @ P @ A.T + K @ R @ K.T
    eye3 = torch.eye(3, dtype=P.dtype, device=P.device)
    reset = torch.block_diag(eye3, eye3, right_jacobian(delta[6:9]), eye3, eye3)
    updated = reset @ updated @ reset.T
    return next_x, .5*(updated+updated.T), info == 0


def _aided(x, P, acceleration, gyroscope, variance, dt, observation,
           gravity, bias_density, position_R, velocity_R, yaw_R, jitter, eye15):
    x, P = _predict(x, P, acceleration, gyroscope, variance, dt, gravity, bias_density)
    x, P, position_valid = _update(x, P, observation[:2]-x[:2], eye15[:2], position_R, jitter, eye15)
    x, P, velocity_valid = _update(x, P, observation[2:4]-x[3:5], eye15[3:5], velocity_R, jitter, eye15)
    rotation = quaternion_matrix(x[6:10])
    rx, ry = rotation[0, 0], rotation[1, 0]
    denominator = rx.square()+ry.square()
    valid = denominator >= 1e-12
    denominator = denominator.clamp_min(1e-12)
    jacobian = torch.stack((rx*0,
        (ry*rotation[0, 2]-rx*rotation[1, 2])/denominator,
        (rx*rotation[1, 1]-ry*rotation[0, 1])/denominator))
    zero = torch.zeros(3, dtype=x.dtype, device=x.device)
    H = torch.cat((zero, zero, jacobian, zero, zero)).reshape(1, 15)
    difference = observation[4]-torch.atan2(ry, rx)
    residual = torch.atan2(torch.sin(difference), torch.cos(difference)).reshape(1)
    x, P, yaw_valid = _update(x, P, residual, H, yaw_R, jitter, eye15)
    return x, P, torch.stack((valid, position_valid & velocity_valid & yaw_valid))


def _height(x, P, acceleration, gyroscope, variance, dt, observation,
            gravity, bias_density, height_R, jitter, eye15):
    x, P = _predict(x, P, acceleration, gyroscope, variance, dt, gravity, bias_density)
    x, P, valid = _update(x, P, observation.reshape(1)-x[2:3], eye15[2:3],
                         height_R, jitter, eye15)
    return x, P, torch.stack((torch.ones_like(valid), valid))


def _aided_height(x, P, acceleration, gyroscope, variance, dt, observation,
                  gravity, bias_density, position_R, velocity_R, yaw_R, height_R, jitter, eye15):
    x, P, valid = _aided(x, P, acceleration, gyroscope, variance, dt, observation,
                         gravity, bias_density, position_R, velocity_R, yaw_R, jitter, eye15)
    x, P, height_valid = _update(x, P, observation[5:6]-x[2:3], eye15[2:3],
                                height_R, jitter, eye15)
    return x, P, valid & torch.stack((torch.ones_like(height_valid), height_valid))


@lru_cache(maxsize=1)
def kernels():
    # AOTAutograd compiles the backward as well as the forward. No CUDA graphs:
    # their reusable output storage conflicts with a full recurrent history.
    return (torch.compile(_aided, fullgraph=True, dynamic=False),
            torch.compile(_predict, fullgraph=True, dynamic=False),
            torch.compile(_height, fullgraph=True, dynamic=False),
            torch.compile(_aided_height, fullgraph=True, dynamic=False))


def aligned_observations(ekf, flight):
    """Return fixed [position xy, velocity xy, yaw, height] per estimate step.

    Reject an unsupported schedule instead of silently dropping observations.
    Preparation uses CPU timestamps, keeping scheduling outside the GPU loop.
    """
    if ekf.observation_types != ('position', 'velocity_xy', 'yaw'):
        raise ValueError('Compiled replay requires position, velocity_xy, yaw in that order.')
    if flight['imu_time'].requires_grad:
        raise ValueError('IMU timestamps must be fixed.')
    times = flight['imu_time'].detach().cpu()
    cutoff = flight['cutoff_timestamp']
    if (times.ndim != 1 or len(times) < 2 or not bool(torch.isfinite(times).all())
            or not bool((times[1:] > times[:-1]).all())
            or not math.isfinite(cutoff) or cutoff <= float(times[0])):
        raise ValueError('Compiled replay requires increasing finite IMU timestamps and a valid cutoff.')
    for name in ('raw_acceleration', 'raw_gyroscope'):
        values = flight[name]
        if values.shape != (len(times), 3) or not bool(torch.isfinite(values).all()):
            raise ValueError(f'{name} must be a finite [N,3] array.')
    aided_indices = [i for i, t in enumerate(times.tolist()) if i and t < cutoff]
    relevant = []
    previous, seen = -math.inf, set()
    for event in flight['vicon_events']:
        timestamp = float(event.timestamp)
        key = timestamp, event.observation_type
        if not math.isfinite(timestamp) or timestamp < previous or key in seen:
            raise ValueError('Observation timestamps must be ordered and channel/timestamp pairs unique.')
        previous = timestamp
        seen.add(key)
        if event.observation_type == 'height':
            if not ekf.height_updates:
                raise ValueError('Height observations require height_updates=True.')
            if float(times[0]) <= timestamp <= float(times[-1]):
                relevant.append(event)
        elif float(times[0]) <= timestamp < cutoff:
            relevant.append(event)
    prepared = ekf._prepare_observations(relevant)
    events = [event for event in prepared if event.observation_type != 'height']
    heights = [event for event in prepared if event.observation_type == 'height']
    if ekf.height_updates:
        if len(heights) != len(times)-1 or any(
                float(event.timestamp) != timestamp
                for event, timestamp in zip(heights, times[1:].tolist())):
            raise ValueError('Compiled height updates require one aligned height event per estimate after initialization.')
    height_by_time = {float(event.timestamp): event.observation.reshape(1) for event in heights}
    if len(events) != 3*len(aided_indices):
        raise ValueError('Compiled replay requires three aligned events per aided timestamp.')
    observations = []
    for offset, i in enumerate(aided_indices):
        group = events[3*offset:3*offset+3]
        if tuple(e.observation_type for e in group) != ekf.observation_types or any(
                float(e.timestamp) != float(times[i]) for e in group):
            raise ValueError('Compiled replay requires Vicon events aligned to the IMU timestamps.')
        observations.append(torch.cat((group[0].observation[:2], group[1].observation[:2],
                                       group[2].observation.reshape(1),
                                       height_by_time[float(times[i])] if ekf.height_updates
                                       else ekf.gravity.new_zeros(1))))
    post_times = [t for t in times[1:].tolist() if t >= cutoff]
    if ekf.height_updates:
        observations.extend(torch.cat((ekf.gravity.new_zeros(5), height_by_time[timestamp]))
                            for timestamp in post_times)
    else:
        observations.extend(ekf.gravity.new_zeros(6) for _ in post_times)
    return torch.stack(observations)


def velocity_sequence(ekf, flight, variance):
    """Exact full-history velocity replay, with one compiled call per sample."""
    aided, predict, height, aided_height = kernels()
    device = ekf.gravity.device
    if (variance.dtype != torch.float64 or variance.device != device
            or variance.shape not in ((len(flight['imu_time']), 6), (len(flight['imu_time'])-1, 6))
            or not bool(torch.isfinite(variance).all()) or bool((variance < 0).any())):
        raise ValueError('IMU variances must be finite nonnegative Float64 [N,6] or [N-1,6] arrays on the filter device.')
    acc = flight['raw_acceleration'].to(device)
    gyro = flight['raw_gyroscope'].to(device)
    times = flight['imu_time'].detach().cpu()
    dt = (times[1:]-times[:-1]).to(device).unbind(0)
    acc_rows, gyro_rows, variance_rows = acc.unbind(0), gyro.unbind(0), variance.unbind(0)
    policy = ekf.observation_types, ekf.height_updates
    if flight.get('_compiled_observation_policy') != policy:
        flight['_compiled_observations'] = aligned_observations(ekf, flight).detach()
        flight['_compiled_observation_policy'] = policy
    observation_rows = flight['_compiled_observations'].to(device).unbind(0)
    initial = flight['initial_state']
    x = torch.cat((initial.position, initial.velocity, initial.quaternion,
                   initial.accelerometer_bias, initial.gyroscope_bias)).to(device)
    P = flight['initial_covariance'].to(device)
    # Consistent differentiability avoids recompiling the first recurrent step.
    if torch.is_grad_enabled():
        x = x.detach().requires_grad_(True)
        P = P.detach().requires_grad_(True)
    velocities = [x[3:6]]
    valid = []
    cutoff = flight['cutoff_timestamp']
    schedule = times.tolist()
    for i in range(len(dt)):
        if schedule[i+1] < cutoff:
            arguments = (x, P, acc_rows[i], gyro_rows[i], variance_rows[i], dt[i],
                         observation_rows[i], ekf.gravity, ekf.bias_noise_density,
                         ekf.position_R, ekf.velocity_xy_R, ekf.yaw_R)
            if ekf.height_updates:
                x, P, yaw_valid = aided_height(*arguments, ekf.height_R, ekf.innovation_jitter, ekf._eye15)
            else:
                x, P, yaw_valid = aided(*arguments, ekf.innovation_jitter, ekf._eye15)
            valid.append(yaw_valid)
        elif ekf.height_updates:
            x, P, height_valid = height(x, P, acc_rows[i], gyro_rows[i], variance_rows[i], dt[i],
                observation_rows[i][5], ekf.gravity, ekf.bias_noise_density,
                ekf.height_R, ekf.innovation_jitter, ekf._eye15)
            valid.append(height_valid)
        else:
            x, P = predict(x, P, acc_rows[i], gyro_rows[i], variance_rows[i], dt[i],
                           ekf.gravity, ekf.bias_noise_density)
        velocities.append(x[3:6])
    if valid:
        validity = torch.stack(valid).all(0).tolist()
        if not validity[0]:
            raise ValueError('Yaw is undefined for a vertical body x-axis.')
        if not validity[1]:
            raise torch.linalg.LinAlgError('ESKF innovation covariance is not positive definite.')
    result = torch.stack(velocities)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('ESKF replay produced non-finite velocities.')
    return result
