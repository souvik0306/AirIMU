"""Float64 right-multiplicative ESKF, with no trainable filter parameters.

Quaternions are Hamilton wxyz, rotating body vectors into the navigation frame.
The attitude error is R_true = R_nominal Exp(delta_theta). Error coordinates are
[position, velocity, attitude, accel bias, gyro bias]. IMU columns are [acc, gyro]
in SI units; timestamps are seconds in one shared reference. No frame conversion,
gravity removal, learned IMU correction, or external bias subtraction is applied.
"""

from dataclasses import dataclass, fields, replace
import math
from typing import Mapping

import torch
from torch import Tensor, nn


def skew(v: Tensor) -> Tensor:
    x, y, z = v.unbind()
    zero = x * 0
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)


def quaternion_multiply(a: Tensor, b: Tensor) -> Tensor:
    aw, av = a[:1], a[1:]
    bw, bv = b[:1], b[1:]
    return torch.cat((aw * bw - (av * bv).sum().reshape(1),
                      aw * bv + bw * av + torch.linalg.cross(av, bv)))


def quaternion_exp(phi: Tensor) -> Tensor:
    # Safe inactive branches also matter to autograd at exactly zero rotation.
    squared = phi.square().sum()
    angle = squared.clamp_min(1e-16).sqrt()
    small = squared < 1e-8
    real = torch.where(small, 1 - squared / 8 + squared.square() / 384,
                       torch.cos(angle / 2))
    scale = torch.where(small, .5 - squared / 48 + squared.square() / 3840,
                        torch.sin(angle / 2) / angle)
    return torch.cat((real.reshape(1), scale * phi))


def normalize_quaternion(q: Tensor) -> Tensor:
    return q / torch.linalg.vector_norm(q)


def quaternion_matrix(q: Tensor) -> Tensor:
    w, x, y, z = normalize_quaternion(q).unbind()
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    )).reshape(3, 3)


def right_jacobian(phi: Tensor) -> Tensor:
    squared = phi.square().sum()
    angle = squared.clamp_min(1e-16).sqrt()
    small = squared < 1e-8
    a = torch.where(small, .5 - squared / 24 + squared.square() / 720,
                    (1 - torch.cos(angle)) / angle.square())
    b = torch.where(small, 1/6 - squared / 120 + squared.square() / 5040,
                    (angle - torch.sin(angle)) / angle.pow(3))
    cross = skew(phi)
    return torch.eye(3, dtype=phi.dtype, device=phi.device) - a * cross + b * cross @ cross


def quaternion_yaw(q: Tensor) -> Tensor:
    rotation = quaternion_matrix(q)
    return torch.atan2(rotation[1, 0], rotation[0, 0])


@dataclass(frozen=True)
class EKFState:
    position: Tensor
    velocity: Tensor
    quaternion: Tensor
    accelerometer_bias: Tensor
    gyroscope_bias: Tensor


@dataclass(frozen=True)
class ViconEvent:
    timestamp: float
    observation_type: str
    observation: Tensor


def _symmetric(matrix: Tensor) -> Tensor:
    return .5 * (matrix + matrix.T)


def _tensor(value, reference: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=torch.float64, device=reference.device)


def _finite_shape(value: Tensor, shape, name: str):
    if tuple(value.shape) != tuple(shape) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite with shape {tuple(shape)}.")


class DifferentiableESKF(nn.Module):
    """Fixed 15-state ESKF with horizontal Vicon aiding defaults.

    Scalar gravity selects z-up navigation: [0, 0, -gravity]. For NED pass
    an explicit vector, e.g. gravity=(0, 0, 9.80665), with matching body/pose
    conventions. Position fusion uses xy. Optional height updates use z only
    before and after cutoff, independently of the pre-cutoff Vicon channels.
    ``velocity`` enables xyz velocity; ``velocity_xy`` enables horizontal only.
    """

    def __init__(self, gravity=9.80665, state_dim=15, dtype=torch.float64,
                 observation_types=("position", "velocity_xy", "yaw"),
                 height_updates=False):
        super().__init__()
        if state_dim != 15 or dtype != torch.float64:
            raise ValueError("The ESKF requires state_dim=15 and dtype=torch.float64.")
        allowed = {"position", "velocity_xy", "velocity", "yaw"}
        if (not observation_types or len(set(observation_types)) != len(observation_types)
                or not set(observation_types) <= allowed
                or {"velocity", "velocity_xy"} <= set(observation_types)):
            raise ValueError("Select unique supported Vicon channels and one velocity model.")
        self.observation_types = tuple(observation_types)
        self.height_updates = bool(height_updates)
        gravity_vector = [0., 0., -float(gravity)] if isinstance(gravity, (int, float)) else gravity
        g = torch.as_tensor(gravity_vector, dtype=dtype).detach().clone()
        _finite_shape(g, (3,), "gravity")
        self.register_buffer("gravity", g)
        self.register_buffer("prior_std", torch.tensor(
            [.10]*3 + [.20]*3 + [math.radians(5)]*3 + [.10]*3 + [.020]*3, dtype=dtype))
        self.register_buffer("bias_noise_density", torch.tensor([.002]*3 + [.0002]*3, dtype=dtype))
        self.register_buffer("position_R", torch.eye(2, dtype=dtype) * .01**2)
        self.register_buffer("velocity_xy_R", torch.eye(2, dtype=dtype) * .05**2)
        self.register_buffer("velocity_R", torch.eye(3, dtype=dtype) * .05**2)
        self.register_buffer("yaw_R", torch.tensor([[math.radians(2)**2]], dtype=dtype))
        self.register_buffer("height_R", torch.tensor([[.1**2]], dtype=dtype))
        self.register_buffer("innovation_jitter", torch.tensor(1e-9, dtype=dtype))
        self.register_buffer("_eye15", torch.eye(15, dtype=dtype), persistent=False)

    def initialize(self, initial_position, initial_velocity, initial_quaternion,
                   initial_accel_bias=None, initial_gyro_bias=None, initial_covariance=None):
        """Fixed initial pose and velocity, never reused at cutoff."""
        values = (initial_position, initial_velocity, initial_quaternion,
                  [0.]*3 if initial_accel_bias is None else initial_accel_bias,
                  [0.]*3 if initial_gyro_bias is None else initial_gyro_bias)
        # Initialization is deliberately independent of training parameters.
        tensors = [_tensor(value, self.gravity).detach().clone() for value in values]
        for value, size, name in zip(tensors, (3, 3, 4, 3, 3),
                                     (field.name for field in fields(EKFState))):
            _finite_shape(value, (size,), name)
        if bool(tensors[2].norm() < 1e-12):
            raise ValueError("Initial quaternion must have nonzero norm.")
        tensors[2] = normalize_quaternion(tensors[2])
        covariance = (torch.diag(self.prior_std.square()) if initial_covariance is None
                      else _tensor(initial_covariance, self.gravity).detach().clone())
        _finite_shape(covariance, (15, 15), "initial_covariance")
        if (not torch.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10)
                or bool(torch.linalg.eigvalsh(covariance).min() < -1e-12)):
            raise ValueError("Initial covariance must be symmetric positive semidefinite.")
        return EKFState(*tensors), covariance

    def propagate_nominal_state(self, state, acceleration, angular_velocity, dt):
        dt = _tensor(dt, self.gravity)
        acceleration_world = quaternion_matrix(state.quaternion) @ (
            acceleration - state.accelerometer_bias) + self.gravity
        increment = quaternion_exp((angular_velocity - state.gyroscope_bias) * dt)
        return EKFState(
            state.position + state.velocity * dt + .5 * acceleration_world * dt.square(),
            state.velocity + acceleration_world * dt,
            normalize_quaternion(quaternion_multiply(state.quaternion, increment)),
            state.accelerometer_bias, state.gyroscope_bias,
        )

    def compute_state_transition(self, state, acceleration, angular_velocity, dt):
        """Exact first derivative of the discrete nominal propagation used here."""
        dt = _tensor(dt, self.gravity)
        rotation = quaternion_matrix(state.quaternion)
        phi = (angular_velocity - state.gyroscope_bias) * dt
        attitude_acceleration = -rotation @ skew(acceleration - state.accelerometer_bias)
        eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
        zero = torch.zeros_like(eye)
        rows = (
            (eye, eye*dt, .5*attitude_acceleration*dt.square(), -.5*rotation*dt.square(), zero),
            (zero, eye, attitude_acceleration*dt, -rotation*dt, zero),
            (zero, zero, quaternion_matrix(quaternion_exp(-phi)), zero, -right_jacobian(phi)*dt),
            (zero, zero, zero, eye, zero),
            (zero, zero, zero, zero, eye),
        )
        return torch.cat([torch.cat(row, dim=1) for row in rows], dim=0)

    def compute_noise_jacobian(self, state, acceleration, angular_velocity, dt):
        dt = _tensor(dt, self.gravity)
        rotation = quaternion_matrix(state.quaternion)
        phi = (angular_velocity - state.gyroscope_bias) * dt
        eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
        zero = torch.zeros_like(eye)
        # Noise coordinates: delta-v body, delta-theta body, delta-ba, delta-bg.
        rows = ((-.5*rotation*dt, zero, zero, zero),
                (-rotation, zero, zero, zero),
                (zero, -right_jacobian(phi), zero, zero),
                (zero, zero, eye, zero), (zero, zero, zero, eye))
        return torch.cat([torch.cat(row, dim=1) for row in rows], dim=0)

    def build_discrete_process_noise(self, imu_variance, dt):
        # Input is per-sample measurement variance, NOT continuous PSD or std.
        dt = _tensor(dt, self.gravity)
        imu_variance = _tensor(imu_variance, self.gravity)
        return torch.diag(torch.cat((imu_variance * dt.square(),
                                     self.bias_noise_density.square() * dt)))

    def predict_covariance(self, state, P, acceleration, angular_velocity, imu_variance, dt):
        F = self.compute_state_transition(state, acceleration, angular_velocity, dt)
        G = self.compute_noise_jacobian(state, acceleration, angular_velocity, dt)
        Qd = self.build_discrete_process_noise(imu_variance, dt)
        return _symmetric(F @ P @ F.T + G @ Qd @ G.T)

    def predict(self, state, covariance, raw_acceleration, raw_gyroscope, imu_variance, dt):
        """Nominal propagation and Jacobians both use the SAME prior state."""
        self._check_precision(state, covariance)
        dt = _tensor(dt, covariance)
        if dt.ndim != 0 or not bool(torch.isfinite(dt)) or not bool(dt > 0):
            raise ValueError("dt must be a finite positive scalar in seconds.")
        acceleration = _tensor(raw_acceleration, covariance)
        gyroscope = _tensor(raw_gyroscope, covariance)
        variance = _tensor(imu_variance, covariance)
        if acceleration.shape != (3,) or gyroscope.shape != (3,) or variance.shape != (6,):
            raise ValueError("Prediction requires acceleration[3], gyroscope[3], and variance[6].")
        if not bool(torch.isfinite(variance).all()) or bool((variance < 0).any()):
            raise ValueError("IMU variances must be finite and nonnegative.")
        return self._predict(state, covariance, acceleration, gyroscope, variance, dt)

    def _predict(self, state, covariance, acceleration, gyroscope, variance, dt):
        """Prediction after run_sequence has validated the complete input arrays."""
        P_pred = self.predict_covariance(state, covariance, acceleration, gyroscope, variance, dt)
        return self.propagate_nominal_state(state, acceleration, gyroscope, dt), P_pred

    def measurement_model(self, state, observation, observation_type, *, yaw_validity=None):
        if observation_type == "height":
            if not self.height_updates:
                raise ValueError("Height updates are not enabled.")
            observation = _tensor(observation, state.position)
            if observation.numel() != 1:
                raise ValueError("Height observation must be a scalar z position.")
            return observation.reshape(1) - state.position[2:3], self._eye15[2:3]
        if observation_type not in self.observation_types:
            raise ValueError(f"Vicon channel {observation_type!r} is not enabled.")
        observation = _tensor(observation, state.position)
        eye = self._eye15
        if observation_type == "position":
            if observation.shape not in ((2,), (3,)):
                raise ValueError("Position observation must contain xy or xyz (z is ignored).")
            return observation[:2] - state.position[:2], eye[:2]
        if observation_type in ("velocity", "velocity_xy"):
            count = 3 if observation_type == "velocity" else 2
            if observation.shape not in ((count,), (3,)):
                raise ValueError(f"{observation_type} observation has the wrong shape.")
            return observation[:count] - state.velocity[:count], eye[3:3+count]
        if observation.shape == (4,):
            if bool(observation.norm() < 1e-12):
                raise ValueError("Vicon quaternion must have nonzero norm.")
            yaw = quaternion_yaw(observation)
        elif observation.numel() == 1:
            yaw = observation.reshape(())
        else:
            raise ValueError("Yaw observation must be a scalar angle or wxyz quaternion.")
        rotation = quaternion_matrix(state.quaternion)
        x, y = rotation[0, 0], rotation[1, 0]
        denominator = x.square() + y.square()
        if yaw_validity is None:
            if bool(denominator < 1e-12):
                raise ValueError("Yaw is undefined for a vertical body x-axis.")
        else:
            # Delay the device-to-host check until replay ends. Keep invalid
            # denominators finite so the same domain error can be reported then.
            yaw_validity.append(denominator >= 1e-12)
            denominator = denominator.clamp_min(1e-12)
        jacobian = torch.stack((x*0,
            (y*rotation[0, 2] - x*rotation[1, 2]) / denominator,
            (x*rotation[1, 1] - y*rotation[0, 1]) / denominator))
        zero = torch.zeros(3, dtype=x.dtype, device=x.device)
        H = torch.cat((zero, zero, jacobian, zero, zero)).reshape(1, 15)
        difference = yaw - quaternion_yaw(state.quaternion)
        return torch.atan2(torch.sin(difference), torch.cos(difference)).reshape(1), H

    @staticmethod
    def inject_error_state(state, delta_x):
        return EKFState(state.position + delta_x[:3], state.velocity + delta_x[3:6],
            normalize_quaternion(quaternion_multiply(state.quaternion, quaternion_exp(delta_x[6:9]))),
            state.accelerometer_bias + delta_x[9:12], state.gyroscope_bias + delta_x[12:15])

    @staticmethod
    def reset_error_state(corrected_state, covariance, delta_x):
        eye = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
        reset = torch.block_diag(eye, eye, right_jacobian(delta_x[6:9]), eye, eye)
        return corrected_state, _symmetric(reset @ covariance @ reset.T)

    def update_vicon(self, state, covariance, observation, observation_type):
        self._check_precision(state, covariance)
        return self._update_vicon(state, covariance, observation, observation_type)

    def _update_vicon(self, state, covariance, observation, observation_type, *, validated=False,
                      yaw_validity=None):
        residual, H = self.measurement_model(state, observation, observation_type,
                                             yaw_validity=yaw_validity)
        if not validated and not bool(torch.isfinite(residual).all()):
            raise ValueError("Vicon observation must be finite.")
        R = getattr(self, observation_type + "_R")
        S = _symmetric(H @ covariance @ H.T + R)
        L = torch.linalg.cholesky(S + self.innovation_jitter * torch.eye(
            S.shape[0], dtype=S.dtype, device=S.device))
        K = torch.cholesky_solve((covariance @ H.T).T, L).T
        delta = K @ residual
        corrected = self.inject_error_state(state, delta)
        A = self._eye15 - K @ H
        updated = A @ covariance @ A.T + K @ R @ K.T
        return self.reset_error_state(corrected, updated, delta)

    def _check_precision(self, state, covariance):
        tensors = [self.gravity, covariance] + [getattr(state, field.name) for field in fields(EKFState)]
        if any(value.dtype != torch.float64 or value.device != covariance.device for value in tensors):
            raise ValueError("State, covariance, and filter must share a device and use Float64.")

    def _prepare_observations(self, events):
        """Transfer observations in batches, and compute fixed quaternion yaw once."""
        groups = {}
        for index, event in enumerate(events):
            observation = torch.as_tensor(event.observation, dtype=torch.float64)
            shape = tuple(observation.shape)
            channel = event.observation_type
            if channel == "position":
                valid = shape in ((2,), (3,))
            elif channel in ("velocity", "velocity_xy"):
                valid = shape in ((3,),) if channel == "velocity" else shape in ((2,), (3,))
            elif channel == "height":
                valid = observation.numel() == 1
            else:
                valid = shape == (4,) or observation.numel() == 1
            if not valid:
                raise ValueError(f"{channel} observation has the wrong shape.")
            groups.setdefault((channel, shape, observation.device), []).append((index, observation))
        prepared = list(events)
        for (channel, shape, _), entries in groups.items():
            values = torch.stack([observation for _, observation in entries]).to(self.gravity.device)
            if not bool(torch.isfinite(values).all()):
                raise ValueError("Vicon observation must be finite.")
            if channel == "yaw" and shape == (4,):
                norms = torch.linalg.vector_norm(values, dim=-1)
                if bool((norms < 1e-12).any()):
                    raise ValueError("Vicon quaternion must have nonzero norm.")
                w, x, y, z = (values / norms[:, None]).unbind(-1)
                values = torch.atan2(2 * (x*y + w*z), 1 - 2 * (y*y + z*z))
            for (index, _), observation in zip(entries, values.unbind(0)):
                event = events[index]
                prepared[index] = ViconEvent(event.timestamp, channel, observation)
        return prepared

    def run_sequence(self, imu, imu_time, imu_variance, vicon_events, cutoff_timestamp,
                     initial_state, initial_covariance, *, return_covariance=True):
        """Replay one chronological flight with zero-order-held raw IMU.

        imu is [N,6] (acc then gyro), or a mapping with ``acc``/``gyro``.
        Variances are [N,6] or [N-1,6]; the final sample has no propagation
        interval. Each subinterval uses its actual dt for increment noise.
        Output includes the initial state and has one row per IMU timestamp.
        Timestamp scheduling is fixed and must not require gradients.
        With return_covariance=False, omit covariance history only. Covariance
        propagation and gradients continue through every replayed interval.
        When enabled, height observations are fused before and after cutoff;
        all other Vicon observations are fused strictly before cutoff.
        """
        if self.gravity.dtype != torch.float64:
            raise ValueError("Filter buffers must remain Float64.")
        if isinstance(imu, Mapping):
            imu = torch.cat((imu["acc"], imu["gyro"]), dim=-1)
        imu = _tensor(imu, self.gravity)
        times = _tensor(imu_time, self.gravity)
        variance = _tensor(imu_variance, self.gravity)
        if times.ndim != 1 or len(times) < 2 or times.requires_grad:
            raise ValueError("At least two fixed one-dimensional IMU timestamps are required.")
        _finite_shape(times, (len(times),), "imu_time")
        _finite_shape(imu, (len(times), 6), "imu")
        if variance.shape not in ((len(times), 6), (len(times)-1, 6)):
            raise ValueError("imu_variance must have shape [N,6] or [N-1,6].")
        if not bool(torch.isfinite(variance).all()) or bool((variance < 0).any()):
            raise ValueError("IMU variances must be finite and nonnegative.")
        if not bool(((times[1:] - times[:-1]) > 0).all()):
            raise ValueError("IMU timestamps must be strictly increasing.")
        if isinstance(cutoff_timestamp, Tensor) and cutoff_timestamp.requires_grad:
            raise ValueError("Cutoff timestamp must be fixed.")
        cutoff = float(cutoff_timestamp)
        schedule = times.tolist()
        if not math.isfinite(cutoff) or cutoff <= schedule[0]:
            raise ValueError("Cutoff must be finite and later than initialization.")
        state = replace(initial_state, **{
            field.name: _tensor(getattr(initial_state, field.name), self.gravity)
            for field in fields(EKFState)})
        P = _tensor(initial_covariance, self.gravity)
        _finite_shape(P, (15, 15), "initial_covariance")
        self._check_precision(state, P)
        # Accept already initialized differentiable states without detaching.
        for field in fields(EKFState):
            _finite_shape(getattr(state, field.name),
                          (4 if field.name == "quaternion" else 3,), field.name)
        if bool(state.quaternion.norm() < 1e-12):
            raise ValueError("Initial quaternion must have nonzero norm.")
        events = []
        previous_time = -math.inf
        seen = set()
        for event in vicon_events:
            event = ViconEvent(**event) if isinstance(event, Mapping) else event
            if isinstance(event.timestamp, Tensor) and event.timestamp.requires_grad:
                raise ValueError("Vicon timestamps must be fixed.")
            timestamp = float(event.timestamp)
            if not math.isfinite(timestamp) or timestamp < previous_time:
                raise ValueError("Vicon events must have finite, nondecreasing timestamps.")
            key = (timestamp, event.observation_type)
            if key in seen:
                raise ValueError("Repeated Vicon channel at the same timestamp.")
            seen.add(key)
            previous_time = timestamp
            if event.observation_type == "height":
                if not self.height_updates:
                    raise ValueError("Height events require height_updates=True.")
                if schedule[0] <= timestamp <= schedule[-1]:
                    events.append(event)
            elif schedule[0] <= timestamp < cutoff and timestamp <= schedule[-1]:
                if event.observation_type not in self.observation_types:
                    raise ValueError(f"Unexpected Vicon channel: {event.observation_type}.")
                events.append(event)
        if (self.height_updates
                and not any(event.observation_type == "height" for event in events)):
            raise ValueError("Height updates are enabled but no height events were supplied.")
        if (self.height_updates and schedule[-1] >= cutoff
                and not any(event.observation_type == "height" and float(event.timestamp) >= cutoff
                            for event in events)):
            raise ValueError("Height updates are enabled but no post-cutoff height events were supplied.")
        events = self._prepare_observations(events)
        boundaries = sorted(set(schedule).union(float(event.timestamp) for event in events))
        durations = sorted({end-start for start, end in zip(boundaries, boundaries[1:])})
        # Copy all integration intervals once rather than creating a CUDA scalar
        # from a Python float at every IMU step (and in each prediction helper).
        device_durations = dict(zip(durations, times.new_tensor(durations).unbind()))
        history = {field.name: [] for field in fields(EKFState)}
        covariance_history = []
        yaw_validity = []
        event_index = 0

        def fuse_at(timestamp, current_state, current_P):
            nonlocal event_index
            while event_index < len(events) and float(events[event_index].timestamp) == timestamp:
                event = events[event_index]
                current_state, current_P = self._update_vicon(
                    current_state, current_P, event.observation, event.observation_type,
                    validated=True, yaw_validity=yaw_validity)
                event_index += 1
            return current_state, current_P

        def record():
            for field in fields(EKFState):
                history[field.name].append(getattr(state, field.name))
            if return_covariance:
                covariance_history.append(P)

        def propagate(current_state, current_P, sample, start, end):
            dt = device_durations[end - start]
            return self._predict(current_state, current_P, imu[sample, :3],
                                 imu[sample, 3:], variance[sample], dt)

        state, P = fuse_at(schedule[0], state, P)
        record()
        for k, end in enumerate(schedule[1:]):
            current_time = schedule[k]
            while event_index < len(events) and float(events[event_index].timestamp) <= end:
                event_time = float(events[event_index].timestamp)
                if event_time > current_time:
                    state, P = propagate(state, P, k, current_time, event_time)
                state, P = fuse_at(event_time, state, P)
                current_time = event_time
            if end > current_time:
                state, P = propagate(state, P, k, current_time, end)
            record()
        result = {key: torch.stack(values) for key, values in history.items()}
        if return_covariance:
            result["covariance"] = torch.stack(covariance_history)
        if yaw_validity and not bool(torch.stack(yaw_validity).all()):
            raise ValueError("Yaw is undefined for a vertical body x-axis.")
        if any(not bool(torch.isfinite(value).all()) for value in result.values()):
            raise ValueError("ESKF replay produced a non-finite trajectory.")
        return result

    def forward(self, *args, **kwargs):
        return self.run_sequence(*args, **kwargs)
