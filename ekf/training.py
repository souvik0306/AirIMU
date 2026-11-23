"""AirIMU covariance-only training through an entire aided/unaided replay."""

from copy import deepcopy
import math

import torch
from torch import nn

from .eskf import DifferentiableESKF


def horizontal_velocity_residual(velocity_est, vicon_velocity_gt, evaluation_mask):
    """Validate and select the horizontal velocity error used by loss and metrics."""
    if (velocity_est.ndim != 2 or velocity_est.shape[-1] != 3
            or vicon_velocity_gt.shape != velocity_est.shape):
        raise ValueError("Estimated and ground-truth velocity must both have shape [N,3].")
    if evaluation_mask.dtype != torch.bool or evaluation_mask.shape != velocity_est.shape[:1]:
        raise ValueError("evaluation_mask must be a boolean vector with shape [N].")
    if not bool(evaluation_mask.any()):
        raise ValueError("The velocity evaluation interval contains no samples.")
    residual = velocity_est[evaluation_mask, :2] - vicon_velocity_gt[evaluation_mask, :2]
    if not bool(torch.isfinite(residual).all()):
        raise ValueError("Evaluation velocities must be finite.")
    return residual


def horizontal_velocity_loss(velocity_est, vicon_velocity_gt, evaluation_mask):
    """Mean squared horizontal speed-vector error (sum xy, then mean time)."""
    residual = horizontal_velocity_residual(
        velocity_est, vicon_velocity_gt, evaluation_mask
    )
    return residual.square().sum(dim=-1).mean()


def velocity_evaluation_mask(imu_time, cutoff_timestamp):
    """Require coverage of [cutoff+5, cutoff+25] seconds, including endpoints."""
    start, end = float(cutoff_timestamp) + 5., float(cutoff_timestamp) + 25.
    if imu_time.ndim != 1 or not len(imu_time):
        raise ValueError("IMU timestamps must be a nonempty vector.")
    if float(imu_time[0]) > start or float(imu_time[-1]) < end:
        raise ValueError("Flight must cover the complete cutoff+5 to cutoff+25 interval.")
    return (imu_time >= start) & (imu_time <= end)


class AirIMUESKF(nn.Module):
    """Train CodeNet uncertainty and selected encoder layers through raw-IMU EKF.

    A freezes the encoder; B trains GRU2; C trains CNN, GRU1 and GRU2.
    Correction-head weights remain frozen in all configurations.
    Covariance heads retain loaded weights. All filter equations and
    measurement covariances are fixed. Each call replays one full flight from
    initialization, retaining the graph through all aided updates and cutoff.
    """

    def __init__(self, network, ekf=None, *, configuration="A"):
        super().__init__()
        if not network.conf.propcov:
            raise ValueError("AirIMU must have propcov=True for ESKF training.")
        uncertainty_names = ("acccov_decoder", "gyrocov_decoder", "acc_bias_gain", "gyro_bias_gain")
        frozen_names = ("cnn", "gru1", "gru2", "accdecoder", "gyrodecoder")
        if not all(hasattr(network, name) for name in uncertainty_names + frozen_names):
            raise TypeError("ESKF training requires CodeNet's encoder and uncertainty heads.")
        network.requires_grad_(False)
        if configuration not in ("A", "B", "C", "Adapter"):
            raise ValueError("Training configuration must be A, B, C or Adapter.")
        if configuration == "Adapter" and getattr(network, "raw_cov_adapter", None) is None:
            raise ValueError("Adapter training requires raw_cov_adapter=True.")
        self.configuration = configuration
        self.cacheable_features = configuration in ("A", "Adapter")
        # Copy only the teacher's feature/correction modules. The unused
        # PyPose integrator contains LieTensor buffers whose CUDA deepcopy
        # is unsupported; it has no role in correction consistency.
        self.reference_modules = (nn.ModuleDict({
            name: deepcopy(getattr(network, name))
            for name in ("cnn", "gru1", "gru2", "accdecoder", "gyrodecoder")
        }).requires_grad_(False).eval() if configuration in ("B", "C") else None)
        trainable = {"gru2"} if configuration == "B" else {"cnn", "gru1", "gru2"} if configuration == "C" else set()
        for name in trainable:
            getattr(network, name).requires_grad_(True)
        if configuration != "Adapter":
            for name in uncertainty_names:
                getattr(network, name).requires_grad_(True)
        if getattr(network, "raw_cov_adapter", None) is not None:
            network.raw_cov_adapter.requires_grad_(True)
        self._frozen_modules = tuple(getattr(network, name) for name in frozen_names if name not in trainable)
        if configuration == "Adapter":
            self._frozen_modules += tuple(getattr(network, name) for name in uncertainty_names)
        for module in self._frozen_modules:
            module.eval()
        self.network = network
        self.ekf = DifferentiableESKF() if ekf is None else ekf
        if any(parameter.requires_grad for parameter in self.ekf.parameters()):
            raise ValueError("The ESKF must contain no trainable parameters.")

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for module in self._frozen_modules:
                module.eval()
        if self.reference_modules is not None:
            self.reference_modules.eval()
        return self

    def optimizer_groups(self, *, head_lr=1e-5, bias_lr=1e-5, gru2_lr=1e-6, gru1_lr=5e-7, cnn_lr=3e-7,
                         adapter_lr=None):
        groups = []
        for name, modules, lr in (
            ("covariance_heads", (self.network.acccov_decoder, self.network.gyrocov_decoder), head_lr),
            ("bias_gains", (self.network.acc_bias_gain, self.network.gyro_bias_gain), bias_lr),
            ("gru2", (self.network.gru2,), gru2_lr), ("gru1", (self.network.gru1,), gru1_lr),
            ("cnn", (self.network.cnn,), cnn_lr),
            ("raw_cov_adapter", (getattr(self.network, "raw_cov_adapter", None),),
             head_lr if adapter_lr is None else adapter_lr),
        ):
            params = [p for module in modules if module is not None for p in module.parameters() if p.requires_grad]
            if params:
                if not math.isfinite(lr) or lr <= 0:
                    raise ValueError("Learning rates must be positive and finite.")
                groups.append({"params": params, "lr": lr, "name": name})
        return groups

    def extract_features(self, raw_acceleration, raw_gyroscope):
        """Fresh differentiable GRUs for B/C; cacheable frozen features for A."""
        if (raw_acceleration.ndim != 2 or raw_acceleration.shape[-1] != 3
                or raw_gyroscope.shape != raw_acceleration.shape
                or not len(raw_acceleration)):
            raise ValueError("Raw IMU must have matching nonempty [N,3] arrays.")
        reference = next(self.network.parameters())
        acc = raw_acceleration.to(device=reference.device, dtype=reference.dtype)
        gyro = raw_gyroscope.to(device=reference.device, dtype=reference.dtype)
        # Supply nine frames of leading context without altering real samples.
        # Repeating raw samples avoids requiring GT attitude for CNN padding.
        context = self.network.interval
        padded_acc = torch.cat((acc[:1].expand(context, -1), acc), dim=0)[None]
        padded_gyro = torch.cat((gyro[:1].expand(context, -1), gyro), dim=0)[None]
        inputs = torch.cat((padded_acc, padded_gyro), dim=-1)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.cacheable_features):
            features = self.network.encoder(inputs)[:, 1:, :]
            correction = self.network.decoder(features)
        if self.reference_modules is None:
            original_correction = correction.detach()
        else:
            with torch.no_grad():
                teacher = self.reference_modules
                original_features = teacher["cnn"](inputs.transpose(-1, -2)).transpose(-1, -2)
                original_features, _ = teacher["gru1"](original_features)
                original_features, _ = teacher["gru2"](original_features)
                original_correction = torch.cat((
                    teacher["accdecoder"](original_features[:, 1:])*self.network.acc_std,
                    teacher["gyrodecoder"](original_features[:, 1:])*self.network.gyro_std,
                ), dim=-1)
        return {"features": features[0], "correction": correction[0], "samples": len(acc),
                "reference_correction": original_correction[0]}

    def predict_variance_batch(self, cached, *, return_adapter_deltas=False):
        """Decode ragged flight features in one GPU batch, then gather IMU rows.

        Gathering replaces thousands of in-place slice updates and their backward
        nodes. Each flight retains its original CNN/GRU context and sample mapping.
        """
        reference = next(self.network.parameters())
        features = nn.utils.rnn.pad_sequence([item["features"] for item in cached], batch_first=True)
        corrections = nn.utils.rnn.pad_sequence([item["correction"] for item in cached], batch_first=True)
        features = features.to(device=reference.device, dtype=reference.dtype)
        corrections = corrections.to(device=reference.device, dtype=reference.dtype)
        decoded = self.network.cov_decoder(features, corrections,
            return_adapter_delta=True) if return_adapter_deltas else self.network.cov_decoder(features, corrections)
        if return_adapter_deltas:
            decoded, deltas = decoded
        variances = []
        selected_deltas = []
        for index, item in enumerate(cached):
            indices = torch.div(torch.arange(item["samples"], device=reference.device)
                                + self.network.inter_head, self.network.interval,
                                rounding_mode="floor").clamp_max(len(item["features"])-1)
            variance = decoded[index].index_select(0, indices)
            variances.append(variance.to(device=self.ekf.gravity.device, dtype=torch.float64))
            if return_adapter_deltas:
                selected_deltas.append(deltas[index].index_select(0, indices))
        return (variances, selected_deltas) if return_adapter_deltas else variances

    def predict_variance(self, raw_acceleration, raw_gyroscope):
        features = self.extract_features(raw_acceleration, raw_gyroscope)
        return self.predict_variance_batch([features])[0]

    def forward(self, raw_acceleration, raw_gyroscope, imu_time, vicon_events,
                cutoff_timestamp, initial_state, initial_covariance, vicon_velocity_gt,
                *, return_covariance=True, vicon_quaternion_gt=None, vicon_time_gt=None):
        variance = self.predict_variance(raw_acceleration, raw_gyroscope)
        result = self.ekf.run_sequence(
            {"acc": raw_acceleration, "gyro": raw_gyroscope}, imu_time, variance,
            vicon_events, cutoff_timestamp, initial_state, initial_covariance,
            return_covariance=return_covariance)
        times = torch.as_tensor(imu_time, dtype=torch.float64, device=variance.device)
        mask = velocity_evaluation_mask(times, cutoff_timestamp)
        gt = torch.as_tensor(vicon_velocity_gt, dtype=torch.float64, device=variance.device)
        loss = horizontal_velocity_loss(result["velocity"], gt, mask)
        return {"loss": loss, "trajectory": result, "imu_variance": variance,
                "evaluation_mask": mask}
