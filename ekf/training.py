"""AirIMU covariance-only training through an entire aided/unaided replay."""

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
    """Train existing CodeNet uncertainty heads; bypass corrected-IMU forward.

    Freezes the existing backbone and correction heads for EKF fine-tuning.
    Covariance heads retain loaded weights. All filter equations and
    measurement covariances are fixed. Each call replays one full flight from
    initialization, retaining the graph through all aided updates and cutoff.
    """

    def __init__(self, network, ekf=None):
        super().__init__()
        if not network.conf.propcov:
            raise ValueError("AirIMU must have propcov=True for ESKF training.")
        uncertainty_names = ("acccov_decoder", "gyrocov_decoder", "acc_bias_gain", "gyro_bias_gain")
        frozen_names = ("cnn", "gru1", "gru2", "accdecoder", "gyrodecoder")
        if not all(hasattr(network, name) for name in uncertainty_names + frozen_names):
            raise TypeError("ESKF training requires CodeNet's encoder and uncertainty heads.")
        network.requires_grad_(False)
        for name in uncertainty_names:
            getattr(network, name).requires_grad_(True)
        self._frozen_modules = tuple(getattr(network, name) for name in frozen_names)
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
        return self

    @torch.no_grad()
    def extract_features(self, raw_acceleration, raw_gyroscope):
        """Cacheable frozen features; covariance decoders remain outside no_grad."""
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
        features = self.network.encoder(torch.cat((padded_acc, padded_gyro), dim=-1))[:, 1:, :]
        correction = self.network.decoder(features)
        return {"features": features[0], "correction": correction[0], "samples": len(acc)}

    def predict_variance_batch(self, cached):
        """Decode ragged flight features in one GPU batch, then gather IMU rows.

        Gathering replaces thousands of in-place slice updates and their backward
        nodes. Each flight retains its original CNN/GRU context and sample mapping.
        """
        reference = next(self.network.parameters())
        features = nn.utils.rnn.pad_sequence([item["features"] for item in cached], batch_first=True)
        corrections = nn.utils.rnn.pad_sequence([item["correction"] for item in cached], batch_first=True)
        features = features.to(device=reference.device, dtype=reference.dtype)
        corrections = corrections.to(device=reference.device, dtype=reference.dtype)
        decoded = self.network.cov_decoder(features, corrections)
        variances = []
        for index, item in enumerate(cached):
            indices = torch.div(torch.arange(item["samples"], device=reference.device)
                                + self.network.inter_head, self.network.interval,
                                rounding_mode="floor").clamp_max(len(item["features"])-1)
            variance = decoded[index].index_select(0, indices)
            variances.append(variance.to(device=self.ekf.gravity.device, dtype=torch.float64))
        return variances

    def predict_variance(self, raw_acceleration, raw_gyroscope):
        features = self.extract_features(raw_acceleration, raw_gyroscope)
        return self.predict_variance_batch([features])[0]

    def forward(self, raw_acceleration, raw_gyroscope, imu_time, vicon_events,
                cutoff_timestamp, initial_state, initial_covariance, vicon_velocity_gt,
                *, return_covariance=True):
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
