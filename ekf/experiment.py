"""Shared experiment setup for full-flight ESKF training and evaluation."""

import math

import torch

from .training import AirIMUESKF, horizontal_velocity_residual


def filter_metadata(ekf):
    """Record the aiding policy independently of the AirIMU model weights."""
    return {"height_updates": ekf.height_updates,
            "height_source": "clean_vicon_z" if ekf.height_updates else None,
            "height_update_frequency": "estimate" if ekf.height_updates else None,
            "height_update_scope": "before_and_after_cutoff" if ekf.height_updates else None,
            "pre_cutoff_observation_types": list(ekf.observation_types) + (["height"] if ekf.height_updates else []),
            "height_measurement_std": math.sqrt(float(ekf.height_R[0, 0])),
            "post_cutoff_observation_types": ["height"] if ekf.height_updates else [],
            "post_cutoff_covariance_propagation": True}


def covariance_statistics(variance):
    """Per-axis predicted IMU variances, ordered acceleration xyz then gyro xyz."""
    values = variance.detach()
    statistics = torch.stack((values.mean(0), values.amin(0), values.amax(0))).cpu().tolist()
    return {"samples": len(values),
            **{name: {stat: statistics[index][offset:offset+3]
                      for index, stat in enumerate(("mean", "min", "max"))}
               for name, offset in (("acc_cov", 0), ("gyro_cov", 3))}}


class CovarianceStatistics:
    """Sample-weighted epoch statistics without retaining flight tensors."""

    def __init__(self):
        self.samples = 0
        self.total = {name: {"mean": [0.]*3, "min": [math.inf]*3, "max": [-math.inf]*3}
                      for name in ("acc_cov", "gyro_cov")}

    def update(self, statistics):
        count = statistics["samples"]
        self.samples += count
        for name, target in self.total.items():
            for axis in range(3):
                target["mean"][axis] += statistics[name]["mean"][axis] * count
                target["min"][axis] = min(target["min"][axis], statistics[name]["min"][axis])
                target["max"][axis] = max(target["max"][axis], statistics[name]["max"][axis])

    def summary(self):
        return {"samples": self.samples,
                **{name: {stat: [x / self.samples if stat == "mean" else x for x in values]
                          for stat, values in target.items()}
                   for name, target in self.total.items()}}


def load_airimu(config_path, checkpoint_path, ekf, device):
    # Keep the filter itself independent of PyPose, HOCON, and the dataset code.
    from pyhocon import ConfigFactory
    from model import net_dict

    conf = ConfigFactory.parse_file(str(config_path))
    conf.train["propcov"] = True
    conf.train["gtrot"] = False
    network_name = conf.train.network
    if network_name not in ("codenet", "codeposenet"):
        raise ValueError("Covariance-only ESKF training supports CodeNet/CodePoseNet.")
    network = net_dict[network_name](conf.train).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    network.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    return AirIMUESKF(network, ekf)


def save_checkpoint(path, wrapper, optimizer, epoch, validation_mse, metadata):
    """Compatible AirIMU network weights plus optimizer and experiment provenance."""
    torch.save({"model_state_dict": wrapper.network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
                "best_loss": validation_mse, "objective": "eskf_horizontal_velocity_mse",
                "ekf_metadata": metadata}, path)


def xy_metrics(velocity, ground_truth, mask):
    error = horizontal_velocity_residual(velocity, ground_truth, mask)
    xy_mse = float(error.square().sum(-1).mean())
    return {"xy_mse": xy_mse, "xy_rmse": math.sqrt(xy_mse),
            "x_rmse": math.sqrt(float(error[:, 0].square().mean())),
            "y_rmse": math.sqrt(float(error[:, 1].square().mean()))}
