"""Shared experiment setup for full-flight ESKF training and evaluation."""

import math

import torch

from .training import AirIMUESKF


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
    error = velocity[mask, :2] - ground_truth[mask, :2]
    return {"xy_mse": float(error.square().sum(-1).mean()),
            "xy_rmse": math.sqrt(float(error.square().sum(-1).mean())),
            "x_rmse": math.sqrt(float(error[:, 0].square().mean())),
            "y_rmse": math.sqrt(float(error[:, 1].square().mean()))}
