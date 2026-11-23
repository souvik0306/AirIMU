"""Shared experiment setup for full-flight ESKF training and evaluation."""

import math
import json
import hashlib
from pathlib import Path

import torch

from .training import AirIMUESKF, horizontal_velocity_residual


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


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


def validate_gru1_evidence(a_run, b_run):
    """C is an explicit subsequent experiment, only after matched B beats A."""
    runs = [Path(a_run), Path(b_run)]
    metadata = [json.loads((run / "eskf_metadata.json").read_text()) for run in runs]
    if [record.get("configuration") for record in metadata] != ["A", "B"]:
        raise ValueError("C evidence must contain A and B experiments in that order.")
    for key in ("initial_checkpoint", "initial_checkpoint_sha256", "config", "flights", "seed", "loss_options", "height_updates",
                "covariance_policy", "flight_batch_size", "gradient_clip", "correction_weight",
                'gyro_reference_quality_policy'):
        if metadata[0].get(key) != metadata[1].get(key):
            raise ValueError(f"A/B validation evidence differs in {key}.")
    best = []
    for run in runs:
        metrics = [json.loads(line)["validation_xy_mse"] for line in (run / "metrics.jsonl").read_text().splitlines()]
        if not metrics or not all(math.isfinite(value) for value in metrics):
            raise ValueError("C evidence requires finite validation metrics.")
        best.append(min(metrics))
    if best[1] >= best[0]:
        raise ValueError("Enable C only when B improves validation velocity MSE over A.")
    return best


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


def load_airimu(config_path, checkpoint_path, ekf, device, *, configuration="A", initialize_adapter=False):
    # Keep the filter itself independent of PyPose, HOCON, and the dataset code.
    from pyhocon import ConfigFactory
    from model import net_dict
    from model.checkpoints import load_covariance_checkpoint, validate_covariance_policy

    conf = ConfigFactory.parse_file(str(config_path))
    conf.train["propcov"] = True
    conf.train["gtrot"] = False
    network_name = conf.train.network
    if network_name not in ("codenet", "codeposenet"):
        raise ValueError("Covariance-only ESKF training supports CodeNet/CodePoseNet.")
    network = net_dict[network_name](conf.train).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    validate_covariance_policy(network, checkpoint.get("covariance_policy"))
    load_covariance_checkpoint(network, checkpoint.get("model_state_dict", checkpoint),
                               initialize_adapter=initialize_adapter)
    return AirIMUESKF(network, ekf, configuration=configuration)


def save_checkpoint(path, wrapper, optimizer, epoch, validation_mse, metadata):
    """Compatible AirIMU network weights plus optimizer and experiment provenance."""
    from model.checkpoints import covariance_policy
    torch.save({"model_state_dict": wrapper.network.state_dict(),
                "covariance_policy": covariance_policy(wrapper.network),
                "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
                "best_loss": validation_mse, "objective": metadata.get("optimizer_objective", "eskf_horizontal_velocity_mse"),
                "ekf_metadata": metadata}, path)


def xy_metrics(velocity, ground_truth, mask):
    error = horizontal_velocity_residual(velocity, ground_truth, mask)
    xy_mse = float(error.square().sum(-1).mean())
    return {"xy_mse": xy_mse, "xy_rmse": math.sqrt(xy_mse),
            "x_rmse": math.sqrt(float(error[:, 0].square().mean())),
            "y_rmse": math.sqrt(float(error[:, 1].square().mean()))}
