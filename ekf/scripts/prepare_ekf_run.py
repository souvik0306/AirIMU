"""Freeze and validate an AirIMU checkpoint before EKF job submission."""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from pyhocon import ConfigFactory, HOCONConverter
import torch

from datasets.initial_sensor_bias import resolve_initial_sensor_bias
from ekf import DifferentiableESKF
from ekf.data import apply_sensor_metadata, read_dataset_flights
from ekf.experiment import load_airimu
from jobs.utils.prepare_stage12_run import flight_map


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-run", required=True)
    parser.add_argument("--checkpoint", help="Explicit source checkpoint; otherwise use stage2/ckpt/best_model.ckpt")
    parser.add_argument("--source-stage", choices=("stage1", "stage2"), default="stage2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    if args.epochs < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs and learning rate must be positive.")
    source = Path(args.stage2_run).resolve()
    if (source / "scratch").is_dir():
        source = source / "scratch"
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else source / "stage2/ckpt/best_model.ckpt"
    config_path = source / "configs/stage2.conf"
    dataset_path = source / "configs/dataset.conf"
    for path in (checkpoint, config_path, dataset_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new experiment directory: {output}")
    conf = ConfigFactory.parse_file(str(config_path))
    dataset = ConfigFactory.parse_file(str(dataset_path))
    if conf.train.get("use_cov_limits", True):
        raise ValueError("Stage 2 must have use_cov_limits=False.")
    for name in ("acc_cov_min", "gyro_cov_min"):
        value = float(conf.train[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid covariance floor: {name}={value}")
    if not conf.train.get("use_initial_sensor_bias", False):
        raise ValueError("Stage 2 must enable initial sensor bias subtraction.")
    splits = {name: flight_map(dataset, name) for name in ("train", "test", "eval", "inference")}
    used = set()
    for name in ("train", "test", "eval"):
        if used.intersection(splits[name]):
            raise ValueError(f"Overlapping flight split: {name}")
        used.update(splits[name])
    if used != set(splits["inference"]):
        raise ValueError("Inference must cover the union of all three flight splits.")
    if any(not splits[name] for name in splits):
        raise ValueError("All flight splits must be nonempty.")
    bias_source = Path(conf.train.initial_sensor_bias_root).resolve()
    biases = {key: resolve_initial_sensor_bias(bias_source, root, drive)
              for key, (root, drive) in splits["inference"].items()}

    configs = output / "configs"
    configs.mkdir(parents=True)
    inputs = output / "inputs"
    inputs.mkdir()
    bias_snapshot = configs / "initial_sensor_biases"
    files = {}
    for bias in biases.values():
        source_table = Path(bias.source_file).resolve()
        destination = bias_snapshot / source_table.relative_to(bias_source)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_table, destination)
            files[str(destination.relative_to(output))] = sha256(destination)
    # Keep data locations independent of the shell used to read the snapshots.
    for name in splits:
        for entry in dataset[name].data_list:
            entry["data_root"] = str(Path(entry.data_root).resolve())
    collate = conf.dataset.get("collate", "padding9")
    conf["dataset"] = ConfigFactory.from_dict(dataset.as_plain_ordered_dict())
    conf.dataset["collate"] = collate
    conf.train["initial_sensor_bias_root"] = str(bias_snapshot)
    frozen_config = configs / "stage2.conf"
    frozen_dataset = configs / "dataset.conf"
    frozen_config.write_text(HOCONConverter.to_json(conf) + "\n")
    frozen_dataset.write_text(HOCONConverter.to_json(dataset) + "\n")
    frozen_checkpoint = inputs / ("initial_airimu.ckpt" if args.source_stage == "stage1" else "stage2_best_model.ckpt")
    shutil.copy2(checkpoint, frozen_checkpoint)
    for path in (frozen_config, frozen_dataset, frozen_checkpoint):
        files[str(path.relative_to(output))] = sha256(path)
    if sha256(checkpoint) != sha256(frozen_checkpoint):
        raise ValueError("Stage 2 checkpoint changed while it was being copied.")

    flights, _ = apply_sensor_metadata(read_dataset_flights(frozen_dataset), bias_snapshot)
    for record in flights:
        bias = biases[record["id"]]
        if (record["initial_accel_bias"] != list(bias.acc)
                or record["initial_gyro_bias"] != list(bias.gyro)
                or record["sensor_metadata"]["cutoff_ros_epoch_s"] != bias.cutoff_ros_epoch_s):
            raise ValueError(f"Stage 2/EKF bias or cutoff mismatch: {record['id']}")
    torch.set_num_threads(1)
    wrapper = load_airimu(frozen_config, frozen_checkpoint, DifferentiableESKF(), "cpu")
    wrapper.eval()
    raw_acc = torch.tensor([[0., 0., 9.80665]]).repeat(256, 1)
    variance = wrapper.predict_variance(raw_acc, torch.zeros_like(raw_acc))
    if variance.shape != (256, 6) or not bool(torch.isfinite(variance).all()):
        raise ValueError("Stage 2 did not produce six finite variances per IMU sample.")
    variance.mean().backward()
    gradients = [p.grad for p in wrapper.network.parameters() if p.requires_grad]
    if not any(g is not None and bool(g.abs().gt(0).any()) for g in gradients):
        raise ValueError("Covariance heads have no nonzero gradients.")
    if any(g is not None and not bool(torch.isfinite(g).all()) for g in gradients):
        raise ValueError("Covariance-head gradients are non-finite.")
    state = torch.load(frozen_checkpoint, map_location="cpu", weights_only=True)
    if any(not bool(torch.isfinite(v).all()) for v in state["model_state_dict"].values()):
        raise ValueError("Stage 2 model weights contain non-finite values.")
    plan = {
        "created_sgt": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "source_stage2_run": str(source), "source_stage2_checkpoint": str(checkpoint),
        "source_initialization_stage": args.source_stage,
        "source_checkpoint": str(checkpoint),
        "requested_checkpoint_path": str(Path(args.checkpoint).absolute()) if args.checkpoint else str(checkpoint),
        "source_checkpoint_epoch": state.get("epoch"),
        "initial_checkpoint": str(frozen_checkpoint), "input_sha256": files,
        "run_root": str(output), "config": str(frozen_config),
        "dataset_config": str(frozen_dataset), "sensor_metadata_root": str(bias_snapshot),
        "epochs": args.epochs, "learning_rate": args.lr,
        "split_counts": {name: len(value) for name, value in splits.items()},
        "ekf_split_counts": dict(Counter(record["split"] for record in flights)),
        "use_initial_sensor_bias": True, "use_cov_limits": False,
        "covariance_floors": {key: float(conf.train[key]) for key in ("acc_cov_min", "gyro_cov_min")},
        "covariance_maximums": None, "replay_backend": "compiled", "replay_workers": 4,
        "flight_batch_size": 4, "network_device": "cuda:0", "ekf_device": "cpu",
        "pre_cutoff_height_fusion": True, "post_cutoff_height_fusion": True,
        "post_cutoff_covariance_propagation": True,
        "height_updates": True, "height_source": "clean_vicon_z",
        "height_update_frequency": "estimate", "height_measurement_std": 0.1,
        "height_update_scope": "before_and_after_cutoff",
        "workflow": ["ekf_training", "best_checkpoint_inference", "state_evaluation",
                     "held_out_ekf_evaluation", "fp32_onnx_and_torch_export"],
        "submission_queue": "auto_free", "requested_resources": {"ncpus": 8, "mem": "64gb", "ngpus": 1,
                                                                    "walltime": "06:00:00"},
        "preflight": {"strict_checkpoint_load": True, "finite_covariance_and_gradients": True,
                      "bias_and_cutoff_consistency": len(flights)},
        "status": "validated; ready for submission",
    }
    (output / "run_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({key: value for key, value in plan.items() if key != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
