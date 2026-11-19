"""Fine-tune only AirIMU covariance heads through full-flight ESKF replay."""

import argparse
import json
import math
from pathlib import Path
import random
from time import perf_counter

import torch

from ekf import DifferentiableESKF
from ekf.data import apply_sensor_metadata, read_dataset_flights
from ekf.experiment import filter_metadata, load_airimu, save_checkpoint
from ekf.runner import FlightTrainer


def log_flight(output_dir, record):
    print(json.dumps(record), flush=True)
    with (output_dir / "flights.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def log_epoch(output_dir, record):
    """Save structured metrics and print readable per-axis epoch statistics."""
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    print(f"epoch={record['epoch']} train_xy_mse={record['train_xy_mse']:.6g} "
          f"validation_xy_mse={record['validation_xy_mse']:.6g} "
          f"best_validation_xy_mse={record['best_validation_xy_mse']:.6g} "
          f"epoch_seconds={record['epoch_seconds']:.3f}", flush=True)
    for phase in ("train", "validation"):
        covariance = record[f"{phase}_covariance"]
        for sensor in ("acc_cov", "gyro_cov"):
            statistics = covariance[sensor]
            values = {
                name: "[" + ", ".join(f"{value:.6e}" for value in statistics[key]) + "]"
                for name, key in (("min", "min"), ("avg", "mean"), ("max", "max"))
            }
            print(f"epoch={record['epoch']} phase={phase} sensor={sensor} axes=x,y,z "
                  f"samples={covariance['samples']} min={values['min']} "
                  f"avg={values['avg']} max={values['max']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Existing AirIMU network HOCON config")
    parser.add_argument("--ckpt", required=True, help="AirIMU checkpoint (covariance heads are retained)")
    parser.add_argument("--dataset-config", required=True, help="TLab HOCON dataset split config")
    parser.add_argument(
        "--sensor-metadata-root", required=True,
        help="Root containing the finalized initial_sensor_bias_*.csv tables",
    )
    parser.add_argument("--output", required=True, help="Experiment directory for ESKF checkpoints")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ekf-device", help="Filter device; defaults to CPU with replay workers, otherwise --device")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gravity", type=float, nargs=3, default=(0., 0., -9.80665))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--replay-workers", type=int, default=0,
                        help="Independent CPU replay processes; 0 replays in the main process")
    parser.add_argument("--flight-batch-size", type=int, default=1,
                        help="Complete flights averaged per Adam step; use >= replay-workers for parallel work")
    parser.add_argument("--feature-cache-mb", type=float, default=512,
                        help="CPU cache budget for frozen encoder features; 0 disables caching")
    parser.add_argument("--replay-backend", choices=("eager", "compiled"), default="compiled",
                        help="Compiled aligned TLab replay preserves full-flight gradients and Float64")
    parser.add_argument("--height-updates", action=argparse.BooleanOptionalAction, default=True,
                        help="Fuse clean Vicon z before and after cutoff at every estimate, std=0.1 m (default: enabled)")
    args = parser.parse_args()
    if args.epochs < 1 or not math.isfinite(args.lr) or args.lr <= 0 or args.cpu_threads < 1:
        parser.error("epochs, lr, and cpu-threads must be positive.")
    if (args.replay_workers < 0 or args.flight_batch_size < 1 or not math.isfinite(args.feature_cache_mb)
            or args.feature_cache_mb < 0):
        parser.error("Use nonnegative workers/cache budget and a positive batch size.")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    flights, cutoffs = apply_sensor_metadata(
        read_dataset_flights(args.dataset_config), args.sensor_metadata_root
    )
    training = [record for record in flights if record["split"] == "train"]
    validation = [record for record in flights if record["split"] == "validation"]
    if not training or not validation:
        parser.error("Dataset config must provide disjoint train and test flights.")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ekf_device = args.ekf_device or ("cpu" if args.replay_workers else args.device)
    if args.replay_workers and torch.device(ekf_device).type != "cpu":
        parser.error("--replay-workers requires --ekf-device cpu (AirIMU can stay on CUDA).")
    ekf = DifferentiableESKF(gravity=args.gravity, height_updates=args.height_updates).to(ekf_device)
    wrapper = load_airimu(args.config, args.ckpt, ekf, args.device)
    covariance_policy = {
        "use_cov_limits": wrapper.network.use_cov_limits,
        "acc_variance_min": float(wrapper.network.acc_cov_min),
        "gyro_variance_min": float(wrapper.network.gyro_cov_min),
        "acc_variance_max": (float(wrapper.network.acc_cov_max)
                             if wrapper.network.use_cov_limits else None),
        "gyro_variance_max": (float(wrapper.network.gyro_cov_max)
                              if wrapper.network.use_cov_limits else None),
    }
    parameters = [p for p in wrapper.network.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    metadata = {"gravity": list(args.gravity), "observation_types": list(ekf.observation_types),
                "timestamp_units": "seconds, original shared reference",
                "evaluation_start_offset": 5., "evaluation_duration": 20.,
                "config": str(Path(args.config).resolve()),
                "initial_checkpoint": str(Path(args.ckpt).resolve()),
                "flights": [{**record, "cutoff_timestamp": cutoffs[record["id"]]}
                            for record in training + validation], "seed": args.seed,
                "network_device": args.device, "ekf_device": ekf_device,
                "return_covariance_during_training": False,
                "replay_workers": args.replay_workers, "worker_threads": 1,
                "flight_batch_size": args.flight_batch_size, "feature_cache_mb": args.feature_cache_mb,
                "replay_backend": args.replay_backend,
                "covariance_policy": covariance_policy,
                "optimizer_objective": "mean of complete-flight horizontal velocity MSEs per batch",
                **filter_metadata(ekf)}
    with (output_dir / "eskf_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    best_validation = math.inf
    print(f"Network device: {args.device}; EKF device: {ekf_device}; "
          f"flights per epoch: {len(training)} train + {len(validation)} validation", flush=True)
    print(f"Initial biases: flight-specific CSV values from {Path(args.sensor_metadata_root).resolve()}; "
          "refined during Vicon aiding, retained at cutoff; raw IMU is unchanged.", flush=True)
    print("Covariance logs contain per-axis variances in x,y,z order, before the optimizer step.", flush=True)
    print(f"Replay workers: {args.replay_workers}; threads/worker: 1; "
          f"flights/Adam step: {args.flight_batch_size}; feature cache: {args.feature_cache_mb:g} MiB", flush=True)
    print(f"Replay backend: {args.replay_backend}", flush=True)
    print(f"Covariance policy: {json.dumps(covariance_policy)}", flush=True)
    print(f"Filter aiding policy: {json.dumps(filter_metadata(ekf))}", flush=True)
    with FlightTrainer(wrapper, optimizer, workers=args.replay_workers, batch_size=args.flight_batch_size,
                       feature_cache_mb=args.feature_cache_mb, replay_backend=args.replay_backend) as trainer:
        for epoch in range(args.epochs):
            epoch_start = perf_counter()
            rng.shuffle(training)
            previous_steps = trainer.optimizer_steps
            train_losses, train_covariance = trainer.run_split(
                training, cutoffs, epoch=epoch+1, phase="train", epoch_start=epoch_start,
                log=lambda record: log_flight(output_dir, record))
            validation_losses, validation_covariance = trainer.run_split(
                validation, cutoffs, epoch=epoch+1, phase="validation", epoch_start=epoch_start,
                log=lambda record: log_flight(output_dir, record))
            validation_mse = sum(validation_losses)/len(validation_losses)
            if not math.isfinite(validation_mse):
                raise RuntimeError("Validation loss is non-finite.")
            if validation_mse < best_validation:
                best_validation = validation_mse
                save_checkpoint(output_dir / "best_model.ckpt", wrapper, optimizer, epoch+1,
                                best_validation, metadata)
            save_checkpoint(output_dir / "newest.ckpt", wrapper, optimizer, epoch+1, best_validation, metadata)
            record = {"epoch": epoch+1, "train_xy_mse": sum(train_losses)/len(train_losses),
                      "validation_xy_mse": validation_mse, "best_validation_xy_mse": best_validation,
                      "train_covariance": train_covariance, "validation_covariance": validation_covariance,
                      "optimizer_steps": trainer.optimizer_steps-previous_steps,
                      "feature_cache_hits": trainer.cache_hits, "feature_cache_misses": trainer.cache_misses,
                      "epoch_seconds": perf_counter()-epoch_start}
            log_epoch(output_dir, record)


if __name__ == "__main__":
    main()
