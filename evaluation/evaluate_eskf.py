"""Replay fixed or AirIMU noise and compare horizontal velocity against Vicon/PX4.

Run from the repository root: python -m evaluation.evaluate_eskf --help
"""

import argparse
import json
import math
from pathlib import Path

import torch

from ekf import DifferentiableESKF, velocity_evaluation_mask
from ekf.data import (interpolate, prepare_flight, read_cutoff_table,
                      read_flight_manifest, read_numeric_csv)
from ekf.experiment import load_airimu, xy_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flights", required=True)
    parser.add_argument("--cutoffs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--config", help="Required with --ckpt for learned covariance")
    parser.add_argument("--ckpt", help="Omit for a fixed-noise comparison")
    parser.add_argument("--accel-variance", type=float, default=.001, help="Per-sample variance on each accel axis")
    parser.add_argument("--gyro-variance", type=float, default=.001, help="Per-sample variance on each gyro axis")
    parser.add_argument("--gravity", type=float, nargs=3, default=(0., 0., -9.80665))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if bool(args.config) != bool(args.ckpt):
        parser.error("Pass both --config and --ckpt, or neither for fixed noise.")
    if args.cpu_threads < 1 or any(not math.isfinite(x) or x < 0
                                  for x in (args.accel_variance, args.gyro_variance)):
        parser.error("Use positive cpu-threads and finite nonnegative variances.")
    torch.set_num_threads(args.cpu_threads)
    flights = [record for record in read_flight_manifest(args.flights) if record["split"] == args.split]
    if not flights:
        parser.error(f"No flights in split={args.split}.")
    cutoffs = read_cutoff_table(args.cutoffs)
    ekf = DifferentiableESKF(gravity=args.gravity).to(args.device)
    wrapper = load_airimu(args.config, args.ckpt, ekf, args.device) if args.ckpt else None
    if wrapper:
        wrapper.eval()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    with torch.no_grad():
        for index, record in enumerate(flights):
            cutoff = cutoffs[record["id"]]
            flight = prepare_flight(record, cutoff, ekf)
            if wrapper:
                output = wrapper(**flight)
                trajectory = output["trajectory"]
                variance = output["imu_variance"]
            else:
                variance = torch.tensor([args.accel_variance]*3 + [args.gyro_variance]*3,
                                         dtype=torch.float64, device=args.device).repeat(len(flight["imu_time"]), 1)
                trajectory = ekf.run_sequence(
                    {"acc": flight["raw_acceleration"], "gyro": flight["raw_gyroscope"]},
                    flight["imu_time"], variance, flight["vicon_events"], cutoff,
                    flight["initial_state"], flight["initial_covariance"])
            mask = velocity_evaluation_mask(flight["imu_time"], cutoff)
            report = {"id": record["id"], "noise_source": "airimu" if wrapper else "fixed",
                      "cutoff_timestamp": cutoff, "evaluation_start": cutoff+5.,
                      "evaluation_end": cutoff+25., "gravity": list(args.gravity),
                      "observation_types": list(ekf.observation_types),
                      "ekf_vs_vicon": xy_metrics(trajectory["velocity"], flight["vicon_velocity_gt"], mask)}
            px4 = read_numeric_csv(record["px4_velocity_csv"], ("time", "vel_x", "vel_y", "vel_z"))
            evaluation_times = flight["imu_time"][mask].cpu()
            if float(px4[0, 0]) <= float(evaluation_times[0]) and float(px4[-1, 0]) >= float(evaluation_times[-1]):
                px4_velocity = interpolate(px4[:, 0], px4[:, 1:4], evaluation_times).to(args.device)
                gt = flight["vicon_velocity_gt"][mask]
                all_samples = torch.ones(len(gt), dtype=torch.bool, device=args.device)
                report["px4_vs_vicon"] = xy_metrics(px4_velocity, gt, all_samples)
                report["ekf_vs_px4"] = xy_metrics(trajectory["velocity"][mask], px4_velocity, all_samples)
            else:
                report["px4_comparison_unavailable"] = "PX4 recording does not cover the evaluation interval."
            artifact_name = f"flight_{index:04d}.pt"
            torch.save({"id": record["id"], "imu_time": flight["imu_time"].cpu(),
                        "trajectory": {key: value.cpu() for key, value in trajectory.items()},
                        "imu_variance": variance.cpu(), "vicon_velocity_gt": flight["vicon_velocity_gt"].cpu(),
                        "evaluation_mask": mask.cpu(), "report": report}, output_dir / artifact_name)
            report["trajectory_file"] = artifact_name
            reports.append(report)
            print(json.dumps(report), flush=True)
            if wrapper:
                del output
            del trajectory, flight, variance
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2)


if __name__ == "__main__":
    main()
