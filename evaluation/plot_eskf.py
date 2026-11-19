"""Plot saved EKF evaluation trajectories and covariance over flight time."""

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_flight(path, output, dpi=160):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    flight_id = saved["id"]
    parts = Path(flight_id).parts
    if Path(flight_id).is_absolute() or ".." in parts:
        raise ValueError(f"Invalid flight id: {flight_id}")
    time = saved["imu_time"].numpy()
    relative = time - time[0]
    velocity = saved["trajectory"]["velocity"].numpy()
    truth = saved["vicon_velocity_gt"].numpy()
    variance = saved["imu_variance"].numpy()
    covariance = saved["trajectory"]["covariance"].numpy()
    n = len(time)
    if (velocity.shape != (n, 3) or truth.shape != (n, 3)
            or variance.shape != (n, 6) or covariance.shape != (n, 15, 15)):
        raise ValueError(f"Unexpected saved trajectory shapes in {path}")
    velocity_variance = np.diagonal(covariance, axis1=1, axis2=2)[:, 3:6]
    for value in (relative, velocity, truth, variance, velocity_variance):
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite plot data in {path}")
    if np.any(velocity_variance < -1e-10):
        raise ValueError(f"Negative velocity variance in {path}")
    sigma = np.sqrt(np.maximum(velocity_variance, 0))
    cutoff = saved["report"]["cutoff_timestamp"] - time[0]
    start, end = cutoff + 5, cutoff + 25
    destination = output.joinpath(*parts, "uncertainty")
    destination.mkdir(parents=True, exist_ok=True)

    def draw(axes, axis):
        label = "xyz"[axis]
        axes[0].plot(relative, truth[:, axis], color="mediumseagreen", lw=1, label="Vicon velocity")
        axes[0].plot(relative, velocity[:, axis], color="red", lw=.9, label="EKF velocity")
        axes[0].fill_between(relative, velocity[:, axis]-2*sigma[:, axis],
                             velocity[:, axis]+2*sigma[:, axis], color="red", alpha=.12,
                             label="EKF velocity ±2σ")
        axes[1].plot(relative, variance[:, axis], color="tab:orange", lw=.9,
                     label=f"Predicted accel variance {label}")
        axes[2].plot(relative, variance[:, axis+3], color="tab:purple", lw=.9,
                     label=f"Predicted gyro variance {label}")
        axes[3].plot(relative, velocity_variance[:, axis], color="tab:blue", lw=.9,
                     label=f"EKF velocity variance P[{axis+3},{axis+3}]")
        for ax, ylabel in zip(axes, ("Velocity (m/s)", "Accel variance\n(m²/s⁴)",
                                     "Gyro variance\n(rad²/s²)", "Velocity variance\n(m²/s²)")):
            ax.axvspan(start, end, color="gray", alpha=.12, label="Scored interval")
            ax.axvline(cutoff, color="black", ls="--", lw=.9, label="Vicon cutoff")
            ax.set_ylabel(ylabel)
            ax.set_xlim(relative[0], relative[-1])
            ax.grid(True, alpha=.3)
            ax.legend(loc="best", fontsize=7)
        axes[-1].set_xlabel("Time since flight initialization (s)")

    paths = []
    for axis, label in enumerate("xyz"):
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 10))
        draw(axes, axis)
        fig.suptitle(f"{flight_id}\nEKF velocity and uncertainty: {label} axis", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .95))
        target = destination / f"velocity_{label}_with_uncertainty_outputs.png"
        fig.savefig(target, dpi=dpi)
        plt.close(fig)
        paths.append(str(target.relative_to(output)))

    fig, axes = plt.subplots(4, 3, sharex=True, figsize=(18, 12))
    for axis, label in enumerate("xyz"):
        draw(axes[:, axis], axis)
        axes[0, axis].set_title(f"{label.upper()} axis")
    score = saved["report"]["ekf_vs_vicon"]["xy_rmse"]
    fig.suptitle(f"{flight_id}\nEKF horizontal velocity RMSE: {score:.3f} m/s", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    overview = destination / "velocity_xyz_with_uncertainty_outputs.png"
    fig.savefig(overview, dpi=dpi)
    plt.close(fig)
    return {"id": flight_id, "overview": str(overview.relative_to(output)), "axis_plots": paths,
            "cutoff_relative_seconds": float(cutoff), "xy_velocity_rmse": score}


def plot_evaluation(evaluation, destination=None, dpi=160):
    """Render all saved flights and an HTML index without rerunning the filter."""
    torch.set_num_threads(1)
    source = Path(evaluation).resolve()
    output = Path(destination).resolve() if destination else source / "plots"
    output.mkdir(parents=True, exist_ok=True)
    reports = json.loads((source / "metrics.json").read_text())
    if not reports:
        raise ValueError("No evaluated flights to plot.")
    plots = []
    for row in reports:
        path = source / row["trajectory_file"]
        result = plot_flight(path, output, dpi)
        if result["id"] != row["id"]:
            raise ValueError(f"Flight id mismatch in {path}")
        plots.append(result)
        print(f"Plotted {len(plots)}/{len(reports)}: {row['id']}", flush=True)
    (output / "manifest.json").write_text(json.dumps(plots, indent=2) + "\n")
    intro = ("<h1>EKF velocity and covariance over time</h1>"
             "<p>These plots use the saved EKF trajectories and IMU variance predictions. "
             "The dashed line marks Vicon cutoff; shading marks cutoff+5 to cutoff+25 seconds. "
             "The velocity band is ±2 standard deviations from the EKF state covariance. "
             "IMU variance outputs and EKF velocity variance are shown separately.</p>")
    sections = []
    for row in plots:
        links = " · ".join(f'<a href="{html.escape(path)}">{label.upper()} axis</a>'
                            for label, path in zip("xyz", row["axis_plots"]))
        sections.append(f'<section><h2>{html.escape(row["id"])}</h2><p>{links}</p>'
                        f'<a href="{html.escape(row["overview"])}">'
                        f'<img loading="lazy" src="{html.escape(row["overview"])}" '
                        'style="width:100%;max-width:1500px" alt="Velocity and uncertainty overview"></a></section>')
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8">'
        '<title>EKF evaluation plots</title><style>body{font:16px sans-serif;margin:32px}'
        'section{margin:40px 0}h2{font-size:18px}</style>' + intro + "".join(sections))
    print(f"Plot index: {output / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, help="Directory containing EKF metrics.json and flight_*.pt")
    parser.add_argument("--output", help="Default: evaluation/plots")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    plot_evaluation(args.evaluation, args.output, args.dpi)


if __name__ == "__main__":
    main()
