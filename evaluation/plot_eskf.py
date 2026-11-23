"""Plot saved EKF velocity and uncertainty as standard deviations over time."""

import argparse
import html
import json
import os
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
    if np.any(velocity_variance < -1e-10) or np.any(variance < 0):
        raise ValueError(f"Negative variance in {path}")
    sigma = np.sqrt(np.maximum(velocity_variance, 0))
    noise_sigma = np.sqrt(variance)
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
        axes[1].plot(relative, noise_sigma[:, axis], color="tab:orange", lw=.9,
                     label=f"Predicted accel σ {label} = √variance")
        axes[2].plot(relative, noise_sigma[:, axis+3], color="tab:purple", lw=.9,
                     label=f"Predicted gyro σ {label} = √variance")
        axes[3].plot(relative, sigma[:, axis], color="tab:blue", lw=.9,
                     label=f"EKF velocity σ = √P[{axis+3},{axis+3}]")
        for ax, ylabel in zip(axes, ("Velocity (m/s)", "Accel noise σ\n(m/s²)",
                                     "Gyro noise σ\n(rad/s)", "Velocity σ\n(m/s)")):
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
    zoom = destination / "velocity_xyz_scoring_window.png"
    for ax in axes.flat:
        ax.set_xlim(max(float(relative[0]), cutoff-2), min(float(relative[-1]), end+2))
    fig.suptitle(f"{flight_id}\nCutoff and scoring window; horizontal velocity RMSE: {score:.3f} m/s", fontsize=13)
    fig.savefig(zoom, dpi=dpi)
    plt.close(fig)
    return {"id": flight_id, "overview": str(overview.relative_to(output)), "axis_plots": paths,
            "scoring_window": str(zoom.relative_to(output)),
            "uncertainty_representation": "standard_deviation_sqrt_covariance",
            "velocity_band_standard_deviations": 2,
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
    intro = ("<h1>EKF velocity and uncertainty over time</h1>"
             "<p>These plots use the saved EKF trajectories and IMU variance predictions. "
             "The dashed line marks Vicon cutoff; shading marks cutoff+5 to cutoff+25 seconds. "
             "The velocity band is ±2 standard deviations from the EKF state covariance. "
             "Accelerometer noise σ (m/s²), gyroscope noise σ (rad/s), and EKF velocity σ (m/s) "
             "are square roots of their respective variances, shown separately. "
             "These bands describe model uncertainty; their observed coverage is evaluated separately.</p>")
    sections = []
    for row in plots:
        links = " · ".join(f'<a href="{html.escape(path)}">{label.upper()} axis</a>'
                            for label, path in zip("xyz", row["axis_plots"]))
        sections.append(f'<section><h2>{html.escape(row["id"])}</h2><p>{links} · '
                        f'<a href="{html.escape(row["overview"])}">Full flight overview</a></p>'
                        f'<a href="{html.escape(row["scoring_window"])}">'
                        f'<img loading="lazy" src="{html.escape(row["scoring_window"])}" '
                        'style="width:100%;max-width:1500px" alt="Velocity and uncertainty overview"></a></section>')
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8">'
        '<title>EKF evaluation plots</title><style>body{font:16px sans-serif;margin:32px}'
        'section{margin:40px 0}h2{font-size:18px}</style>' + intro + "".join(sections))
    print(f"Plot index: {output / 'index.html'}", flush=True)


def verify_plot_outputs(evaluation):
    """Require a complete five-image plot set for every evaluated flight."""
    source = Path(evaluation).resolve()
    output = source / "plots"
    reports = json.loads((source / "metrics.json").read_text())
    plots = json.loads((output / "manifest.json").read_text())
    expected = {row["id"] for row in reports}
    if not expected or len(plots) != len(expected) or {row["id"] for row in plots} != expected:
        raise ValueError(f"Plot flight IDs differ from evaluation: {source}")
    if not (output / "index.html").is_file():
        raise ValueError(f"Missing per-model plot gallery: {source}")
    for row in plots:
        if (row["uncertainty_representation"] != "standard_deviation_sqrt_covariance"
                or row["velocity_band_standard_deviations"] != 2 or len(row["axis_plots"]) != 3):
            raise ValueError(f"Unexpected plot uncertainty convention: {row['id']}")
        paths = [row["overview"], row["scoring_window"], *row["axis_plots"]]
        if len(set(paths)) != 5:
            raise ValueError(f"Incomplete plot set: {row['id']}")
        for path in paths:
            target = (output / path).resolve()
            if not target.is_relative_to(output) or not target.is_file() or target.stat().st_size == 0:
                raise ValueError(f"Missing or invalid plot: {path}")
    return {row["id"]:row for row in plots}


def write_comparison_index(evaluations, comparison, output):
    """Build a single gallery pairing the four model plots for each flight."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plots = {label:verify_plot_outputs(path) for label,path in evaluations.items()}
    flight_rows = comparison["paired_flights"]
    expected = {row["flight"] for row in flight_rows}
    if any(set(rows) != expected for rows in plots.values()):
        raise ValueError("Plot flight IDs differ from the paired model comparison.")

    def link(path):
        return html.escape(os.path.relpath(path, output), quote=True)

    intro = ('<!doctype html><meta charset="utf-8"><title>Held-out flight velocity and uncertainty</title>'
             '<style>body{font:16px sans-serif;margin:32px}img{width:100%;max-width:1500px}'
             'details{margin:32px 0}summary{cursor:pointer;font-weight:bold}'
             'td,th{padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}</style>'
             '<h1>Held-out flight velocity and uncertainty</h1>'
             f'<p>Stage 1, A, B Strong and C use the same {comparison["flight_count"]} reserved evaluation flights. '
             'Dashed line: Vicon cutoff. Shading: scored interval, cutoff+5 to cutoff+25 s. '
             'Velocity bands: ±2√Pvv from the EKF. Noise panels: predicted √variance, '
             'in m/s² for accelerometer and rad/s for gyroscope. Velocity uncertainty is in m/s.</p>')
    model_links = ' · '.join(f'<a href="{link(Path(path)/"plots/index.html")}">{html.escape(label)}</a>'
                             for label,path in evaluations.items())
    parts = [intro, f'<p>Full model galleries: {model_links}</p>',
             f'<p><a href="{link(output/"comparison.png")}">Aggregate comparison plot</a> · '
             f'<a href="{link(output/"comparison_flights.csv")}">Per-flight metrics CSV</a></p>']
    for row in flight_rows:
        parts.append(f'<details><summary>{html.escape(row["flight"])}</summary>'
                     '<table><tr><th>Model</th><th>XY RMSE (m/s)</th><th>Improvement vs Stage 1</th></tr>')
        for label in evaluations:
            gain = row[label+"_improvement_percent_vs_Stage1"]
            gain_text = f'{gain:+.2f}%' if gain is not None else 'Unavailable'
            parts.append(f'<tr><td>{html.escape(label)}</td><td>{row[label+"_xy_rmse"]:.4f}</td>'
                         f'<td>{gain_text}</td></tr>')
        parts.append('</table>')
        for label,path in evaluations.items():
            plot = plots[label][row["flight"]]
            base = Path(path)/"plots"
            parts.append(f'<h3>{html.escape(label)}</h3><p>'
                         f'<a href="{link(base/plot["overview"])}">Full flight</a> · '
                         + ' · '.join(f'<a href="{link(base/file)}">{axis.upper()} axis</a>'
                                      for axis,file in zip('xyz',plot['axis_plots'])) + '</p>'
                         f'<a href="{link(base/plot["scoring_window"])}">'
                         f'<img loading="lazy" src="{link(base/plot["scoring_window"])}" '
                         f'alt="{html.escape(label)} velocity and uncertainty"></a>')
        parts.append('</details>')
    target = output / "index.html"
    target.write_text(''.join(parts))
    print(f"Four-model per-flight plot gallery: {target}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, help="Directory containing EKF metrics.json and flight_*.pt")
    parser.add_argument("--output", help="Default: evaluation/plots")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    plot_evaluation(args.evaluation, args.output, args.dpi)


if __name__ == "__main__":
    main()
