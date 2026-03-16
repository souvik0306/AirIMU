#!/usr/bin/env python3
"""
Plot raw IMU readings vs AirIMU corrected readings for XYZ axes.

Creates comparison plots showing the difference between raw IMU measurements
and AirIMU-corrected measurements for accelerometer and gyroscope data.
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import CPU_Unpickler


def plot_imu_comparison(data, sequence_name, save_dir, time_span=None):
    """
    Plot raw vs corrected IMU data for XYZ axes.

    Args:
        data: Dictionary containing 'raw_acc', 'raw_gyro', 'corrected_acc', 'corrected_gyro', 'dt'
        sequence_name: Name of the sequence for plot titles
        save_dir: Directory to save plots
        time_span: Tuple (start_time, end_time) in seconds, or None for full sequence
    """
    # Extract data
    raw_acc = data['raw_acc'].numpy()
    raw_gyro = data['raw_gyro'].numpy()
    corrected_acc = data['corrected_acc'].numpy()
    corrected_gyro = data['corrected_gyro'].numpy()
    dt = data['dt'].numpy()

    # Create time axis
    time_axis = np.cumsum(dt) - dt[0]  # Start from 0

    # Handle time span selection
    if time_span is not None:
        start_time, end_time = time_span
        time_mask = (time_axis >= start_time) & (time_axis <= end_time)
        time_axis = time_axis[time_mask]
        raw_acc = raw_acc[time_mask]
        raw_gyro = raw_gyro[time_mask]
        corrected_acc = corrected_acc[time_mask]
        corrected_gyro = corrected_gyro[time_mask]

        plot_title_suffix = f" ({start_time:.1f}s - {end_time:.1f}s)"
        filename_suffix = f"_{int(start_time*10)}s_{int(end_time*10)}s"
    else:
        plot_title_suffix = " (Full Sequence)"
        filename_suffix = "_full"

    # Create figure with 3 rows: raw vs corrected, difference, zoomed difference
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Raw IMU vs AirIMU Corrected - {sequence_name}{plot_title_suffix}',
                 fontsize=16, fontweight='bold')

    # Axis labels
    axis_labels = ['X', 'Y', 'Z']
    
    # ========== ACCELEROMETER PLOT ==========
    fig_acc, axes_acc = plt.subplots(1, 3, figsize=(18, 5))
    fig_acc.suptitle(f'Raw vs AirIMU Corrected Accelerometer - {sequence_name}{plot_title_suffix}',
                     fontsize=14, fontweight='bold')

    # Row 1: Raw vs Corrected overlay for accelerometer
    for i in range(3):
        ax = axes_acc[i]
        ax.plot(time_axis, raw_acc[:, i], color='red', linestyle='-',
                linewidth=2.5, alpha=0.7, label=f'Raw {axis_labels[i]}')
        ax.plot(time_axis, corrected_acc[:, i], color='blue', linestyle='-',
                linewidth=2.5, alpha=0.9, label=f'AirIMU {axis_labels[i]}')
        ax.set_title(f'Accelerometer {axis_labels[i]}-axis', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # Set finer y-scale
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05*y_range, y_max + 0.05*y_range)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=8))

    plt.tight_layout()

    # Save accelerometer plot
    os.makedirs(save_dir, exist_ok=True)
    acc_filename = f"acc_comparison_{sequence_name}{filename_suffix}.png"
    acc_path = os.path.join(save_dir, acc_filename)
    plt.savefig(acc_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved accelerometer plot: {acc_path}")

    # ========== GYROSCOPE PLOT ==========
    fig_gyro, axes_gyro = plt.subplots(1, 3, figsize=(18, 5))
    fig_gyro.suptitle(f'Raw vs AirIMU Corrected Gyroscope - {sequence_name}{plot_title_suffix}',
                      fontsize=14, fontweight='bold')

    # Row 2: Raw vs Corrected overlay for gyroscope
    for i in range(3):
        ax = axes_gyro[i]
        ax.plot(time_axis, raw_gyro[:, i], color='red', linestyle='-',
                linewidth=2.5, alpha=0.7, label=f'Raw {axis_labels[i]}')
        ax.plot(time_axis, corrected_gyro[:, i], color='blue', linestyle='-',
                linewidth=2.5, alpha=0.9, label=f'AirIMU {axis_labels[i]}')
        ax.set_title(f'Gyroscope {axis_labels[i]}-axis', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # Set finer y-scale
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05*y_range, y_max + 0.05*y_range)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=8))

    plt.tight_layout()

    # Save gyroscope plot
    gyro_filename = f"gyro_comparison_{sequence_name}{filename_suffix}.png"
    gyro_path = os.path.join(save_dir, gyro_filename)
    plt.savefig(gyro_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved gyroscope plot: {gyro_path}")

    # Print summary statistics
    print(f"\nIMU Correction Summary for {sequence_name}{plot_title_suffix}:")
    print("=" * 70)

    print(f"\n{'Sensor':<15} {'Axis':<8} {'Raw Std':<15} {'Corrected Std':<15} {'Improvement':<12}")
    print("-" * 70)
    
    for sensor_name, raw_data, corrected_data, unit in [
        ("Accelerometer", raw_acc, corrected_acc, "m/s²"),
        ("Gyroscope", raw_gyro, corrected_gyro, "rad/s")
    ]:
        for i, axis in enumerate(['X', 'Y', 'Z']):
            raw_std = np.std(raw_data[:, i])
            corrected_std = np.std(corrected_data[:, i])
            improvement = (raw_std - corrected_std) / raw_std * 100 if raw_std > 0 else 0
            
            print(f"{sensor_name:<15} {axis:<8} {raw_std:<15.6e} {corrected_std:<15.6e} {improvement:>10.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Plot raw IMU vs AirIMU corrected readings",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--pickle', type=str, required=True,
                        help='Path to the inference output pickle file')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Specific sequence name to plot (if not provided, plots all)')
    parser.add_argument('--save-dir', type=str, default='./imu_plots',
                        help='Directory to save plots')
    parser.add_argument('--time-span', type=float, nargs=2,
                        help='Time span to plot in seconds (start end), e.g., --time-span 0 0.2. Default: 0.2s from start')

    args = parser.parse_args()

    # Load pickle file
    print(f"Loading inference results from: {args.pickle}")
    if not os.path.exists(args.pickle):
        raise FileNotFoundError(f"Pickle file not found: {args.pickle}")

    with open(args.pickle, 'rb') as f:
        results = CPU_Unpickler(f).load()

    print(f"✓ Loaded results for {len(results)} sequences")

    # Determine which sequences to plot
    if args.sequence:
        if args.sequence not in results:
            available_sequences = list(results.keys())
            raise ValueError(f"Sequence '{args.sequence}' not found. Available: {available_sequences}")
        sequences_to_plot = [args.sequence]
    else:
        sequences_to_plot = list(results.keys())

    # Plot each sequence
    for seq_name in sequences_to_plot:
        print(f"\nProcessing sequence: {seq_name}")
        data = results[seq_name]

        # Check if required data is present
        required_keys = ['raw_acc', 'raw_gyro', 'corrected_acc', 'corrected_gyro', 'dt']
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            print(f"⚠️  Skipping {seq_name} - missing keys: {missing_keys}")
            continue

        # Plot full sequence
        plot_imu_comparison(data, seq_name, args.save_dir, time_span=None)

        # Plot 2-second span if requested
        if args.time_span:
            start_time, end_time = args.time_span
            plot_imu_comparison(data, seq_name, args.save_dir,
                              time_span=(start_time, end_time))
        else:
            # Plot a default 0.2-second span from the start of the sequence
            total_time = np.sum(data['dt'].numpy())
            if total_time > 0.2:  # Only if sequence is long enough
                plot_imu_comparison(data, seq_name, args.save_dir,
                                  time_span=(0, 0.2))

    print(f"\n✓ All plots saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
