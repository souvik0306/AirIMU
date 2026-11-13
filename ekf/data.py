"""Offline flight preparation. Original Vicon events and GT stay separate.

CSV schemas use SI units, seconds in a shared reference, and Hamilton wxyz
body-to-navigation quaternions. No automatic frame or timestamp conversion is
performed. The aligned TLab loader is intentionally bypassed for EKF replay.
"""

import csv
import json
import math
from pathlib import Path

import torch

from .eskf import ViconEvent, normalize_quaternion


def read_cutoff_table(path):
    """Read dataset,data_name,cutoff_timestamp keyed like the initial bias tables.

    The stable key is DATASET/CATEGORY/GROUP/flight_N (or DATASET/GROUP/flight_N).
    Cutoffs must use the original CSV timestamp reference, not elapsed time.
    """
    result = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "data_name", "cutoff_timestamp"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"Cutoff table requires columns {sorted(required)}.")
        for line, row in enumerate(reader, 2):
            dataset = row["dataset"].strip()
            data_name = row["data_name"].strip().replace("\\", "/").strip("/")
            key = f"{dataset}/{data_name}"
            cutoff = float(row["cutoff_timestamp"])
            if not dataset or not data_name or not math.isfinite(cutoff) or key in result:
                raise ValueError(f"Invalid or duplicate cutoff at {path}:{line}.")
            result[key] = cutoff
    return result


def read_flight_manifest(path):
    """Resolve paths relative to the manifest. Each id must match the cutoff key."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        flights = json.load(handle)
    if not isinstance(flights, list) or not flights:
        raise ValueError("Flight manifest must be a nonempty JSON list.")
    seen = set()
    for flight in flights:
        if flight["id"] in seen:
            raise ValueError(f"Duplicate flight id: {flight['id']}")
        seen.add(flight["id"])
        if flight.get("split") not in ("train", "validation", "test"):
            raise ValueError("Each flight requires split=train, validation, or test.")
        for key in ("imu_csv", "vicon_csv", "ground_truth_csv", "px4_velocity_csv"):
            flight[key] = str((path.parent / flight[key]).resolve())
    return flights


def read_numeric_csv(path, columns):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(columns) <= set(reader.fieldnames or ()):
            raise ValueError(f"{path} requires columns {columns}.")
        rows = [[float(row[key]) for key in columns] for row in reader]
    values = torch.tensor(rows, dtype=torch.float64)
    if values.ndim != 2 or len(values) < 2 or not bool(torch.isfinite(values).all()):
        raise ValueError(f"{path} must contain at least two finite numeric rows.")
    if not bool((values[1:, 0] > values[:-1, 0]).all()):
        raise ValueError(f"{path} timestamps must be strictly increasing; export each arrival once.")
    return values


def interpolate(timestamps, values, target):
    """Fixed-observation linear interpolation, with no extrapolation."""
    if bool((target < timestamps[0]).any()) or bool((target > timestamps[-1]).any()):
        raise ValueError("Recorded observations do not cover the requested timestamps.")
    upper = torch.searchsorted(timestamps.contiguous(), target.contiguous()).clamp(1, len(timestamps)-1)
    lower = upper - 1
    weight = ((target - timestamps[lower]) / (timestamps[upper] - timestamps[lower]))[:, None]
    return values[lower] * (1-weight) + values[upper] * weight


def _interpolate_quaternion(timestamps, quaternions, target):
    """Shortest-arc SLERP for the fixed initial Vicon attitude."""
    if float(target) < float(timestamps[0]) or float(target) > float(timestamps[-1]):
        raise ValueError("Vicon attitude does not cover initialization.")
    upper = int(torch.searchsorted(timestamps.contiguous(), target).clamp(1, len(timestamps)-1))
    lower = upper - 1
    a, b = (normalize_quaternion(quaternions[index]) for index in (lower, upper))
    if float(a @ b) < 0:
        b = -b
    dot = (a @ b).clamp(-1, 1)
    weight = (target - timestamps[lower]) / (timestamps[upper] - timestamps[lower])
    if float(dot) > .9995:
        return normalize_quaternion((1-weight)*a + weight*b)
    angle = torch.acos(dot)
    return (torch.sin((1-weight)*angle)*a + torch.sin(weight*angle)*b) / torch.sin(angle)


def prepare_flight(record, cutoff_timestamp, ekf):
    """Initialize with Vicon pose/PX4 velocity and keep genuine aiding arrivals.

    Vicon event CSV: time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,
                     quat_w,quat_x,quat_y,quat_z.
    GT CSV: time,vel_x,vel_y,vel_z (aligned data.csv is suitable for GT).
    PX4 CSV: time,vel_x,vel_y,vel_z in the SAME navigation frame.
    """
    imu = read_numeric_csv(record["imu_csv"], (
        "time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"))
    vicon = read_numeric_csv(record["vicon_csv"], (
        "time", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
        "quat_w", "quat_x", "quat_y", "quat_z"))
    gt = read_numeric_csv(record["ground_truth_csv"], ("time", "vel_x", "vel_y", "vel_z"))
    px4 = read_numeric_csv(record["px4_velocity_csv"], ("time", "vel_x", "vel_y", "vel_z"))
    start = float(record.get("start_timestamp", imu[0, 0]))
    end = float(cutoff_timestamp) + 25.
    if start >= cutoff_timestamp or float(imu[-1, 0]) < end:
        raise ValueError("Flight requires an aided start and IMU coverage through cutoff+25.")
    selected = imu[(imu[:, 0] >= start) & (imu[:, 0] <= end)]
    if len(selected) < 2 or float(selected[0, 0]) >= cutoff_timestamp:
        raise ValueError("No valid aided IMU prefix for this flight.")
    if float(selected[-1, 0]) < end:
        # Terminal output at exactly T_e; last real IMU sample is held to T_e.
        selected = torch.cat((selected, torch.cat((torch.tensor([end], dtype=torch.float64),
                                                   selected[-1, 1:]))[None]))
    times = selected[:, 0].contiguous()
    aided_vicon = vicon[vicon[:, 0] < cutoff_timestamp]
    if len(aided_vicon) < 2:
        raise ValueError("At least two original Vicon samples are needed before cutoff.")
    if not bool((aided_vicon[:, 0] > times[0]).any()):
        raise ValueError("At least one aided update must follow initialization before cutoff.")
    initial_time = times[:1]
    position = interpolate(aided_vicon[:, 0], aided_vicon[:, 1:4], initial_time)[0]
    quaternion = _interpolate_quaternion(aided_vicon[:, 0], aided_vicon[:, 7:11], times[0])
    velocity = interpolate(px4[:, 0], px4[:, 1:4], initial_time)[0]
    state, covariance = ekf.initialize(position, velocity, quaternion,
        record.get("initial_accel_bias"), record.get("initial_gyro_bias"))
    events = []
    for row in aided_vicon:
        timestamp = float(row[0])
        if timestamp < float(times[0]):
            continue
        observations = {"position": row[1:4], "velocity_xy": row[4:6],
                        "velocity": row[4:7], "yaw": row[7:11]}
        for channel in ekf.observation_types:
            events.append(ViconEvent(timestamp, channel, observations[channel]))
    gt_velocity = interpolate(gt[:, 0], gt[:, 1:4], times)
    return dict(raw_acceleration=selected[:, 1:4].to(ekf.gravity.device),
                raw_gyroscope=selected[:, 4:7].to(ekf.gravity.device),
                imu_time=times.to(ekf.gravity.device), vicon_events=events,
                cutoff_timestamp=float(cutoff_timestamp), initial_state=state,
                initial_covariance=covariance,
                vicon_velocity_gt=gt_velocity.to(ekf.gravity.device))
