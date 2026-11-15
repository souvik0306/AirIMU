"""Offline flight preparation from the standard TLab dataset layout.

CSV schemas use SI units, seconds in a shared reference, and Hamilton wxyz
body-to-navigation quaternions. No automatic frame or timestamp conversion is
performed.
"""

import csv
import math
from pathlib import Path
import re

from pyhocon import ConfigFactory
import torch

from datasets.TLabdataset import TLab

from .eskf import ViconEvent


_DATASET_METADATA_DIRECTORIES = {
    "T-Lab_31st_July_dataset": "2026-07-31",
    "T-Lab_28th_July_dataset": "2026-07-28",
}

_SEPTEMBER_METADATA_DIRECTORIES = {
    "High-Dynamic": "10th_Sept_High",
    "Hover": "10th_Sept_AI_Hover_Rosbags",
    "Low-Dynamic": "10th_Sept_Low-Dynamic",
    "Medium-Dynamic": "10th_Sept_Medium",
    "Super-High-Dynamic": "10th_Sept_Super-High",
    "Yaw": "10th_Sept_AI_Yaw_Rosbags",
}

_BIAS_COLUMNS = (
    "accel_bias_x_mps2", "accel_bias_y_mps2", "accel_bias_z_mps2",
    "gyro_bias_x_radps", "gyro_bias_y_radps", "gyro_bias_z_radps",
)


def _sensor_metadata_location(flight_id):
    """Map a stable dataset flight id to its exported metadata table row."""
    parts = str(flight_id).replace("\\", "/").strip("/").split("/")
    dataset = parts[0] if parts else ""
    if dataset in _DATASET_METADATA_DIRECTORIES:
        if len(parts) != 3:
            raise ValueError(f"Expected DATASET/GROUP/flight_N, got {flight_id!r}.")
        directory = _DATASET_METADATA_DIRECTORIES[dataset]
        group, flight_name = parts[1:]
    elif dataset == "T-Lab_10th_September_dataset":
        if len(parts) != 4:
            raise ValueError(
                f"Expected DATASET/CATEGORY/GROUP/flight_N, got {flight_id!r}."
            )
        category, group, flight_name = parts[1:]
        if category not in _SEPTEMBER_METADATA_DIRECTORIES:
            raise KeyError(f"No sensor-metadata mapping for category {category!r}.")
        directory = _SEPTEMBER_METADATA_DIRECTORIES[category]
    else:
        raise KeyError(f"No sensor-metadata mapping for dataset {dataset!r}.")

    group = group.upper()
    if group == "UN":
        group = "AI"
    if group not in ("AI", "RAW"):
        raise KeyError(f"Unsupported sensor-metadata group {group!r}.")
    match = re.fullmatch(r"flight_(\d+)", flight_name)
    if match is None:
        raise ValueError(f"Cannot determine flight number from {flight_id!r}.")
    return directory, group, int(match.group(1))


def read_sensor_metadata_table(path):
    """Read the finalized bag/ULG, cutoff, sample-count, and bias schema."""
    path = Path(path).resolve()
    required = {
        "bag_file", "ulg_file", "group", "flight_number",
        "cutoff_rosbag_s", "cutoff_ros_epoch_s", "bias_samples",
        *_BIAS_COLUMNS,
    }
    records = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for line, row in enumerate(reader, 2):
            try:
                group = row["group"].strip().upper()
                flight_number = int(row["flight_number"])
                bias_samples = int(row["bias_samples"])
                cutoff_rosbag_s = float(row["cutoff_rosbag_s"])
                cutoff_ros_epoch_s = float(row["cutoff_ros_epoch_s"])
                bias = tuple(float(row[column]) for column in _BIAS_COLUMNS)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid sensor metadata at {path}:{line}: {error}") from error
            bag_file, ulg_file = row["bag_file"].strip(), row["ulg_file"].strip()
            numeric = bias + (cutoff_rosbag_s, cutoff_ros_epoch_s)
            key = (group, flight_number)
            if (not bag_file or not ulg_file or group not in ("AI", "RAW") or
                    flight_number < 1 or bias_samples < 1 or cutoff_rosbag_s < 0 or
                    cutoff_ros_epoch_s <= 0 or not all(math.isfinite(x) for x in numeric) or
                    key in records):
                raise ValueError(f"Invalid or duplicate sensor metadata at {path}:{line}.")
            records[key] = {
                "bag_file": bag_file,
                "ulg_file": ulg_file,
                "group": group,
                "flight_number": flight_number,
                "bias_samples": bias_samples,
                "cutoff_rosbag_s": cutoff_rosbag_s,
                "cutoff_ros_epoch_s": cutoff_ros_epoch_s,
                "initial_accel_bias": list(bias[:3]),
                "initial_gyro_bias": list(bias[3:]),
                "source_csv": str(path),
            }
    if not records:
        raise ValueError(f"Sensor metadata table is empty: {path}")
    return records


def apply_sensor_metadata(flights, root):
    """Attach fixed biases and exact epoch cutoffs to dataset flight records."""
    root = Path(root).resolve()
    tables = {}
    enriched, cutoffs = [], {}
    for original in flights:
        record = dict(original)
        directory, group, flight_number = _sensor_metadata_location(record["id"])
        table_path = root / directory / f"initial_sensor_bias_{group}.csv"
        if table_path not in tables:
            tables[table_path] = read_sensor_metadata_table(table_path)
        key = (group, flight_number)
        if key not in tables[table_path]:
            raise KeyError(f"No sensor metadata for {record['id']} in {table_path}.")
        metadata = dict(tables[table_path][key])
        record["initial_accel_bias"] = metadata["initial_accel_bias"]
        record["initial_gyro_bias"] = metadata["initial_gyro_bias"]
        record["sensor_metadata"] = metadata
        cutoffs[record["id"]] = metadata["cutoff_ros_epoch_s"]
        enriched.append(record)
    return enriched, cutoffs


def read_dataset_flights(path):
    """Build ESKF records from the train/test/eval TLab dataset config."""
    config = ConfigFactory.parse_file(str(Path(path).resolve()))
    if "dataset" in config:
        config = config["dataset"]
    split_names = {"train": "train", "test": "validation", "eval": "test"}
    flights = []
    seen = set()
    for config_split, eskf_split in split_names.items():
        if config_split not in config or "data_list" not in config[config_split]:
            raise ValueError(f"Dataset config requires a {config_split}.data_list section.")
        for source in config[config_split].data_list:
            if str(source.get("name", "")) != "TLab":
                continue
            data_root = Path(str(source["data_root"]))
            if not data_root.is_absolute():
                data_root = Path.cwd() / data_root
            data_root = data_root.resolve()
            for drive in source.data_drive:
                drive = str(drive).replace("\\", "/").strip("/")
                flight_id = f"{data_root.name}/{drive}"
                if flight_id in seen:
                    raise ValueError(f"Flight appears in more than one split: {flight_id}")
                seen.add(flight_id)
                flight_dir = data_root / drive
                imu_csv, data_csv = flight_dir / "imu.csv", flight_dir / "data.csv"
                missing = [str(item) for item in (imu_csv, data_csv) if not item.is_file()]
                if missing:
                    raise FileNotFoundError(f"Missing TLab flight files: {missing}")
                flights.append({
                    "id": flight_id,
                    "split": eskf_split,
                    "data_root": str(data_root),
                    "data_drive": drive,
                })
    if not flights:
        raise ValueError("Dataset config contains no TLab flights.")
    return flights


def prepare_flight(record, cutoff_timestamp, ekf):
    """Initialize and replay the standard aligned TLab IMU/Vicon files.

    data.csv: time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,
              quat_w,quat_x,quat_y,quat_z.
    """
    sequence = TLab(record["data_root"], record["data_drive"])
    imu_time = sequence.data["time"]
    imu = torch.cat((imu_time[:, None], sequence.data["acc"], sequence.data["gyro"]), dim=1)
    vicon = torch.cat((
        sequence.data["gt_time"][:, None], sequence.data["gt_translation"],
        sequence.data["velocity"], torch.as_tensor(sequence.data["quat"], dtype=torch.float64),
    ), dim=1)
    if not bool((imu_time[1:] > imu_time[:-1]).all()):
        raise ValueError("TLab timestamps must be strictly increasing.")
    start = float(record.get("start_timestamp", imu[0, 0]))
    end = float(cutoff_timestamp) + 25.
    if start >= cutoff_timestamp or float(imu[-1, 0]) < end:
        raise ValueError("Flight requires an aided start and IMU coverage through cutoff+25.")
    selected_mask = (imu[:, 0] >= start) & (imu[:, 0] <= end)
    selected = imu[selected_mask]
    selected_vicon = vicon[selected_mask]
    if len(selected) < 2 or float(selected[0, 0]) >= cutoff_timestamp:
        raise ValueError("No valid aided IMU prefix for this flight.")
    terminal_velocity = None
    if float(selected[-1, 0]) < end:
        # Terminal output at exactly T_e; last real IMU sample is held to T_e.
        upper = int(torch.searchsorted(vicon[:, 0].contiguous(), vicon.new_tensor(end)))
        lower = upper - 1
        weight = (end - vicon[lower, 0]) / (vicon[upper, 0] - vicon[lower, 0])
        terminal_velocity = torch.lerp(vicon[lower, 4:7], vicon[upper, 4:7], weight)
        selected = torch.cat((selected, torch.cat((torch.tensor([end], dtype=torch.float64),
                                                   selected[-1, 1:]))[None]))
    times = selected[:, 0].contiguous()
    aided_vicon = selected_vicon[selected_vicon[:, 0] < cutoff_timestamp]
    if len(aided_vicon) < 2:
        raise ValueError("At least two aligned Vicon samples are needed before cutoff.")
    if not bool((aided_vicon[:, 0] > times[0]).any()):
        raise ValueError("At least one aided update must follow initialization before cutoff.")
    position, velocity, quaternion = aided_vicon[0, 1:4], aided_vicon[0, 4:7], aided_vicon[0, 7:11]
    state, covariance = ekf.initialize(position, velocity, quaternion,
        record.get("initial_accel_bias"), record.get("initial_gyro_bias"))
    events = []
    for row in aided_vicon:
        timestamp = float(row[0])
        if timestamp <= float(times[0]):
            continue
        observations = {"position": row[1:4], "velocity_xy": row[4:6],
                        "velocity": row[4:7], "yaw": row[7:11]}
        for channel in ekf.observation_types:
            events.append(ViconEvent(timestamp, channel, observations[channel]))
    gt_velocity = selected_vicon[:, 4:7]
    if terminal_velocity is not None:
        gt_velocity = torch.cat((gt_velocity, terminal_velocity[None]))
    return dict(raw_acceleration=selected[:, 1:4].to(ekf.gravity.device),
                raw_gyroscope=selected[:, 4:7].to(ekf.gravity.device),
                imu_time=times.to(ekf.gravity.device), vicon_events=events,
                cutoff_timestamp=float(cutoff_timestamp), initial_state=state,
                initial_covariance=covariance,
                vicon_velocity_gt=gt_velocity.to(ekf.gravity.device))
