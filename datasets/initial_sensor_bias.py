"""Resolve one fixed EKF2 accelerometer/gyroscope bias per TLab flight."""

import csv
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache


_DATASET_BIAS_DIRECTORIES = {
    "T-Lab_31st_July_dataset": "2026-07-31",
    "T-Lab_28th_July_dataset": "2026-07-28",
}

_SEPTEMBER_BIAS_DIRECTORIES = {
    "High-Dynamic": "10th_Sept_High",
    "Hover": "10th_Sept_AI_Hover_Rosbags",
    "Low-Dynamic": "10th_Sept_Low-Dynamic",
    "Medium-Dynamic": "10th_Sept_Medium",
    "Super-High-Dynamic": "10th_Sept_Super-High",
    "Yaw": "10th_Sept_AI_Yaw_Rosbags",
}

_CATEGORY_BIAS_DIRECTORIES = {
    "T-Lab_10th_September_dataset": _SEPTEMBER_BIAS_DIRECTORIES,
    "T-Lab_14th_October_dataset": {
        "High-Dynamic": "14th_Oct_High",
        "Low-Dynamic": "14th_Oct_Low",
        "Medium-Dynamic": "14th_Oct_Medium",
    },
    "T-Lab_15th_October_dataset": {
        "High-Dynamic": "15th_Oct_High",
        "Low-Dynamic": "15th_Oct_Low",
        "Medium-Dynamic": "15th_Oct_Medium",
    },
}

_BIAS_COLUMNS = (
    "accel_bias_x_mps2",
    "accel_bias_y_mps2",
    "accel_bias_z_mps2",
    "gyro_bias_x_radps",
    "gyro_bias_y_radps",
    "gyro_bias_z_radps",
)


@dataclass(frozen=True)
class InitialSensorBias:
    acc: tuple
    gyro: tuple
    source_file: str
    source_flight_name: str
    bag_file: str
    ulg_file: str
    group: str
    flight_number: int
    samples: int
    cutoff_rosbag_s: float
    cutoff_ros_epoch_s: float


def _flight_number(data_name):
    flight_name = data_name.replace("\\", "/").rstrip("/").split("/")[-1]
    match = re.fullmatch(r"flight_(\d+)", flight_name)
    if match is None:
        raise ValueError(
            f"Cannot determine a flight number from TLab sequence {data_name!r}."
        )
    return int(match.group(1))


def _bias_location(data_root, data_name):
    dataset_name = os.path.basename(os.path.normpath(str(data_root)))
    parts = data_name.replace("\\", "/").strip("/").split("/")

    if dataset_name in _DATASET_BIAS_DIRECTORIES:
        if len(parts) != 2:
            raise ValueError(
                f"Expected GROUP/flight_N for {dataset_name}, got {data_name!r}."
            )
        bias_directory = _DATASET_BIAS_DIRECTORIES[dataset_name]
        group = parts[0].upper()
    elif dataset_name in _CATEGORY_BIAS_DIRECTORIES:
        if len(parts) != 3:
            raise ValueError(
                f"Expected CATEGORY/GROUP/flight_N for {dataset_name}, "
                f"got {data_name!r}."
            )
        category, group = parts[0], parts[1].upper()
        directories = _CATEGORY_BIAS_DIRECTORIES[dataset_name]
        if category not in directories:
            raise KeyError(
                f"No initial-bias directory mapping is defined for {category!r}."
            )
        bias_directory = directories[category]
    else:
        raise KeyError(
            f"No initial-bias mapping is defined for dataset {dataset_name!r}."
        )

    # The July data folders call the AI recordings "UN". The bias export uses
    # its original AI group name; flight_number is the shared stable key.
    if group == "UN":
        group = "AI"
    if group not in ("AI", "RAW"):
        raise KeyError(f"Unsupported TLab bias group {group!r} for {data_name!r}.")

    return bias_directory, group, _flight_number(data_name)


@lru_cache(maxsize=None)
def _read_bias_table(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Initial sensor bias table was not found: {path}")

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "bag_file",
            "ulg_file",
            "group",
            "flight_number",
            "cutoff_rosbag_s",
            "cutoff_ros_epoch_s",
            "bias_samples",
            *_BIAS_COLUMNS,
        }
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        records = {}
        for row_number, row in enumerate(reader, start=2):
            try:
                flight_number = int(row["flight_number"])
                values = tuple(float(row[column]) for column in _BIAS_COLUMNS)
                samples = int(row["bias_samples"])
                cutoff_rosbag_s = float(row["cutoff_rosbag_s"])
                cutoff_ros_epoch_s = float(row["cutoff_ros_epoch_s"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid initial sensor bias at {path}:{row_number}: {error}"
                ) from error

            numeric_values = values + (cutoff_rosbag_s, cutoff_ros_epoch_s)
            if not all(math.isfinite(value) for value in numeric_values):
                raise ValueError(
                    f"Non-finite sensor metadata at {path}:{row_number}."
                )
            bag_file = row["bag_file"].strip()
            ulg_file = row["ulg_file"].strip()
            if not bag_file or not ulg_file:
                raise ValueError(
                    f"bag_file and ulg_file must be set at {path}:{row_number}."
                )
            if cutoff_rosbag_s < 0 or cutoff_ros_epoch_s <= 0:
                raise ValueError(
                    f"Invalid cutoff timestamp at {path}:{row_number}."
                )
            if samples <= 0:
                raise ValueError(
                    f"bias_samples must be positive at {path}:{row_number}."
                )
            if flight_number in records:
                raise ValueError(
                    f"Duplicate flight_number {flight_number} in {path}."
                )

            records[flight_number] = InitialSensorBias(
                acc=values[:3],
                gyro=values[3:],
                source_file=path,
                source_flight_name=bag_file,
                bag_file=bag_file,
                ulg_file=ulg_file,
                group=row["group"].upper(),
                flight_number=flight_number,
                samples=samples,
                cutoff_rosbag_s=cutoff_rosbag_s,
                cutoff_ros_epoch_s=cutoff_ros_epoch_s,
            )

    return records


def resolve_initial_sensor_bias(bias_root, data_root, data_name):
    """Return the fixed bias matching ``data_root/data_name``.

    Matching deliberately uses the recording collection, AI/RAW group, and
    flight number. The bag-derived ``flight_name`` is not always the same as
    the normalized ``flight_N`` directory used by the training dataset.
    """
    if not bias_root:
        raise ValueError("initial_sensor_bias_root must be configured.")

    bias_directory, group, flight_number = _bias_location(data_root, data_name)
    table_path = os.path.abspath(
        os.path.join(
            str(bias_root), bias_directory, f"initial_sensor_bias_{group}.csv"
        )
    )
    records = _read_bias_table(table_path)
    if flight_number not in records:
        raise KeyError(
            f"No {group} initial sensor bias for flight {flight_number} "
            f"in {table_path}."
        )

    record = records[flight_number]
    if record.group != group:
        raise ValueError(
            f"Bias group mismatch in {table_path}: expected {group}, "
            f"got {record.group}."
        )
    return record
