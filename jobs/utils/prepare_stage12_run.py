"""Validate flight splits and freeze the inputs for a scratch training run."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from pyhocon import ConfigFactory, HOCONConverter

from datasets.initial_sensor_bias import resolve_initial_sensor_bias


def flight_map(dataset, split):
    flights = {}
    for source in dataset[split].data_list:
        root = Path(source.data_root).resolve()
        for drive in source.data_drive:
            path = root / str(drive)
            key = f"{root.name}/{drive}"
            if key in flights:
                raise ValueError(f"Duplicate {split} flight: {key}")
            for name in ("imu.csv", "data.csv"):
                if not (path / name).is_file():
                    raise FileNotFoundError(path / name)
            flights[key] = (root, str(drive))
    return flights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-config", required=True)
    parser.add_argument("--stage2-config", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=("stage1", "stage2", "all"), default="all")
    parser.add_argument("--sensor-bias-root")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / "dataset.conf"
    dataset = ConfigFactory.parse_file(str(dataset_path if dataset_path.exists() else args.dataset_config))
    splits = {split: flight_map(dataset, split) for split in ("train", "test", "eval", "inference")}
    used = set()
    for split in ("train", "test", "eval"):
        overlap = used.intersection(splits[split])
        if overlap:
            raise ValueError(f"Flight overlap in {split}: {sorted(overlap)}")
        used.update(splits[split])
    if used != set(splits["inference"]):
        raise ValueError("Inference must cover exactly the union of train/test/eval flights.")
    roots = {root for root, _ in splits["inference"].values()}
    on_disk = {f"{root.name}/{imu.parent.relative_to(root)}"
               for root in roots for imu in root.rglob("imu.csv")}
    if used != on_disk:
        raise ValueError(f"Dataset inventory mismatch: omitted={sorted(on_disk-used)}, extra={sorted(used-on_disk)}")
    if not dataset_path.exists():
        dataset_path.write_text(HOCONConverter.to_json(dataset) + "\n")

    report = {"splits": {split: len(flights) for split, flights in splits.items()},
              "flights": {split: sorted(flights) for split, flights in splits.items()},
              "dataset_snapshot": str(dataset_path), "bias_enabled_in_stage2": True,
              "use_cov_limits": False, "variance_max": None}
    for stage, source in (("stage1", args.stage1_config), ("stage2", args.stage2_config)):
        snapshot = output / f"{stage}.conf"
        conf = ConfigFactory.parse_file(str(snapshot if snapshot.exists() else source))
        collate = conf.dataset.get("collate", "padding9")
        conf["dataset"] = ConfigFactory.from_dict(dataset.as_plain_ordered_dict())
        conf.dataset["collate"] = collate
        if conf.train.get("use_cov_limits", True):
            raise ValueError(f"{stage} must have use_cov_limits=False.")
        for name in ("acc_cov_min", "gyro_cov_min"):
            if float(conf.train[name]) <= 0:
                raise ValueError(f"{stage} requires positive {name}.")
        if stage == "stage2":
            if not conf.train.get("use_initial_sensor_bias", False):
                raise ValueError("Stage 2 must enable initial sensor bias subtraction.")
            if args.sensor_bias_root:
                conf.train["initial_sensor_bias_root"] = str(Path(args.sensor_bias_root).resolve())
            if args.stage != "stage1":
                source_root = Path(conf.train.initial_sensor_bias_root).resolve()
                biases = {key: resolve_initial_sensor_bias(source_root, root, drive)
                          for key, (root, drive) in splits["inference"].items()}
                bias_snapshot = output / "initial_sensor_biases"
                tables = {}
                for bias in biases.values():
                    source_table = Path(bias.source_file).resolve()
                    relative = source_table.relative_to(source_root)
                    destination = bias_snapshot / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if source_table != destination:
                        if destination.exists() and destination.read_bytes() != source_table.read_bytes():
                            raise ValueError(f"A frozen bias table differs from the supplied input: {destination}")
                        if not destination.exists():
                            shutil.copy2(source_table, destination)
                    tables[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
                conf.train["initial_sensor_bias_root"] = str(bias_snapshot)
                report["sensor_bias_snapshot_root"] = str(bias_snapshot)
                report["sensor_bias_flights"] = len(biases)
                report["sensor_bias_tables"] = len(tables)
                manifest = {"source_root": str(source_root), "snapshot_root": str(bias_snapshot),
                            "table_sha256": tables,
                            "flights": {key: {"acc": bias.acc, "gyro": bias.gyro,
                                               "csv": str(Path(bias.source_file).resolve().relative_to(source_root)),
                                               "flight_number": bias.flight_number,
                                               "cutoff_ros_epoch_s": bias.cutoff_ros_epoch_s}
                                        for key, bias in biases.items()}}
                manifest_path = output / "sensor_bias_manifest.json"
                if not manifest_path.exists():
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            else:
                # Freeze Stage 2 only once its metadata has been resolved.
                continue
        if not snapshot.exists():
            snapshot.write_text(HOCONConverter.to_json(conf) + "\n")
        report[f"{stage}_snapshot"] = str(snapshot)
    (output / f"preflight_{args.stage}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "flights"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
