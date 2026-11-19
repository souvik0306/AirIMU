"""Keep date/category/group mappings identical in Stage 2 and EKF loading."""

import csv
from pathlib import Path
import tempfile
import unittest

from datasets.initial_sensor_bias import resolve_initial_sensor_bias
from ekf.data import apply_sensor_metadata


class OctoberSensorBiasTests(unittest.TestCase):
    def test_each_collection_category_and_group_uses_its_own_bias_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = []
            for day in (14, 15):
                for category, suffix in (("High-Dynamic", "High"),
                                         ("Low-Dynamic", "Low"),
                                         ("Medium-Dynamic", "Medium")):
                    for group in ("AI", "RAW"):
                        # Every file has flight_1: a wrong mapping still finds
                        # a row, but its distinctive bias/cutoff exposes it.
                        value = len(expected) + 1.
                        table = root / f"{day}th_Oct_{suffix}" / f"initial_sensor_bias_{group}.csv"
                        table.parent.mkdir(exist_ok=True)
                        with table.open("w", newline="") as handle:
                            writer = csv.writer(handle)
                            writer.writerow(("group", "flight_number", "bag_file", "ulg_file",
                                             "cutoff_rosbag_s", "cutoff_ros_epoch_s", "bias_samples",
                                             "accel_bias_x_mps2", "accel_bias_y_mps2", "accel_bias_z_mps2",
                                             "gyro_bias_x_radps", "gyro_bias_y_radps", "gyro_bias_z_radps"))
                            writer.writerow((group, 1, "flight_1.bag", "log_1.ulg", value,
                                             1000.+value, 2, value, value+1, value+2,
                                             value+3, value+4, value+5))
                        dataset = f"T-Lab_{day}th_October_dataset"
                        drive = f"{category}/{group}/flight_1"
                        expected.append((dataset, drive, table.resolve(), value))
            for dataset, drive, table, value in expected:
                with self.subTest(dataset=dataset, drive=drive):
                    bias = resolve_initial_sensor_bias(root, dataset, drive)
                    record = {"id": f"{dataset}/{drive}"}
                    flights, cutoffs = apply_sensor_metadata([record], root)
                    self.assertEqual(Path(bias.source_file), table)
                    self.assertEqual(Path(flights[0]["sensor_metadata"]["source_csv"]), table)
                    self.assertEqual(bias.acc, (value, value+1, value+2))
                    self.assertEqual(bias.gyro, (value+3, value+4, value+5))
                    self.assertEqual(flights[0]["initial_accel_bias"], list(bias.acc))
                    self.assertEqual(flights[0]["initial_gyro_bias"], list(bias.gyro))
                    self.assertEqual(cutoffs[record["id"]], bias.cutoff_ros_epoch_s)
                    self.assertEqual(bias.cutoff_ros_epoch_s, 1000.+value)
            with self.assertRaises(KeyError):
                resolve_initial_sensor_bias(root, "T-Lab_14th_October_dataset", "High-Dynamic/AI/flight_2")
            with self.assertRaises(KeyError):
                resolve_initial_sensor_bias(root, "T-Lab_14th_October_dataset", "Hover/AI/flight_1")


if __name__ == "__main__":
    unittest.main()
