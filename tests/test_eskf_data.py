"""Input preparation checks, including cutoff tables and distinct PX4 velocity."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch

from ekf import DifferentiableESKF
from ekf.data import (interpolate, prepare_flight, read_cutoff_table,
                      read_flight_manifest, read_numeric_csv)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header.split(","))
        writer.writerows(rows)


class FlightDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_cutoff_table_keys_and_duplicate_rejection(self):
        path = self.root / "cutoffs.csv"
        header = "dataset,data_name,cutoff_timestamp"
        write_csv(path, header, [("T-Lab_10th_September_dataset", "High-Dynamic/AI/flight_1", 1234.5)])
        self.assertEqual(read_cutoff_table(path), {
            "T-Lab_10th_September_dataset/High-Dynamic/AI/flight_1": 1234.5})
        write_csv(path, header, [("collection", "AI/flight_1", 1.), ("collection", "AI/flight_1", 2.)])
        with self.assertRaises(ValueError):
            read_cutoff_table(path)

    def test_interpolation_refuses_extrapolation_and_arrival_duplicates(self):
        time = torch.tensor([0., 1., 2.], dtype=torch.float64)
        values = torch.stack((time, time*2), dim=-1)
        result = interpolate(time, values, torch.tensor([.5, 2.], dtype=torch.float64))
        torch.testing.assert_close(result, torch.tensor([[.5, 1.], [2., 4.]], dtype=torch.float64))
        with self.assertRaises(ValueError):
            interpolate(time, values, torch.tensor([3.], dtype=torch.float64))
        path = self.root / "repeated.csv"
        write_csv(path, "time,x", [(0., 1.), (0., 2.)])
        with self.assertRaises(ValueError):
            read_numeric_csv(path, ("time", "x"))

    def make_record(self):
        write_csv(self.root / "imu.csv", "time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z",
                  [(float(k), 0., 0., 9.80665, 0., 0., 0.) for k in range(29)])
        write_csv(self.root / "vicon.csv", "time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,quat_w,quat_x,quat_y,quat_z",
                  [(t, .5*t, 0., 1., .5, 0., 0., sign, 0., 0., 0.)
                   for t, sign in ((0., 1.), (.5, -1.), (1.5, 1.), (2.5, -1.), (28., 1.))])
        write_csv(self.root / "gt.csv", "time,vel_x,vel_y,vel_z", [(0., .5, 0., 0.), (28., .5, 0., 0.)])
        write_csv(self.root / "px4.csv", "time,vel_x,vel_y,vel_z", [(0., 1., 2., 3.), (28., 1., 2., 3.)])
        return {"id": "collection/AI/flight_1", "split": "train", "start_timestamp": 1.,
                "imu_csv": str(self.root / "imu.csv"), "vicon_csv": str(self.root / "vicon.csv"),
                "ground_truth_csv": str(self.root / "gt.csv"), "px4_velocity_csv": str(self.root / "px4.csv")}

    def test_pose_initialization_px4_velocity_and_terminal_timestamp(self):
        ekf = DifferentiableESKF()
        record = self.make_record()
        flight = prepare_flight(record, 2.2, ekf)
        state = flight["initial_state"]
        torch.testing.assert_close(state.position, torch.tensor([.5, 0., 1.], dtype=torch.float64))
        torch.testing.assert_close(state.velocity, torch.tensor([1., 2., 3.], dtype=torch.float64))
        self.assertAlmostEqual(abs(float(state.quaternion[0])), 1.)
        self.assertEqual(float(flight["imu_time"][-1]), 27.2)
        self.assertEqual(len(flight["vicon_events"]), 3)
        self.assertTrue(all(event.timestamp == 1.5 for event in flight["vicon_events"]))
        self.assertEqual(flight["raw_acceleration"].dtype, torch.float64)
        self.assertFalse(state.velocity.requires_grad)
        torch.testing.assert_close(state.accelerometer_bias, torch.zeros(3, dtype=torch.float64))

    def test_optional_initial_bias_and_missing_recording_coverage(self):
        record = self.make_record()
        record["initial_accel_bias"] = [.1, .2, .3]
        flight = prepare_flight(record, 2.2, DifferentiableESKF())
        torch.testing.assert_close(flight["initial_state"].accelerometer_bias,
                                  torch.tensor([.1, .2, .3], dtype=torch.float64))
        with self.assertRaises(ValueError):
            prepare_flight(record, 10., DifferentiableESKF())

    def test_manifest_path_resolution_and_split_uniqueness(self):
        record = self.make_record()
        record["imu_csv"] = "imu.csv"
        manifest = self.root / "flights.json"
        manifest.write_text(json.dumps([record]))
        flights = read_flight_manifest(manifest)
        self.assertEqual(flights[0]["imu_csv"], str(self.root / "imu.csv"))
        manifest.write_text(json.dumps([record, {**record, "split": "validation"}]))
        with self.assertRaises(ValueError):
            read_flight_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
