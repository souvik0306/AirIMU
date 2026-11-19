"""Input preparation checks for the standard TLab dataset layout."""

import csv
import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
import unittest
from time import perf_counter

import torch

from ekf import DifferentiableESKF
from ekf.data import (apply_sensor_metadata, prepare_flight, read_dataset_flights,
                      read_sensor_metadata_table)
from ekf.runner import FlightTrainer


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

    def test_finalized_sensor_metadata_supplies_biases_and_epoch_cutoff(self):
        metadata_root = self.root / "metadata"
        table = metadata_root / "2026-07-31" / "initial_sensor_bias_AI.csv"
        table.parent.mkdir(parents=True)
        header = (
            "bag_file,ulg_file,group,flight_number,cutoff_rosbag_s,"
            "cutoff_ros_epoch_s,bias_samples,accel_bias_x_mps2,"
            "accel_bias_y_mps2,accel_bias_z_mps2,gyro_bias_x_radps,"
            "gyro_bias_y_radps,gyro_bias_z_radps"
        )
        write_csv(table, header, [
            ("flight_1.bag", "log_1.ulg", "AI", 1, 2.2, 1002.2, 200,
             .1, .2, .3, .01, .02, .03)
        ])
        parsed = read_sensor_metadata_table(table)
        self.assertEqual(parsed[("AI", 1)]["bag_file"], "flight_1.bag")
        record = self.make_record()
        record["id"] = "T-Lab_31st_July_dataset/UN/flight_1"
        flights, cutoffs = apply_sensor_metadata([record], metadata_root)
        self.assertEqual(cutoffs[record["id"]], 1002.2)
        self.assertEqual(flights[0]["initial_accel_bias"], [.1, .2, .3])
        self.assertEqual(flights[0]["initial_gyro_bias"], [.01, .02, .03])
        self.assertEqual(flights[0]["sensor_metadata"]["bias_samples"], 200)

    def make_record(self):
        write_csv(self.root / "imu.csv", "time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z",
                  [(float(k), 0., 0., 9.80665, 0., 0., 0.) for k in range(29)])
        write_csv(self.root / "data.csv", "time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,quat_w,quat_x,quat_y,quat_z",
                  [(float(k), .5*k, 0., 1., .5, 0., 0., 1., 0., 0., 0.)
                   for k in range(29)])
        return {"id": "collection/AI/flight_1", "split": "train", "start_timestamp": 1.,
                "data_root": str(self.root), "data_drive": "."}

    def test_pose_velocity_initialization_and_terminal_timestamp(self):
        ekf = DifferentiableESKF()
        record = self.make_record()
        flight = prepare_flight(record, 2.2, ekf)
        state = flight["initial_state"]
        # The shared TLab loader translates the flight's first position to the origin.
        torch.testing.assert_close(state.position, torch.tensor([.5, 0., 0.], dtype=torch.float64))
        torch.testing.assert_close(state.velocity, torch.tensor([.5, 0., 0.], dtype=torch.float64))
        self.assertAlmostEqual(abs(float(state.quaternion[0])), 1.)
        self.assertEqual(float(flight["imu_time"][-1]), 27.2)
        self.assertEqual(len(flight["vicon_events"]), 3)
        self.assertTrue(all(event.timestamp == 2. for event in flight["vicon_events"]))
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

    def test_dataset_config_paths_and_split_uniqueness(self):
        roots = {}
        for split in ("train", "test", "eval"):
            root = self.root / f"TLab_{split}"
            flight = root / "AI" / "flight_1"
            flight.mkdir(parents=True)
            write_csv(flight / "imu.csv", "time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z",
                      [(0., 0., 0., 0., 0., 0., 0.), (1., 0., 0., 0., 0., 0., 0.)])
            write_csv(flight / "data.csv", "time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,quat_w,quat_x,quat_y,quat_z",
                      [(0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.),
                       (1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.)])
            roots[split] = root
        config = self.root / "dataset.conf"
        config.write_text("\n".join(
            f'{split}: {{ data_list: [{{ name: TLab, data_root: "{root}", '
            'data_drive: [AI/flight_1] }] }'
            for split, root in roots.items()
        ))
        flights = read_dataset_flights(config)
        self.assertEqual([record["split"] for record in flights],
                         ["train", "validation", "test"])
        self.assertEqual(flights[0]["data_root"], str(roots["train"].resolve()))
        self.assertEqual(flights[0]["data_drive"], "AI/flight_1")

        duplicate = self.root / "duplicate.conf"
        duplicate.write_text("\n".join(
            f'{split}: {{ data_list: [{{ name: TLab, data_root: "{roots["train"]}", '
            'data_drive: [AI/flight_1] }] }'
            for split in ("train", "test", "eval")
        ))
        with self.assertRaises(ValueError):
            read_dataset_flights(duplicate)

    def test_parallel_trainer_matches_serial_batch_and_reuses_frozen_features(self):
        from model.code import CodeNet
        from ekf import AirIMUESKF
        from pyhocon import ConfigFactory
        records = []
        for index in (1, 2):
            directory = self.root / str(index)
            directory.mkdir()
            write_csv(directory/"imu.csv", "time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z",
                      [(k*.5, .01*index, -.02, 9.80665, .001, .002, .003) for k in range(62)])
            write_csv(directory/"data.csv", "time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,quat_w,quat_x,quat_y,quat_z",
                      [(k*.5, k*.1, 0., 0., .2, 0., 0., 1., 0., 0., 0.) for k in range(62)])
            records.append({"id": f"fixture/{index}", "split": "train",
                            "data_root": str(self.root), "data_drive": str(index)})
        cutoffs = {record["id"]: 2.2 for record in records}
        with contextlib.redirect_stdout(io.StringIO()):
            network = CodeNet(ConfigFactory.from_dict(dict(network="codenet", propcov=True, gtrot=False))).float()
        serial = AirIMUESKF(network, DifferentiableESKF())
        parallel = copy.deepcopy(serial)
        serial_optimizer = torch.optim.Adam([p for p in serial.parameters() if p.requires_grad], lr=1e-5)
        parallel_optimizer = torch.optim.Adam([p for p in parallel.parameters() if p.requires_grad], lr=1e-5)
        logged = []
        with contextlib.redirect_stdout(io.StringIO()):
            with FlightTrainer(serial, serial_optimizer, batch_size=2) as trainer:
                serial_losses, _ = trainer.run_split(records, cutoffs, epoch=1, phase="train",
                    epoch_start=perf_counter(), log=lambda row: None)
                self.assertEqual(trainer.optimizer_steps, 1)
            with FlightTrainer(parallel, parallel_optimizer, workers=2, batch_size=2) as trainer:
                parallel_losses, statistics = trainer.run_split(records, cutoffs, epoch=1, phase="train",
                    epoch_start=perf_counter(), log=logged.append)
                misses = trainer.cache_misses
                trainer.run_split(records, cutoffs, epoch=1, phase="validation",
                    epoch_start=perf_counter(), log=lambda row: None)
                self.assertEqual(trainer.cache_misses, misses)
                self.assertEqual(trainer.cache_hits, 2)
                self.assertLessEqual(trainer.feature_bytes, trainer.feature_limit)
                self.assertEqual(trainer.optimizer_steps, 1)
        torch.testing.assert_close(torch.tensor(serial_losses), torch.tensor(parallel_losses))
        self.assertEqual(statistics["samples"], 112)
        self.assertTrue(all(row["worker_pid"] != os.getpid() for row in logged))
        for name, value in serial.network.state_dict().items():
            torch.testing.assert_close(value, parallel.network.state_dict()[name], atol=1e-8, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
