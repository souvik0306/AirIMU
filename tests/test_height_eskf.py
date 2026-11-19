"""Height-only aiding, continuous covariance and post-cutoff velocity gradients."""

import contextlib
import copy
import csv
import io
from pathlib import Path
import tempfile
from time import perf_counter
import unittest
from unittest.mock import patch

import torch

from ekf import AirIMUESKF, DifferentiableESKF, ViconEvent
from ekf.compiled import velocity_sequence
from ekf.data import prepare_flight
from ekf.eskf import quaternion_exp
from ekf.runner import FlightTrainer
from ekf.training import horizontal_velocity_loss


torch.set_num_threads(1)
DTYPE = torch.float64


def fixture():
    ekf = DifferentiableESKF(height_updates=True)
    times = torch.arange(21, dtype=DTYPE) * .04
    cutoff = float(times[6])
    state, P = ekf.initialize([0., 0., .1], [.2, -.1, .3],
        quaternion_exp(torch.tensor([.18, -.13, .1], dtype=DTYPE)),
        [.01, -.02, .03], [.001, -.002, .003])
    imu = torch.tensor([1., .4, 9.65, .2, -.1, .15], dtype=DTYPE).repeat(len(times), 1)
    events = []
    for t in times[1:].tolist():
        if t < cutoff:
            events.extend((ViconEvent(t, "position", torch.tensor([.2*t, -.1*t], dtype=DTYPE)),
                           ViconEvent(t, "velocity_xy", torch.tensor([.2, -.1], dtype=DTYPE)),
                           ViconEvent(t, "yaw", torch.tensor(.1+.15*t, dtype=DTYPE))))
        events.append(ViconEvent(t, "height", torch.tensor([.08+.2*t+.08*t*t], dtype=DTYPE)))
    flight = dict(raw_acceleration=imu[:, :3], raw_gyroscope=imu[:, 3:], imu_time=times,
                  vicon_events=events, cutoff_timestamp=cutoff,
                  initial_state=state, initial_covariance=P)
    return ekf, flight


def reference(ekf, flight, variance, return_covariance=False):
    return ekf.run_sequence({"acc": flight["raw_acceleration"], "gyro": flight["raw_gyroscope"]},
        flight["imu_time"], variance, flight["vicon_events"], flight["cutoff_timestamp"],
        flight["initial_state"], flight["initial_covariance"], return_covariance=return_covariance)


def objective(velocity, flight):
    gt = velocity.new_tensor([.1, -.05, 0.]).repeat(len(velocity), 1)
    mask = (flight["imu_time"] >= flight["cutoff_timestamp"]+.08).to(velocity.device)
    return horizontal_velocity_loss(velocity, gt, mask)


class HeightTests(unittest.TestCase):
    def test_scalar_height_model_and_horizontal_covariance_coupling(self):
        ekf, flight = fixture()
        state, P = flight["initial_state"], flight["initial_covariance"].clone()
        P[2, 3] = P[3, 2] = .005
        residual, H = ekf.measurement_model(state, torch.tensor([1.], dtype=DTYPE), "height")
        torch.testing.assert_close(H, torch.eye(15, dtype=DTYPE)[2:3])
        self.assertAlmostEqual(float(ekf.height_R[0, 0]), .01)
        corrected, covariance = ekf.update_vicon(state, P, torch.tensor([1.], dtype=DTYPE), "height")
        self.assertGreater(float(corrected.position[2]), float(state.position[2]))
        self.assertGreater(float(corrected.velocity[0]), float(state.velocity[0]))
        torch.testing.assert_close(covariance, covariance.T)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(covariance).min()), -1e-12)

    def test_post_cutoff_xy_velocity_and_yaw_are_ignored(self):
        ekf, flight = fixture()
        variance = torch.full((len(flight["imu_time"]), 6), .003, dtype=DTYPE)
        expected = reference(ekf, flight, variance, True)
        changed = dict(flight)
        changed["vicon_events"] = sorted(flight["vicon_events"] + [
            ViconEvent(flight["cutoff_timestamp"], "position", torch.tensor([999., -999.])),
            ViconEvent(flight["cutoff_timestamp"], "velocity_xy", torch.tensor([999., -999.])),
            ViconEvent(flight["cutoff_timestamp"], "yaw", torch.tensor(-2.))], key=lambda e: e.timestamp)
        actual = reference(ekf, changed, variance, True)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)

    def test_pre_cutoff_height_changes_state_and_covariance(self):
        ekf, flight = fixture()
        variance = torch.full((len(flight["imu_time"]), 6), .003, dtype=DTYPE)
        actual = reference(ekf, flight, variance, True)
        post_only = dict(flight, vicon_events=[e for e in flight["vicon_events"]
            if e.observation_type != "height" or e.timestamp >= flight["cutoff_timestamp"]])
        previous = reference(ekf, post_only, variance, True)
        index = int((flight["imu_time"] < flight["cutoff_timestamp"]).sum())-1
        self.assertGreater(float((actual["position"][index]-previous["position"][index]).norm()), 1e-6)
        self.assertLess(float(actual["covariance"][index, 2, 2]), float(previous["covariance"][index, 2, 2]))
        with self.assertRaisesRegex(ValueError, "per estimate after initialization"):
            velocity_sequence(ekf, post_only, variance)

    def test_invalid_and_missing_height_observations_fail(self):
        ekf, flight = fixture()
        variance = torch.full((len(flight["imu_time"]), 6), .003, dtype=DTYPE)
        for observation in (torch.tensor([float("nan")]), torch.tensor([1., 2.])):
            changed = dict(flight)
            changed["vicon_events"] = list(flight["vicon_events"])
            changed["vicon_events"][-1] = ViconEvent(float(flight["imu_time"][-1]), "height", observation)
            with self.assertRaises(ValueError):
                reference(ekf, changed, variance)
        changed = dict(flight)
        changed["vicon_events"] = [e for e in flight["vicon_events"] if e.observation_type != "height"]
        with self.assertRaisesRegex(ValueError, "no height events"):
            reference(ekf, changed, variance)
        changed["vicon_events"] = [ViconEvent(.01, "height", torch.tensor(1.))]
        with self.assertRaisesRegex(ValueError, "no post-cutoff height"):
            reference(ekf, changed, variance)
        with self.assertRaisesRegex(ValueError, "height_updates=True"):
            reference(DifferentiableESKF(), flight, variance)

    def test_covariance_history_flag_does_not_skip_propagation(self):
        ekf, flight = fixture()
        variance = torch.full((len(flight["imu_time"]), 6), .003, dtype=DTYPE, requires_grad=True)
        full = reference(ekf, flight, variance, True)
        with patch.object(ekf, "predict_covariance", wraps=ekf.predict_covariance) as propagate:
            compact = reference(ekf, flight, variance, False)
            self.assertEqual(propagate.call_count, len(flight["imu_time"])-1)
        for name in compact:
            torch.testing.assert_close(compact[name], full[name], atol=0, rtol=0)
        a, = torch.autograd.grad(objective(full["velocity"], flight), variance)
        b, = torch.autograd.grad(objective(compact["velocity"], flight), variance)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        # The no-height reference must also keep propagating P when history is omitted.
        legacy = DifferentiableESKF()
        no_height = dict(flight, vicon_events=[e for e in flight["vicon_events"] if e.observation_type != "height"])
        with patch.object(legacy, "predict_covariance", wraps=legacy.predict_covariance) as propagate:
            reference(legacy, no_height, variance, False)
            self.assertEqual(propagate.call_count, len(flight["imu_time"])-1)

    def test_post_cutoff_velocity_gradient_matches_finite_difference(self):
        ekf, flight = fixture()
        variance = torch.full((len(flight["imu_time"]), 6), .003, dtype=DTYPE, requires_grad=True)
        result = reference(ekf, flight, variance)
        gradient, = torch.autograd.grad(objective(result["velocity"], flight), variance)
        post = flight["imu_time"] >= flight["cutoff_timestamp"]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient[post].norm()), 1e-9)
        candidates = gradient.abs() * post[:, None]
        row, axis = divmod(int(candidates.argmax()), 6)
        step = 1e-6
        plus, minus = variance.detach().clone(), variance.detach().clone()
        plus[row, axis] += step
        minus[row, axis] -= step
        finite = (objective(reference(ekf, flight, plus)["velocity"], flight)
                  - objective(reference(ekf, flight, minus)["velocity"], flight)) / (2*step)
        torch.testing.assert_close(gradient[row, axis], finite, atol=1e-7, rtol=1e-3)

    def check_compiled(self, device):
        ekf, flight = fixture()
        ekf.to(device)
        variance = torch.full((len(flight["imu_time"]), 6), .003, dtype=DTYPE,
                              device=device, requires_grad=True)
        expected = reference(ekf, flight, variance)["velocity"]
        a, = torch.autograd.grad(objective(expected, flight), variance)
        actual = velocity_sequence(ekf, flight, variance)
        b, = torch.autograd.grad(objective(actual, flight), variance)
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-9)
        torch.testing.assert_close(b, a, atol=1e-10, rtol=1e-7)
        post = (flight["imu_time"] >= flight["cutoff_timestamp"]).to(device)
        self.assertGreater(float(b[post].norm()), 1e-9)
        with torch.no_grad():
            validation = velocity_sequence(ekf, flight, variance.detach())
        torch.testing.assert_close(validation, expected, atol=1e-11, rtol=1e-9)

    def test_compiled_cpu_height_forward_backward_and_validation(self):
        self.check_compiled("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_compiled_cuda_height_forward_backward_and_validation(self):
        self.check_compiled("cuda:0")

    def test_covariance_heads_receive_only_post_cutoff_gradient(self):
        from model.code import CodeNet
        from pyhocon import ConfigFactory
        torch.manual_seed(123)
        ekf, flight = fixture()
        with contextlib.redirect_stdout(io.StringIO()):
            network = CodeNet(ConfigFactory.from_dict(dict(propcov=True, gtrot=False,
                              use_cov_limits=False, network="codenet"))).float()
        wrapper = AirIMUESKF(network, ekf)
        predicted = wrapper.predict_variance(flight["raw_acceleration"], flight["raw_gyroscope"])
        predicted.retain_grad()
        post = flight["imu_time"] >= flight["cutoff_timestamp"]
        effective = torch.where(post[:, None], predicted, predicted.detach())
        objective(reference(ekf, flight, effective)["velocity"], flight).backward()
        torch.testing.assert_close(predicted.grad[~post], torch.zeros_like(predicted.grad[~post]), atol=0, rtol=0)
        self.assertGreater(float(predicted.grad[post].norm()), 1e-9)
        gradients = [p.grad for p in network.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
        self.assertTrue(all(p.grad is None for p in network.parameters() if not p.requires_grad))


class HeightDataAndWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def recording(self, number):
        directory = self.root / str(number)
        directory.mkdir()
        columns = [("imu.csv", "time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z",
                    [(1000+k*.1, .2, -.1, 9.81, .01, -.02, .03) for k in range(300)]),
                   ("data.csv", "time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,quat_w,quat_x,quat_y,quat_z",
                    [(1000+k*.1, .02*k, 0., 3.+.001*k*k, .2, 0., .02*k, 1., 0., 0., 0.)
                     for k in range(300)])]
        for name, header, rows in columns:
            with (directory/name).open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(header.split(","))
                writer.writerows(rows)
        return {"id": f"fixture/{number}", "split": "train", "data_root": str(self.root), "data_drive": str(number)}

    def test_clean_height_at_each_estimate_and_terminal_interpolation(self):
        record = self.recording(1)
        flight = prepare_flight(record, 1002.03, DifferentiableESKF(height_updates=True))
        height = [e for e in flight["vicon_events"] if e.observation_type == "height"]
        times = flight["imu_time"]
        self.assertEqual([e.timestamp for e in height], times[1:].tolist())
        self.assertTrue(all(e.observation_type == "height" for e in flight["vicon_events"] if e.timestamp >= 1002.03))
        self.assertGreater(sum(e.timestamp < 1002.03 for e in height), 0)
        for timestamp in times[(times > times[0]) & (times < 1002.03)].tolist():
            self.assertEqual([e.observation_type for e in flight["vicon_events"] if e.timestamp == timestamp],
                             ["position", "velocity_xy", "yaw", "height"])
        for event in height[:-1]:
            self.assertAlmostEqual(float(event.observation), .1*(event.timestamp-1000)**2, places=9)
        self.assertAlmostEqual(float(height[-1].observation), .7*(.1*27**2)+.3*(.1*27.1**2), places=9)

    def test_parallel_height_training_matches_serial(self):
        from model.code import CodeNet
        from pyhocon import ConfigFactory
        records = [self.recording(i) for i in (1, 2)]
        cutoffs = {r["id"]: 1002.03 for r in records}
        with contextlib.redirect_stdout(io.StringIO()):
            network = CodeNet(ConfigFactory.from_dict(dict(network="codenet", propcov=True,
                                                          gtrot=False, use_cov_limits=False))).float()
        wrappers = [AirIMUESKF(network, DifferentiableESKF(height_updates=True))]
        wrappers.append(copy.deepcopy(wrappers[0]))
        losses = []
        for wrapper, workers in zip(wrappers, (0, 2)):
            optimizer = torch.optim.Adam([p for p in wrapper.parameters() if p.requires_grad], lr=1e-5)
            logged = []
            with contextlib.redirect_stdout(io.StringIO()):
                with FlightTrainer(wrapper, optimizer, workers=workers, batch_size=2) as trainer:
                    loss, _ = trainer.run_split(records, cutoffs, epoch=1, phase="train",
                        epoch_start=perf_counter(), log=logged.append)
            self.assertTrue(all(row["height_update_count"] > 200 for row in logged))
            losses.append(loss)
        torch.testing.assert_close(torch.tensor(losses[0]), torch.tensor(losses[1]))
        for name, value in wrappers[0].network.state_dict().items():
            torch.testing.assert_close(value, wrappers[1].network.state_dict()[name], atol=1e-8, rtol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
