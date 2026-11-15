"""Acceptance checks: python -m unittest discover -s tests -v."""

import contextlib
import io
import math
import unittest
from dataclasses import fields, replace

import torch

from ekf import (AirIMUESKF, DifferentiableESKF, EKFState, ViconEvent,
                 horizontal_velocity_loss, velocity_evaluation_mask)
from ekf.eskf import (quaternion_exp, quaternion_matrix, quaternion_multiply,
                      quaternion_yaw, right_jacobian)
from ekf.experiment import CovarianceStatistics, covariance_statistics
from ekf.parallel import replay_velocity


torch.set_num_threads(1)
DTYPE = torch.float64


def fixture():
    ekf = DifferentiableESKF()
    state, P = ekf.initialize([0., 0., 0.], [.2, -.1, 0.], [1., 0., 0., 0.])
    time = torch.arange(21, dtype=DTYPE) * .1
    imu = torch.tensor([.3, -.2, 9.9, .02, -.03, .1], dtype=DTYPE).repeat(len(time), 1)
    events = []
    for t in (.15, .41, .7, .95):
        events.extend((ViconEvent(t, "position", torch.tensor([t*.4, -.05, 90.], dtype=DTYPE)),
                       ViconEvent(t, "velocity_xy", torch.tensor([.4, .05], dtype=DTYPE)),
                       ViconEvent(t, "yaw", torch.tensor(t*.08, dtype=DTYPE))))
    return ekf, state, P, time, imu, events


def state_error(actual, nominal):
    conjugate = torch.cat((nominal.quaternion[:1], -nominal.quaternion[1:]))
    relative = quaternion_multiply(conjugate, actual.quaternion)
    return torch.cat((actual.position-nominal.position, actual.velocity-nominal.velocity,
                      2*relative[1:], actual.accelerometer_bias-nominal.accelerometer_bias,
                      actual.gyroscope_bias-nominal.gyroscope_bias))


class ESKFTests(unittest.TestCase):
    def setUp(self):
        self.ekf = DifferentiableESKF()
        self.state, self.P = self.ekf.initialize([0.]*3, [0.]*3, [1., 0., 0., 0.])
        self.acc = torch.tensor([0., 0., 9.80665], dtype=DTYPE)
        self.gyro = torch.zeros(3, dtype=DTYPE)
        self.variance = torch.full((6,), .001, dtype=DTYPE)

    def test_initialization_fixed_prior_and_no_parameters(self):
        self.assertEqual(len(list(self.ekf.parameters())), 0)
        self.assertEqual(self.P.shape, (15, 15))
        expected = torch.tensor([.1**2]*3 + [.2**2]*3 + [math.radians(5)**2]*3
                                 + [.1**2]*3 + [.02**2]*3, dtype=DTYPE)
        torch.testing.assert_close(self.P.diag(), expected)
        for field in fields(EKFState):
            value = getattr(self.state, field.name)
            self.assertTrue(torch.isfinite(value).all())
            self.assertEqual(value.dtype, DTYPE)
        state, P = self.ekf.initialize(torch.zeros(3, requires_grad=True), [0.]*3,
                                     [2., 0., 0., 0.], initial_covariance=self.P.requires_grad_())
        self.assertFalse(state.position.requires_grad)
        self.assertFalse(P.requires_grad)
        self.assertEqual(float(state.quaternion.norm()), 1.)

    def test_stationary_gravity_and_irregular_timestamps(self):
        time = torch.tensor([0., .003, .010, .017, .029], dtype=DTYPE)
        imu = torch.cat((self.acc, self.gyro)).repeat(len(time), 1)
        output = self.ekf(imu, time, self.variance.repeat(len(time), 1), [], .02, self.state, self.P)
        torch.testing.assert_close(output["velocity"], torch.zeros(len(time), 3, dtype=DTYPE))
        torch.testing.assert_close(output["position"], torch.zeros(len(time), 3, dtype=DTYPE))
        self.assertTrue(torch.isfinite(output["covariance"]).all())
        ned = DifferentiableESKF(gravity=(0., 0., 9.80665))
        result, _ = ned.predict(self.state, self.P, -self.acc, self.gyro, self.variance, .01)
        torch.testing.assert_close(result.velocity, torch.zeros(3, dtype=DTYPE))

    def test_angular_motion_and_normalization(self):
        gyro = torch.tensor([0., 0., math.pi/2], dtype=DTYPE)
        state, _ = self.ekf.predict(self.state, self.P, self.acc, gyro, self.variance, 1.)
        torch.testing.assert_close(state.quaternion,
            torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)], dtype=DTYPE))
        self.assertAlmostEqual(float(state.quaternion.norm()), 1., places=14)

    def test_bias_subtraction_and_constant_bias_propagation(self):
        ba = torch.tensor([.4, -.2, .1], dtype=DTYPE)
        bg = torch.tensor([.01, .02, -.03], dtype=DTYPE)
        state = replace(self.state, accelerometer_bias=ba, gyroscope_bias=bg)
        predicted, _ = self.ekf.predict(state, self.P, self.acc+ba, bg, self.variance, .2)
        torch.testing.assert_close(predicted.velocity, self.state.velocity)
        torch.testing.assert_close(predicted.quaternion, self.state.quaternion)
        torch.testing.assert_close(predicted.accelerometer_bias, ba)
        torch.testing.assert_close(predicted.gyroscope_bias, bg)

    def test_covariance_psd_symmetry_and_noise_units(self):
        zero_P = torch.zeros_like(self.P)
        for dt in (.005, .013):
            _, P = self.ekf.predict(self.state, zero_P, self.acc, self.gyro, self.variance, dt)
            torch.testing.assert_close(P, P.T, atol=1e-14, rtol=0)
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(P).min()), -1e-14)
            torch.testing.assert_close(P[3:6, 3:6].diag(), self.variance[:3]*dt**2)
            torch.testing.assert_close(P[6:9, 6:9].diag(), self.variance[3:]*dt**2)
            torch.testing.assert_close(P[9:12, 9:12].diag(), torch.full((3,), .002**2*dt, dtype=DTYPE))
            torch.testing.assert_close(P[:3, 3:6], torch.eye(3, dtype=DTYPE)*.5*.001*dt**3)

    def test_transition_matches_discrete_nominal_finite_difference(self):
        state = replace(self.state, quaternion=quaternion_exp(torch.tensor([.2, -.4, .3], dtype=DTYPE)),
                        accelerometer_bias=torch.tensor([.1, .2, -.1], dtype=DTYPE),
                        gyroscope_bias=torch.tensor([.1, -.05, .03], dtype=DTYPE))
        gyro = torch.tensor([.4, .2, -.2], dtype=DTYPE)
        dt = torch.tensor(.12, dtype=DTYPE)
        F = self.ekf.compute_state_transition(state, self.acc, gyro, dt)
        nominal = self.ekf.propagate_nominal_state(state, self.acc, gyro, dt)
        columns = []
        for axis in range(15):
            delta = torch.eye(15, dtype=DTYPE)[axis] * 1e-6
            plus = self.ekf.propagate_nominal_state(
                self.ekf.inject_error_state(state, delta), self.acc, gyro, dt)
            minus = self.ekf.propagate_nominal_state(
                self.ekf.inject_error_state(state, -delta), self.acc, gyro, dt)
            columns.append((state_error(plus, nominal)-state_error(minus, nominal)) / 2e-6)
        torch.testing.assert_close(F, torch.stack(columns, dim=1), atol=2e-9, rtol=2e-7)

    def test_noise_map_matches_sample_increment_finite_difference(self):
        state = replace(self.state, quaternion=quaternion_exp(torch.tensor([.2, -.1, .3], dtype=DTYPE)))
        gyro = torch.tensor([.4, -.2, .1], dtype=DTYPE)
        dt = torch.tensor(.13, dtype=DTYPE)
        nominal = self.ekf.propagate_nominal_state(state, self.acc, gyro, dt)
        G = self.ekf.compute_noise_jacobian(state, self.acc, gyro, dt)
        columns = []
        for axis in range(6):
            increment = torch.eye(6, dtype=DTYPE)[axis]*1e-6
            plus = self.ekf.propagate_nominal_state(state, self.acc-increment[:3]/dt,
                                                   gyro-increment[3:]/dt, dt)
            minus = self.ekf.propagate_nominal_state(state, self.acc+increment[:3]/dt,
                                                    gyro+increment[3:]/dt, dt)
            columns.append((state_error(plus, nominal)-state_error(minus, nominal))/2e-6)
        torch.testing.assert_close(G[:, :6], torch.stack(columns, dim=1), atol=3e-9, rtol=2e-7)

    def test_position_update_ignores_height(self):
        observation = torch.tensor([1., -1., 500.], dtype=DTYPE)
        state, P = self.ekf.update_vicon(self.state, self.P, observation, "position")
        self.assertLess(float((state.position[:2]-observation[:2]).norm()), .02)
        self.assertEqual(float(state.position[2]), 0.)
        torch.testing.assert_close(P, P.T)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(P).min()), 0.)

    def test_velocity_update_and_optional_xyz(self):
        observation = torch.tensor([1., -2., 3.], dtype=DTYPE)
        state, _ = self.ekf.update_vicon(self.state, self.P, observation, "velocity_xy")
        self.assertLess(float((state.velocity[:2]-observation[:2]).norm()), .15)
        self.assertEqual(float(state.velocity[2]), 0.)
        ekf = DifferentiableESKF(observation_types=("velocity",))
        state, _ = ekf.update_vicon(self.state, self.P, observation, "velocity")
        self.assertLess(float((state.velocity-observation).norm()), .23)

    def test_yaw_wrap_and_right_error_measurement_jacobian(self):
        state = replace(self.state, quaternion=quaternion_exp(
            torch.tensor([0., 0., math.radians(179)], dtype=DTYPE)))
        residual, _ = self.ekf.measurement_model(state, torch.tensor(math.radians(-179)), "yaw")
        self.assertAlmostEqual(float(residual), math.radians(2), places=6)
        state, _ = self.ekf.update_vicon(state, self.P, torch.tensor(math.radians(-179)), "yaw")
        yaw_error = quaternion_yaw(state.quaternion) - math.radians(-179)
        self.assertLess(abs(float(torch.atan2(yaw_error.sin(), yaw_error.cos()))), math.radians(1))
        state = replace(self.state, quaternion=quaternion_exp(torch.tensor([.4, -.3, .2], dtype=DTYPE)))
        _, H = self.ekf.measurement_model(state, torch.tensor(0.), "yaw")
        for axis in range(3):
            delta = torch.eye(15, dtype=DTYPE)[6+axis]*1e-6
            plus = quaternion_yaw(self.ekf.inject_error_state(state, delta).quaternion)
            minus = quaternion_yaw(self.ekf.inject_error_state(state, -delta).quaternion)
            self.assertAlmostEqual(float(H[0, 6+axis]), float((plus-minus)/2e-6), places=8)

    def test_attitude_reset_matches_finite_difference(self):
        delta = torch.tensor([.2, -.1, .4], dtype=DTYPE)
        q_nom = quaternion_exp(delta)
        nominal = replace(self.state, quaternion=q_nom)
        columns = []
        for axis in range(3):
            perturbation = torch.eye(3, dtype=DTYPE)[axis]*1e-6
            plus = replace(self.state, quaternion=quaternion_exp(delta+perturbation))
            minus = replace(self.state, quaternion=quaternion_exp(delta-perturbation))
            columns.append((state_error(plus, nominal)[6:9]-state_error(minus, nominal)[6:9])/2e-6)
        torch.testing.assert_close(right_jacobian(delta), torch.stack(columns, 1), atol=1e-10, rtol=1e-8)

    def test_zero_rotation_autograd_is_finite(self):
        phi = torch.zeros(3, dtype=DTYPE, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(quaternion_exp, (phi,)))
        self.assertTrue(torch.autograd.gradcheck(right_jacobian, (phi,)))

    def test_state_dependencies_in_jacobians_and_covariance(self):
        perturbation = torch.zeros(15, dtype=DTYPE, requires_grad=True)
        state = self.ekf.inject_error_state(self.state, perturbation)
        gyro = torch.tensor([.3, -.2, .1], dtype=DTYPE)
        predicted = self.ekf.predict_covariance(state, self.P, self.acc, gyro,
                                               self.variance, torch.tensor(.04, dtype=DTYPE))
        gradient, = torch.autograd.grad((predicted*torch.arange(225, dtype=DTYPE).reshape(15, 15)).sum(),
                                       perturbation)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient[6:15].norm()), 0.)

    def test_invalid_initialization_and_precision(self):
        with self.assertRaises(ValueError):
            DifferentiableESKF(dtype=torch.float32)
        with self.assertRaises(ValueError):
            self.ekf.initialize([0.]*3, [0.]*3, [0.]*4)
        with self.assertRaises(ValueError):
            self.ekf.initialize([0.]*3, [0.]*3, [1., 0., 0., 0.], initial_covariance=-self.P)
        with self.assertRaises(ValueError):
            self.ekf.predict(self.state, self.P, self.acc, self.gyro, self.variance, 0.)


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.ekf, self.state, self.P, self.time, self.imu, self.events = fixture()

    def run_filter(self, variance, events=None, cutoff=1.):
        return self.ekf.run_sequence(
            self.imu, self.time, variance,
            self.events if events is None else events, cutoff, self.state, self.P,
        )

    def objective(self, variance):
        result = self.run_filter(variance)
        return horizontal_velocity_loss(result["velocity"], torch.zeros(len(self.time), 3, dtype=DTYPE),
                                        self.time >= 1.2)

    def test_complete_history_finite_and_psd(self):
        variance = torch.full((len(self.time), 6), .001, dtype=DTYPE, requires_grad=True)
        result = self.run_filter(variance)
        self.assertEqual(set(result), {field.name for field in fields(EKFState)} | {"covariance"})
        for key, value in result.items():
            self.assertEqual(len(value), len(self.time))
            self.assertEqual(value.dtype, DTYPE)
            self.assertTrue(torch.isfinite(value).all(), key)
        torch.testing.assert_close(result["quaternion"].norm(dim=1), torch.ones(len(self.time), dtype=DTYPE))
        P = result["covariance"]
        torch.testing.assert_close(P, P.transpose(-1, -2))
        self.assertGreater(float(torch.linalg.eigvalsh(P).min()), -1e-12)
        gradient, = torch.autograd.grad(P[-1].trace(), variance)
        self.assertGreater(float(gradient[self.time >= 1.].norm()), 0.)

    def test_strict_cutoff_and_post_cutoff_observation_independence(self):
        variance = torch.full((len(self.time), 6), .001, dtype=DTYPE)
        extra = [ViconEvent(1., "velocity_xy", torch.tensor([999., 999.])),
                 ViconEvent(1.5, "position", torch.tensor([999., 999., 999.])),
                 ViconEvent(2., "yaw", torch.tensor(-2.))]
        a, b = self.run_filter(variance), self.run_filter(variance, self.events + extra)
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)

    def test_cutoff_has_no_reset_or_state_detachment(self):
        variance = torch.full((len(self.time), 6), .001, dtype=DTYPE)
        a, b = self.run_filter(variance, [], .85), self.run_filter(variance, [], 1.15)
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        self.assertGreater(float(a["position"][-1].norm()), 0.)

    def test_event_inside_interval_matches_manual_split_and_fuses_once(self):
        time = torch.tensor([0., .2, .4], dtype=DTYPE)
        event = ViconEvent(.075, "velocity_xy", torch.tensor([.8, -.2], dtype=DTYPE))
        variance = torch.full((3, 6), .01, dtype=DTYPE)
        result = self.ekf(self.imu[:3], time, variance, [event], .3, self.state, self.P)
        state, P = self.ekf.predict(self.state, self.P, self.imu[0, :3], self.imu[0, 3:], variance[0], .075)
        state, P = self.ekf.update_vicon(state, P, event.observation, event.observation_type)
        state, P = self.ekf.predict(state, P, self.imu[0, :3], self.imu[0, 3:], variance[0], .125)
        state, P = self.ekf.predict(state, P, self.imu[1, :3], self.imu[1, 3:], variance[1], .2)
        for field in fields(EKFState):
            torch.testing.assert_close(result[field.name][-1], getattr(state, field.name))
        torch.testing.assert_close(result["covariance"][-1], P)

    def test_boundary_and_initial_events_applied_once(self):
        event = ViconEvent(0., "velocity_xy", torch.tensor([.5, .6], dtype=DTYPE))
        boundary = ViconEvent(.1, "position", torch.tensor([.2, .3, 50.], dtype=DTYPE))
        variance = torch.full((len(self.time), 6), .001, dtype=DTYPE)
        result = self.run_filter(variance, [event, boundary])
        state, P = self.ekf.update_vicon(self.state, self.P, event.observation, event.observation_type)
        torch.testing.assert_close(result["velocity"][0], state.velocity)
        state, P = self.ekf.predict(state, P, self.imu[0, :3], self.imu[0, 3:], variance[0], .1)
        state, P = self.ekf.update_vicon(state, P, boundary.observation, boundary.observation_type)
        torch.testing.assert_close(result["position"][1], state.position)

    def test_velocity_gradients_before_cutoff_and_zero_after(self):
        variance = torch.full((len(self.time), 6), .001, dtype=DTYPE, requires_grad=True)
        self.objective(variance).backward()
        self.assertIsNotNone(variance.grad)
        self.assertTrue(torch.isfinite(variance.grad).all())
        self.assertGreater(float(variance.grad[self.time < 1.].norm()), 1e-10)
        torch.testing.assert_close(variance.grad[self.time >= 1.],
                                  torch.zeros_like(variance.grad[self.time >= 1.]), atol=0, rtol=0)

    def test_full_200hz_replay_through_cutoff_plus_25_seconds(self):
        times = torch.arange(5201, dtype=DTYPE) / 200.
        imu = torch.tensor([.12, -.04, 9.80665, .005, -.008, .01], dtype=DTYPE).repeat(len(times), 1)
        variance = torch.full((len(times), 6), .001, dtype=DTYPE, requires_grad=True)
        result = self.ekf(imu, times, variance, self.events, 1., self.state, self.P)
        mask = velocity_evaluation_mask(times, 1.)
        loss = horizontal_velocity_loss(result["velocity"], torch.zeros(len(times), 3, dtype=DTYPE), mask)
        loss.backward()
        self.assertIsNotNone(variance.grad)
        self.assertTrue(torch.isfinite(variance.grad).all())
        self.assertGreater(float(variance.grad[times < 1.].norm()), 0.)
        torch.testing.assert_close(variance.grad[times >= 1.],
                                  torch.zeros_like(variance.grad[times >= 1.]), atol=0, rtol=0)
        self.assertEqual(result["covariance"].shape, (5201, 15, 15))

    def test_velocity_only_replay_preserves_states_and_complete_gradients(self):
        variance = torch.full((len(self.time), 6), .003, dtype=DTYPE, requires_grad=True)
        full = self.run_filter(variance)
        fast = self.ekf.run_sequence(self.imu, self.time, variance, self.events, 1.,
                                    self.state, self.P, return_covariance=False)
        self.assertNotIn("covariance", fast)
        for key in fast:
            torch.testing.assert_close(fast[key], full[key], atol=0, rtol=0)
        mask = self.time >= 1.2
        gt = torch.zeros(len(self.time), 3, dtype=DTYPE)
        a, = torch.autograd.grad(horizontal_velocity_loss(full["velocity"], gt, mask), variance)
        b, = torch.autograd.grad(horizontal_velocity_loss(fast["velocity"], gt, mask), variance)
        torch.testing.assert_close(a, b, atol=1e-15, rtol=1e-12)
        self.assertGreater(float(b[self.time < 1.].norm()), 0.)
        torch.testing.assert_close(b[self.time >= 1.], torch.zeros_like(b[self.time >= 1.]), atol=0, rtol=0)

    def test_batched_quaternion_aiding_matches_public_update_and_gradient(self):
        quaternion = quaternion_exp(torch.tensor([.2, -.1, .3], dtype=DTYPE)).requires_grad_()
        event = ViconEvent(0., "yaw", quaternion)
        variance = torch.full((len(self.time), 6), .001, dtype=DTYPE)
        result = self.run_filter(variance, [event])
        expected, P = self.ekf.update_vicon(self.state, self.P, quaternion, "yaw")
        torch.testing.assert_close(result["quaternion"][0], expected.quaternion, atol=0, rtol=0)
        torch.testing.assert_close(result["covariance"][0], P, atol=0, rtol=0)
        a, = torch.autograd.grad(result["quaternion"][0].sum(), quaternion, retain_graph=True)
        b, = torch.autograd.grad(expected.quaternion.sum(), quaternion)
        torch.testing.assert_close(a, b, atol=1e-14, rtol=1e-12)

    def test_sequence_rejects_nonfinite_and_zero_quaternion_observations(self):
        variance = torch.full((len(self.time), 6), .001, dtype=DTYPE)
        for observation in (torch.zeros(4), torch.tensor([float("nan")])):
            with self.assertRaises(ValueError):
                self.run_filter(variance, [ViconEvent(.1, "yaw", observation)])

    def test_covariance_statistics_are_per_axis_and_sample_weighted(self):
        a = torch.arange(6, dtype=DTYPE)[None]
        b = torch.arange(12, dtype=DTYPE).reshape(2, 6) + 10
        aggregate = CovarianceStatistics()
        aggregate.update(covariance_statistics(a))
        aggregate.update(covariance_statistics(b))
        expected = covariance_statistics(torch.cat((a, b)))
        actual = aggregate.summary()
        self.assertEqual(actual["samples"], 3)
        for name in ("acc_cov", "gyro_cov"):
            for stat in ("mean", "min", "max"):
                torch.testing.assert_close(torch.tensor(actual[name][stat]), torch.tensor(expected[name][stat]))

    def test_autograd_agrees_with_finite_difference(self):
        variance = torch.full((len(self.time), 6), .005, dtype=DTYPE, requires_grad=True)
        gradient, = torch.autograd.grad(self.objective(variance), variance)
        for index in ((0, 0), (3, 1), (4, 3), (8, 4), (15, 0)):
            step = torch.zeros_like(variance)
            step[index] = 1e-5
            finite = (self.objective(variance+step)-self.objective(variance-step))/2e-5
            torch.testing.assert_close(gradient[index], finite, atol=2e-9, rtol=3e-4)

    def test_pre_cutoff_noise_changes_bias_and_unaided_velocity(self):
        small = torch.full((len(self.time), 6), .001, dtype=DTYPE)
        large = torch.cat((small[:10]*100., small[10:]))
        a, b = self.run_filter(small), self.run_filter(large)
        self.assertGreater(float((a["velocity"][-1]-b["velocity"][-1]).norm()), 1e-6)
        self.assertGreater(float((a["accelerometer_bias"][-1]-b["accelerometer_bias"][-1]).norm()), 1e-6)
        torch.testing.assert_close(a["accelerometer_bias"][10:], a["accelerometer_bias"][10:11].expand(11, -1))

    def test_post_cutoff_noise_changes_covariance_only(self):
        a = torch.full((len(self.time), 6), .001, dtype=DTYPE)
        b = torch.cat((a[:10], a[10:]*1e6))
        result_a, result_b = self.run_filter(a), self.run_filter(b)
        for field in fields(EKFState):
            torch.testing.assert_close(result_a[field.name], result_b[field.name], atol=0, rtol=0)
        self.assertGreater(float((result_a["covariance"][-1]-result_b["covariance"][-1]).norm()), 1.)

    def test_invalid_schedule_and_variance(self):
        variance = torch.full((len(self.time), 6), .001, dtype=DTYPE)
        with self.assertRaises(ValueError):
            self.run_filter(-variance)
        with self.assertRaises(ValueError):
            self.run_filter(variance, self.events[::-1])
        with self.assertRaises(ValueError):
            self.run_filter(variance, [self.events[0], self.events[0]])
        with self.assertRaises(ValueError):
            self.run_filter(variance, [], 0.)

    def test_loss_uses_xy_sum_and_exact_evaluation_horizon(self):
        time = torch.arange(28, dtype=DTYPE)
        mask = velocity_evaluation_mask(time, 1.)
        self.assertEqual(int(mask.sum()), 21)
        velocity = torch.tensor([1., 2., 900.], dtype=DTYPE).repeat(len(time), 1)
        self.assertEqual(float(horizontal_velocity_loss(velocity, torch.zeros_like(velocity), mask)), 5.)
        with self.assertRaises(ValueError):
            velocity_evaluation_mask(time[:20], 1.)

    def test_actual_airimu_heads_receive_gradients_and_only_heads_update(self):
        from model.code import CodeNet
        from pyhocon import ConfigFactory
        conf = ConfigFactory.from_dict(dict(network="codenet", propcov=True, gtrot=False))
        with contextlib.redirect_stdout(io.StringIO()):
            network = CodeNet(conf).float()
        wrapper = AirIMUESKF(network, self.ekf)
        wrapper.train()
        self.assertEqual(len(list(wrapper.ekf.parameters())), 0)
        frozen_before = {name: value.clone() for name, value in network.state_dict().items()
                         if not name.startswith(("acccov_decoder", "gyrocov_decoder", "acc_bias_gain", "gyro_bias_gain"))}
        times = torch.arange(55, dtype=DTYPE)*.5  # Includes full cutoff+5..cutoff+25 interval.
        imu = self.imu[0].repeat(len(times), 1)
        optimizer = torch.optim.Adam((p for p in wrapper.parameters() if p.requires_grad), lr=1e-3)
        with contextlib.redirect_stdout(io.StringIO()):
            output = wrapper(imu[:, :3], imu[:, 3:], times, self.events, 1., self.state,
                             self.P, torch.zeros(len(times), 3, dtype=DTYPE))
        output["imu_variance"].retain_grad()
        output["loss"].backward()
        self.assertEqual(output["trajectory"]["velocity"].dtype, DTYPE)
        self.assertTrue(torch.isfinite(output["imu_variance"].grad).all())
        self.assertGreater(float(output["imu_variance"].grad[times < 1.].norm()), 0.)
        torch.testing.assert_close(output["imu_variance"].grad[times >= 1.],
                                  torch.zeros_like(output["imu_variance"].grad[times >= 1.]), atol=0, rtol=0)
        for name, parameter in network.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertIsNone(parameter.grad, name)
        self.assertGreater(float(network.acccov_decoder[-1].bias.grad.norm()), 0.)
        optimizer.step()
        for name, value in frozen_before.items():
            torch.testing.assert_close(network.state_dict()[name], value, atol=0, rtol=0)

    def make_wrapper(self):
        from model.code import CodeNet
        from pyhocon import ConfigFactory
        conf = ConfigFactory.from_dict(dict(network="codenet", propcov=True, gtrot=False))
        with contextlib.redirect_stdout(io.StringIO()):
            network = CodeNet(conf).float()
        with torch.no_grad():
            network.acccov_decoder[-1].weight.normal_(std=.01)
            network.gyrocov_decoder[-1].weight.normal_(std=.01)
            network.acc_bias_gain.weight.normal_(std=.1)
            network.gyro_bias_gain.weight.normal_(std=.1)
        return AirIMUESKF(network, self.ekf)

    def test_gather_covariance_matches_original_inference_and_gradients(self):
        wrapper = self.make_wrapper()
        parameters = [p for p in wrapper.network.parameters() if p.requires_grad]
        for samples in (1, 4, 5, 6, 9, 13, 14, 15, 55, 200):
            imu = self.imu[0].repeat(samples, 1)
            context = wrapper.network.interval
            data = {name: torch.cat((values[:1].expand(context, -1), values), 0)[None].float()
                    for name, values in (("acc", imu[:, :3]), ("gyro", imu[:, 3:]))}
            with contextlib.redirect_stdout(io.StringIO()):
                output = wrapper.network.inference(data)["cov_state"]
            expected = torch.cat((output["acc_cov"][0], output["gyro_cov"][0]), -1).double()
            actual = wrapper.predict_variance(imu[:, :3], imu[:, 3:])
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            weights = torch.linspace(.1, 1., samples*6, dtype=DTYPE).reshape(samples, 6)
            a = torch.autograd.grad((expected*weights).sum(), parameters)
            b = torch.autograd.grad((actual*weights).sum(), parameters)
            for left, right in zip(a, b):
                torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-5)

    def test_batched_head_vjp_matches_direct_mean_full_flight_gradient(self):
        wrapper = self.make_wrapper()
        parameters = [p for p in wrapper.network.parameters() if p.requires_grad]
        flights = []
        for samples, scale in ((55, 1.), (58, 1.03)):
            imu = self.imu[0].repeat(samples, 1)*scale
            flights.append(dict(raw_acceleration=imu[:, :3], raw_gyroscope=imu[:, 3:],
                imu_time=torch.arange(samples, dtype=DTYPE)*.5, vicon_events=self.events,
                cutoff_timestamp=1., initial_state=self.state, initial_covariance=self.P,
                vicon_velocity_gt=torch.zeros(samples, 3, dtype=DTYPE)))
        direct = [wrapper(**flight, return_covariance=False)["loss"] for flight in flights]
        expected = torch.autograd.grad(torch.stack(direct).mean(), parameters)
        features = [wrapper.extract_features(flight["raw_acceleration"], flight["raw_gyroscope"])
                    for flight in flights]
        self.assertTrue(all(not value.requires_grad for item in features for value in item.values()
                            if torch.is_tensor(value)))
        variances = wrapper.predict_variance_batch(features)
        outputs = [replay_velocity(self.ekf, flight, variance, backward=True)
                   for flight, variance in zip(flights, variances)]
        actual = torch.autograd.grad(variances, parameters,
                                    grad_outputs=[row["gradient"]/2 for row in outputs])
        for row, loss in zip(outputs, direct):
            # Batching changes Float32 GEMM reduction order in the heads.
            torch.testing.assert_close(torch.tensor(row["loss"], dtype=DTYPE), loss.detach(),
                                       atol=1e-6, rtol=1e-7)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, atol=3e-6, rtol=5e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_heads_receive_cpu_replay_gradient(self):
        wrapper = self.make_wrapper()
        wrapper.network.to("cuda")
        samples = 55
        imu = self.imu[0].repeat(samples, 1)
        flight = dict(raw_acceleration=imu[:, :3], raw_gyroscope=imu[:, 3:],
            imu_time=torch.arange(samples, dtype=DTYPE)*.5, vicon_events=self.events,
            cutoff_timestamp=1., initial_state=self.state, initial_covariance=self.P,
            vicon_velocity_gt=torch.zeros(samples, 3, dtype=DTYPE))
        parameters = [p for p in wrapper.network.parameters() if p.requires_grad]
        direct = wrapper(**flight, return_covariance=False)
        expected = torch.autograd.grad(direct["loss"], parameters)
        variance = wrapper.predict_variance(imu[:, :3], imu[:, 3:])
        self.assertEqual(variance.device.type, "cpu")
        output = replay_velocity(self.ekf, flight, variance, backward=True)
        actual = torch.autograd.grad(variance, parameters, grad_outputs=output["gradient"])
        for left, right in zip(actual, expected):
            self.assertEqual(left.device.type, "cuda")
            torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
