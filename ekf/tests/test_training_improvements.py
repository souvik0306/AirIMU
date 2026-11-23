"""Independent calibration, GRU, adapter and smoothness regressions."""

import contextlib
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from pyhocon import ConfigFactory
from scipy.spatial.transform import Rotation
import torch
import pypose as pp

from ekf import AirIMUESKF, DifferentiableESKF, ViconEvent
from ekf.compiled import velocity_sequence
from ekf.parallel import replay_velocity
from ekf.regularization import gyro_reference, gyro_calibration_loss, log_variance_smoothness
from ekf.runner import FlightTrainer
from ekf.training import horizontal_velocity_loss, velocity_evaluation_mask
from ekf.experiment import validate_gru1_evidence
from ekf.gradients import component_gradient_diagnostics
from ekf.scripts.compare_training_runs import validate_matched_ab, summarize, paired_report, main as compare_main
from model.checkpoints import load_covariance_checkpoint, configure_stage2_parameters
from model.code import CodeNet


torch.set_num_threads(1)


def network(**overrides):
    conf = ConfigFactory.from_dict(dict(network="codenet", propcov=True, gtrot=False,
        use_cov_limits=False, lr=1e-5, **overrides))
    with contextlib.redirect_stdout(io.StringIO()):
        return CodeNet(conf).float()


def flight_fixture():
    ekf = DifferentiableESKF(height_updates=True)
    state, P = ekf.initialize([0., 0., .1], [.2, -.1, .3], [1., .1, -.06, .03],
                              [.01, -.02, .03], [.001, -.002, .003])
    time = torch.arange(56, dtype=torch.float64)*.5
    imu = torch.tensor([1., .4, 9.65, .02, -.01, .015], dtype=torch.float64).repeat(len(time), 1)
    events = []
    for t in time[1:].tolist():
        if t < 1.5:
            events.extend((ViconEvent(t, "position", torch.tensor([.2*t, -.1*t], dtype=torch.float64)),
                           ViconEvent(t, "velocity_xy", torch.tensor([.2, -.1], dtype=torch.float64)),
                           ViconEvent(t, "yaw", torch.tensor(.1+.015*t, dtype=torch.float64))))
        events.append(ViconEvent(t, "height", torch.tensor([.1+.02*t], dtype=torch.float64)))
    return ekf, dict(raw_acceleration=imu[:, :3], raw_gyroscope=imu[:, 3:], imu_time=time,
        vicon_events=events, cutoff_timestamp=1.5, initial_state=state, initial_covariance=P,
        vicon_velocity_gt=torch.zeros_like(imu[:, :3]),
        vicon_quaternion_gt=torch.tensor([1., 0., 0., 0.], dtype=torch.float64).repeat(len(time), 1))


class CalibrationTests(unittest.TestCase):
    def test_orientation_jump_rejects_entire_reference_window(self):
        time = np.arange(401)*.005
        rotations = Rotation.from_rotvec(time[:,None]*np.array([0.,0.,.2]))
        q = rotations.as_quat()
        q[200:] = (rotations[200:]*Rotation.from_rotvec([np.pi,0.,0.])).as_quat()
        ref = gyro_reference(time, q[:,[3,0,1,2]])
        touching = (time >= .975) & (time <= 1.02)
        self.assertFalse(bool(ref['mask'][touching].any()))
        self.assertTrue(bool(ref['mask'][(time>.1)&(time<.9)].all()))
        self.assertTrue(bool(ref['mask'][(time>1.1)&(time<1.9)].all()))
        self.assertEqual(ref['quality']['orientation_jump_intervals'],1)
        self.assertGreater(ref['quality']['orientation_jump_windows'],1)
        torch.testing.assert_close(ref['rate'][ref['mask']].norm(dim=1),
                                   torch.full((int(ref['mask'].sum()),),.2,dtype=torch.double),atol=1e-12,rtol=1e-12)

    def test_rate_quality_preserves_real_fast_rotation_and_masks_spikes(self):
        time = np.arange(401)*.005
        rotation = Rotation.from_rotvec(time[:,None]*np.array([0.,0.,10.]))
        q = rotation.as_quat()[:,[3,0,1,2]]
        q[::2] *= -1
        ref = gyro_reference(time,q)
        self.assertEqual(ref['quality']['angular_rate_intervals'],0)
        torch.testing.assert_close(ref['rate'][ref['mask'],2],torch.full((int(ref['mask'].sum()),),10.,dtype=torch.double),atol=1e-11,rtol=1e-11)
        q = np.tile([1.,0.,0.,0.],(len(time),1))
        q[200:] = Rotation.from_rotvec([.2,0.,0.]).as_quat()[[3,0,1,2]]
        ref = gyro_reference(time,q)
        self.assertEqual(ref['quality']['orientation_jump_intervals'],0)
        self.assertEqual(ref['quality']['angular_rate_intervals'],1)
        self.assertFalse(bool(ref['mask'][(time>=.975)&(time<=1.02)].any()))

    def test_invalid_quaternions_are_local_and_have_no_calibration_gradient(self):
        time = np.arange(401)*.005
        q = np.tile([1.,0.,0.,0.],(len(time),1))
        q[200] = np.nan
        q[250] = 0.
        ref = gyro_reference(time,q)
        self.assertEqual(ref['quality']['invalid_quaternions'],2)
        self.assertTrue(bool(torch.isfinite(ref['rate']).all()))
        self.assertFalse(bool(ref['mask'][200] | ref['mask'][250]))
        variance=torch.full((len(time),6),1e-5,dtype=torch.double,requires_grad=True)
        loss,*_=gyro_calibration_loss(variance,torch.zeros(len(time),3),torch.zeros(len(time),3),ref)
        gradient,=torch.autograd.grad(loss,variance)
        self.assertEqual(float(gradient[~ref['mask']].abs().sum()),0.)
        self.assertGreater(float(gradient[ref['mask']].abs().sum()),0.)

    def test_all_invalid_references_keep_velocity_training(self):
        ekf,flight=flight_fixture()
        flight['vicon_quaternion_gt'][:]=float('nan')
        q=torch.full((len(flight['imu_time']),6),.001,dtype=torch.double)
        result=replay_velocity(ekf,flight,q,backward=True,
            loss_options={'gyro_weight':1e-4,'calibration_window':2.})
        self.assertEqual(result['gyro_reference_quality']['valid_samples'],0)
        self.assertEqual(result['gyro_calibration_loss'],0.)
        self.assertEqual(float(result['gyro_calibration_gradient'].abs().sum()),0.)
        torch.testing.assert_close(result['gradient'],result['velocity_gradient'])

    def test_varying_body_rates_and_actual_vicon_timestamps(self):
        time = np.arange(401)*.005
        gt_time = time + .0001
        rotation = (Rotation.from_rotvec([.4, -.1, .2])
                    * Rotation.from_rotvec(gt_time[:, None]*np.array([0., 0., 1.]))
                    * Rotation.from_rotvec((.5*gt_time**2)[:, None]*np.array([1., 0., 0.])))
        q = rotation.as_quat()[:, [3, 0, 1, 2]]
        reference = gyro_reference(time, q, orientation_time=gt_time, window_seconds=.05)
        expected = torch.from_numpy(np.column_stack((time, np.sin(.5*time**2), np.cos(.5*time**2))))
        valid = reference["mask"]
        torch.testing.assert_close(reference["rate"][valid], expected[valid], atol=1e-3, rtol=1e-3)

    def test_body_frame_sign_flip_irregular_time_and_noise(self):
        time = np.r_[0., np.cumsum(np.tile([.004, .006], 100))]
        omega = np.array([.3, -.2, .1])
        initial = Rotation.from_rotvec([.4, -.3, .2])
        rotation = initial * Rotation.from_rotvec(time[:, None]*omega)
        q = rotation.as_quat()[:, [3, 0, 1, 2]]
        q[::2] *= -1  # Quaternion signs cannot change angular velocity.
        ref = gyro_reference(time, q, window_seconds=.05, orientation_std=.001)
        mask = ref["mask"]
        torch.testing.assert_close(ref["rate"][mask], torch.tensor(omega).expand(int(mask.sum()), 3), atol=1e-12, rtol=1e-12)
        self.assertFalse(bool(mask[0] | mask[-1]))
        self.assertTrue(bool((ref["variance"][mask] > 0).all()))

    def test_calibration_stable_at_floor_and_discourages_inflation(self):
        q = torch.full((20, 6), 1e-8, dtype=torch.float64, requires_grad=True)
        ref = dict(rate=torch.zeros(20, 3), variance=torch.full((20, 3), 1e-4), mask=torch.ones(20, dtype=torch.bool))
        loss, *_ = gyro_calibration_loss(q, torch.zeros(20, 3), torch.zeros(20, 3), ref)
        grad, = torch.autograd.grad(loss, q)
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertTrue(bool((grad[:, 3:] > 0).all()))
        larger, *_ = gyro_calibration_loss(q.detach()*1e5, torch.zeros(20, 3), torch.zeros(20, 3), ref)
        self.assertGreater(float(larger), float(loss))


class SmoothnessTests(unittest.TestCase):
    def test_formula_and_irregular_sampling_normalization(self):
        for time in (torch.arange(21, dtype=torch.float64)*.005,
                     torch.tensor([0., .01, .015, .032, .08, .1], dtype=torch.float64)):
            q = torch.cat((torch.ones(len(time), 3), (time[:, None]*torch.tensor([1., 2., 3.])).exp()), -1).requires_grad_()
            loss = log_variance_smoothness(q, time)
            torch.testing.assert_close(loss, loss.new_tensor(14*.005**2))
            grad, = torch.autograd.grad(loss, q)
            self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertEqual(float(log_variance_smoothness(torch.ones(3, 6), torch.arange(3))), 0.)


class AdapterTests(unittest.TestCase):
    def test_raw_padding_cannot_depend_on_ground_truth(self):
        from datasets.dataset_utils import padding_collate
        values = {"acc": torch.randn(1, 10, 3), "gyro": torch.randn(1, 10, 3)}
        # A deliberately empty initial state proves padding never reads it.
        with patch("datasets.dataset_utils.custom_collate", return_value=(values, {}, {})):
            actual, _, _ = padding_collate([], pad_len=9, raw_padding=True)
        for name in ("acc", "gyro"):
            torch.testing.assert_close(actual[name][:, :9], actual[name][:, 9:10].expand(-1, 9, -1))

    def test_raw_branch_joint_nll_trains_adapter_without_learned_corrections(self):
        from model.losses import cholesky_nll
        model = network(raw_cov_adapter=True, stage=2, sampling=0, use_initial_sensor_bias=True).double()
        configure_stage2_parameters(model, model.conf)
        samples = 30
        data = {"acc": torch.tensor([.2, -.1, 9.81], dtype=torch.float64).repeat(1, samples+9, 1),
                "gyro": torch.tensor([.02, -.01, .03], dtype=torch.float64).repeat(1, samples+9, 1),
                "dt": torch.full((1, samples, 1), .005, dtype=torch.float64),
                "initial_acc_bias": torch.tensor([[.01, .02, .03]], dtype=torch.float64),
                "initial_gyro_bias": torch.tensor([[.001, -.002, .003]], dtype=torch.float64)}
        initial = {"pos": torch.zeros(1, 1, 3, dtype=torch.float64),
                   "vel": torch.zeros(1, 1, 3, dtype=torch.float64),
                   "rot": pp.identity_SO3(1, 1, dtype=torch.float64)}
        with patch.object(model, "integrate", wraps=model.integrate) as integrate:
            with contextlib.redirect_stdout(io.StringIO()):
                output = model(dict(data), initial)
            raw_call = integrate.call_args.kwargs
        self.assertFalse(raw_call["use_gtrot"])
        torch.testing.assert_close(raw_call["data"]["corrected_acc"], data["acc"][:, 9:] - data["initial_acc_bias"][:, None])
        torch.testing.assert_close(raw_call["data"]["corrected_gyro"], data["gyro"][:, 9:] - data["initial_gyro_bias"][:, None])
        self.assertEqual(output["cov"].shape[-2:], (9, 9))
        nll = cholesky_nll(torch.full((1, 1, 9), .001, dtype=torch.float64), output["cov"])
        nll.backward()
        self.assertTrue(bool(torch.isfinite(nll)))
        self.assertGreater(float(model.raw_cov_adapter[-1].weight.grad.norm()), 0.)
        self.assertIsNone(model.acccov_decoder[-1].weight.grad)
        with torch.no_grad():
            model.accdecoder[-1].bias.add_(50.)
            model.gyrodecoder[-1].bias.add_(50.)
            other = model(dict(data), initial)
        torch.testing.assert_close(other["raw_vel"], output["raw_vel"], atol=0, rtol=0)
        torch.testing.assert_close(other["raw_rot"], output["raw_rot"], atol=0, rtol=0)

    def test_preserved_weights_zero_adapter_and_only_adapter_updates(self):
        source = network()
        adapted = network(raw_cov_adapter=True)
        load_covariance_checkpoint(adapted, source.state_dict(), initialize_adapter=True)
        x, correction = torch.randn(2, 12, 256), torch.randn(2, 12, 6)
        torch.testing.assert_close(source.cov_decoder(x, correction), adapted.cov_decoder(x, correction))
        before = {name: p.clone() for name, p in adapted.named_parameters()}
        optimizer = torch.optim.Adam(configure_stage2_parameters(adapted, adapted.conf))
        q = adapted.cov_decoder(x, correction)
        (q.log() + .02/q).mean().backward()
        optimizer.step()
        self.assertGreater(float(adapted.raw_cov_adapter[-1].weight.norm()), 0.)
        for name, parameter in adapted.named_parameters():
            if not name.startswith("raw_cov_adapter."):
                torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
                self.assertIsNone(parameter.grad)
        restored = network(raw_cov_adapter=True)
        load_covariance_checkpoint(restored, adapted.state_dict())
        wrapper = AirIMUESKF(restored, configuration="B")
        self.assertTrue(wrapper.network.gru2.weight_ih_l0.requires_grad)

    def test_checkpoint_rejects_missing_nonfinite_and_wrong_shapes(self):
        model = network(raw_cov_adapter=True)
        state = network().state_dict()
        for name, value in (("gru1.weight_ih_l0", torch.zeros(1)),
                            ("acccov_decoder.2.bias", torch.full((3,), float("nan")))):
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_covariance_checkpoint(model, {**state, name: value}, initialize_adapter=True)
        with self.assertRaises(ValueError):
            load_covariance_checkpoint(model, {k: v for k, v in state.items() if not k.startswith("gyrocov_decoder")}, initialize_adapter=True)


class GRUTrainingTests(unittest.TestCase):
    def test_matched_comparison_rejects_loss_changes_and_reports_paired_flights(self):
        with tempfile.TemporaryDirectory() as directory:
            runs=[Path(directory)/name for name in ('A','B')]
            covariance={sensor:{'mean':[.001]*3,'min':[.0001]*3,'max':[.01]*3}
                        for sensor in ('acc_cov','gyro_cov')}
            for configuration,run,losses in zip(('A','B'),runs,((4.,3.9),(3.8,3.7))):
                run.mkdir()
                groups=[{'name':'covariance_heads','lr':1e-5},{'name':'bias_gains','lr':1e-5}]
                if configuration=='B':groups.append({'name':'gru2','lr':1e-6})
                metadata={'configuration':configuration,'optimizer_groups':groups,'loss_options':{'gyro_weight':1e-4}}
                (run/'eskf_metadata.json').write_text(json.dumps(metadata))
                (run/'initial_validation.json').write_text(json.dumps({'optimizer_steps':0,'weights_unchanged':True,
                    'validation_xy_mse':4.1,'validation_xy_rmse':4.1**.5}))
                (run/'initial_validation_flights.jsonl').write_text(json.dumps({
                    'flight':'fixture','validation_xy_rmse':4.1**.5})+'\n')
                epochs=[];flights=[]
                for epoch,loss in enumerate(losses,1):
                    epochs.append({'epoch':epoch,'validation_xy_mse':loss,'validation_xy_rmse':loss**.5,
                        'train_xy_rmse':loss**.5,'epoch_seconds':1.,'validation_covariance':covariance})
                    flights.append({'epoch':epoch,'phase':'validation','flight':'fixture','validation_xy_rmse':loss**.5,
                        'gyro_calibration_loss':1.,'smoothness_loss':0.,'gyro_diagnostics':{},'correction_drift_rmse':[0.]*6})
                (run/'metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in epochs)+'\n')
                (run/'flights.jsonl').write_text('\n'.join(json.dumps(r) for r in flights)+'\n')
            validate_matched_ab(runs,2)
            report=paired_report([summarize(run) for run in runs])
            self.assertEqual(report['b_better_flights'],1)
            self.assertGreater(report['b_rmse_improvement_percent_vs_a'],0.)
            self.assertGreater(report['paired_flights'][0]['b_rmse_improvement_percent_vs_initial'],0.)
            output=Path(directory)/'comparison.json'
            with patch('sys.argv',['compare_training_runs',*(str(run) for run in runs),
                       '--require-matched-ab','--expected-epochs','2','--plots','--output',str(output)]):
                compare_main()
            self.assertTrue(output.with_suffix('.png').is_file())
            self.assertTrue(output.with_name('comparison_flights.csv').is_file())
            self.assertEqual(json.loads(output.read_text())['b_better_flights'],1)
            metadata=json.loads((runs[1]/'eskf_metadata.json').read_text())
            metadata['loss_options']['gyro_weight']=1e-3
            (runs[1]/'eskf_metadata.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'loss_options'):validate_matched_ab(runs,2)

    def test_worker_component_gradients_sum_to_joint_objective(self):
        ekf,flight=flight_fixture()
        q=torch.full((len(flight['imu_time']),6),.001,dtype=torch.double,requires_grad=True)
        result=ekf.run_sequence({'acc':flight['raw_acceleration'],'gyro':flight['raw_gyroscope']},
            flight['imu_time'],q,flight['vicon_events'],flight['cutoff_timestamp'],
            flight['initial_state'],flight['initial_covariance'],return_covariance=False)
        velocity=horizontal_velocity_loss(result['velocity'],flight['vicon_velocity_gt'],
            velocity_evaluation_mask(flight['imu_time'],flight['cutoff_timestamp']))
        ref=gyro_reference(flight['imu_time'],flight['vicon_quaternion_gt'],window_seconds=2.)
        gyro,*_=gyro_calibration_loss(q,flight['raw_gyroscope'],result['gyroscope_bias'],ref)
        smooth=log_variance_smoothness(q,flight['imu_time'])
        expected,=torch.autograd.grad(velocity+1e-4*gyro+1e-3*smooth,q)
        replay=replay_velocity(ekf,flight,q,backward=True,loss_options={
            'gyro_weight':1e-4,'smoothness_weight':1e-3,'calibration_window':2.})
        torch.testing.assert_close(replay['gradient'],expected,atol=1e-10,rtol=1e-8)

    def test_parameter_component_norms_match_direct_objectives_without_accumulation(self):
        layer=torch.nn.Linear(6,6).double()
        parameters=list(layer.parameters())
        outputs=[];variances=[];velocity_losses=[];gyro_losses=[]
        for count in (3,5):
            q=layer(torch.randn(count,6,dtype=torch.double)).exp()
            velocity=(q-.5).square().mean()
            gyro=(q[:,3:].log()+.2/q[:,3:]).mean()
            vg,=torch.autograd.grad(velocity,q,retain_graph=True)
            gg,=torch.autograd.grad(gyro,q,retain_graph=True)
            outputs.append({'velocity_gradient':vg,'gyro_calibration_gradient':gg})
            variances.append(q);velocity_losses.append(velocity);gyro_losses.append(gyro)
        velocity=sum(velocity_losses)/2;gyro=sum(gyro_losses)/2
        expected_v=torch.autograd.grad(velocity,parameters,retain_graph=True)
        expected_g=torch.autograd.grad(gyro,parameters,retain_graph=True)
        norm=lambda values:float(torch.cat([v.flatten() for v in values]).norm())
        diag=component_gradient_diagnostics(variances,outputs,parameters,
            [{'name':'layer','params':parameters}],.15)
        self.assertAlmostEqual(diag['gradient_norm_velocity'],norm(expected_v),places=12)
        self.assertAlmostEqual(diag['gradient_norm_gyro_calibration'],.15*norm(expected_g),places=12)
        self.assertTrue(all(p.grad is None for p in parameters))
        (velocity+.15*gyro).backward()
        expected_total=[v+.15*g for v,g in zip(expected_v,expected_g)]
        for p,g in zip(parameters,expected_total):torch.testing.assert_close(p.grad,g)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_component_vjps_preserve_recurrent_graph(self):
        wrapper=AirIMUESKF(network().cuda(),DifferentiableESKF(),configuration='B')
        q=wrapper.predict_variance(torch.randn(100,3),torch.randn(100,3)*.02)
        parameters=[p for p in wrapper.network.parameters() if p.requires_grad]
        velocity=(q-.001).square().mean()
        gyro=(q[:,3:].log()+.0001/q[:,3:]).mean()
        vg,=torch.autograd.grad(velocity,q,retain_graph=True)
        gg,=torch.autograd.grad(gyro,q,retain_graph=True)
        expected=torch.autograd.grad(velocity+1e-4*gyro,parameters,retain_graph=True,allow_unused=True)
        diag=component_gradient_diagnostics([q],[{'velocity_gradient':vg,'gyro_calibration_gradient':gg}],
            parameters,wrapper.optimizer_groups(),1e-4)
        q.backward(vg+1e-4*gg)
        self.assertGreater(diag['gradient_norm_velocity'],0.)
        for p,g in zip(parameters,expected):
            if g is not None:torch.testing.assert_close(p.grad,g,atol=1e-7,rtol=1e-4)
        self.assertGreater(float(wrapper.network.gru2.weight_ih_l0.grad.norm()),0.)
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_teacher_copy_and_gru_backpropagation(self):
        for config in ("B", "C"):
            with self.subTest(configuration=config):
                wrapper = AirIMUESKF(network().cuda(), DifferentiableESKF(), configuration=config)
                acc, gyro = torch.randn(100, 3), torch.randn(100, 3)*.02
                wrapper.eval()
                with torch.no_grad():
                    initial_features = wrapper.extract_features(acc, gyro)
                    torch.testing.assert_close(initial_features['correction'],initial_features['reference_correction'],
                                               atol=1e-7,rtol=1e-5)
                wrapper.train()
                features = wrapper.extract_features(acc, gyro)
                variance = wrapper.predict_variance_batch([features])[0]
                self.assertEqual(variance.device.type, "cpu")
                variance.mean().backward()
                gradient = wrapper.network.gru2.weight_ih_l0.grad
                self.assertTrue(bool(torch.isfinite(gradient).all()))
                self.assertGreater(float(gradient.norm()), 0.)
                if config == "C":
                    self.assertGreater(float(wrapper.network.gru1.weight_ih_l0.grad.norm()), 0.)
                    self.assertGreater(float(wrapper.network.cnn.net[0].weight.grad.norm()),0.)
                self.assertTrue(all(p.grad is None for p in wrapper.reference_modules.parameters()))

    def test_gru1_requires_improved_matched_validation_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = [Path(directory)/name for name in ("A", "B")]
            for config, run, mse in zip(("A", "B"), runs, (2., 1.)):
                run.mkdir()
                (run/"eskf_metadata.json").write_text(json.dumps({"configuration": config, "seed": 0}))
                (run/"metrics.jsonl").write_text(json.dumps({"validation_xy_mse": mse}) + "\n")
            self.assertEqual(validate_gru1_evidence(*runs), [2., 1.])
            (runs[1]/"metrics.jsonl").write_text(json.dumps({"validation_xy_mse": 3.}) + "\n")
            with self.assertRaises(ValueError):
                validate_gru1_evidence(*runs)

    def test_velocity_reaches_gru2_through_worker_vjp_and_frozen_weights_hold(self):
        ekf, flight = flight_fixture()
        wrapper = AirIMUESKF(network(), ekf, configuration="B")
        wrapper.train()
        before = {name: p.clone() for name, p in wrapper.network.named_parameters() if not p.requires_grad}
        parameters = [p for p in wrapper.network.parameters() if p.requires_grad]
        q = wrapper.predict_variance(flight["raw_acceleration"], flight["raw_gyroscope"])
        direct = ekf.run_sequence({"acc": flight["raw_acceleration"], "gyro": flight["raw_gyroscope"]},
            flight["imu_time"], q, flight["vicon_events"], flight["cutoff_timestamp"],
            flight["initial_state"], flight["initial_covariance"], return_covariance=False)
        loss = horizontal_velocity_loss(direct["velocity"], flight["vicon_velocity_gt"],
            velocity_evaluation_mask(flight["imu_time"], flight["cutoff_timestamp"]))
        expected = torch.autograd.grad(loss, parameters)
        q = wrapper.predict_variance(flight["raw_acceleration"], flight["raw_gyroscope"])
        # This fixture has coarse spacing; use a wider supervision window.
        output = replay_velocity(ekf, flight, q, backward=True, loss_options={"calibration_window": 2.})
        grad = output["gradient"]
        self.assertGreater(float(grad[flight["imu_time"] < 1.5].norm()), 0.)
        self.assertGreater(float(grad[(flight["imu_time"] >= 1.5) & (flight["imu_time"] < 26.)].norm()), 0.)
        actual = torch.autograd.grad(q, parameters, grad_outputs=grad)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)
        optimizer = torch.optim.Adam(wrapper.optimizer_groups())
        for p, g in zip(parameters, actual):
            p.grad = g
        self.assertGreater(float(wrapper.network.gru2.weight_ih_l0.grad.norm()), 0.)
        optimizer.step()
        for name, value in before.items():
            torch.testing.assert_close(dict(wrapper.network.named_parameters())[name], value, atol=0, rtol=0)
        self.assertFalse(wrapper.cacheable_features)

    def test_two_updates_disable_cache_and_record_diagnostics(self):
        ekf, flight = flight_fixture()
        for config in ("A", "B", "C"):
            with self.subTest(configuration=config):
                wrapper = AirIMUESKF(network(), ekf, configuration=config)
                optimizer = torch.optim.Adam(wrapper.optimizer_groups())
                records = [{"id": "fixture", "split": "train"}]
                logs = []
                with patch("ekf.runner.prepare_flight", return_value=flight):
                    with FlightTrainer(wrapper, optimizer, feature_cache_mb=10,
                            loss_options={"gyro_weight": 1e-4, "smoothness_weight": 1e-3,
                                          "calibration_window": 2.}) as trainer:
                        for epoch in (1, 2):
                            trainer.run_split(records, {"fixture": 1.5}, epoch=epoch,
                                phase="train", epoch_start=0., log=logs.append)
                        self.assertEqual(trainer.cache_hits, 1 if config == "A" else 0)
                        self.assertEqual(len(trainer.features), 1 if config == "A" else 0)
                self.assertIn("p95", logs[-1]["gyro_diagnostics"])
                self.assertIn("train_xy_rmse", logs[-1])
                for key in ('gradient_norm_velocity','gradient_norm_gyro_calibration',
                            'gradient_norm_gyro_calibration_unweighted','gradient_clip_scale'):
                    self.assertTrue(np.isfinite(logs[-1][key]))
                self.assertLessEqual(logs[-1]['gradient_norm_after_clip'],1.00001)
                self.assertEqual(trainer.last_split_diagnostics['gradients']['batches'],1)
                if config != "A":
                    self.assertGreater(max(logs[-1]["correction_drift_rmse"]), 0.)

    def test_compiled_gyro_bias_matches_eager(self):
        ekf, flight = flight_fixture()
        q = torch.full((len(flight["imu_time"]), 6), .001, dtype=torch.float64)
        with torch.no_grad():
            _, bias = velocity_sequence(ekf, flight, q, return_gyro_bias=True)
            eager = ekf.run_sequence({"acc": flight["raw_acceleration"], "gyro": flight["raw_gyroscope"]},
                flight["imu_time"], q, flight["vicon_events"], flight["cutoff_timestamp"],
                flight["initial_state"], flight["initial_covariance"], return_covariance=False)
        torch.testing.assert_close(bias, eager["gyroscope_bias"], atol=1e-11, rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
