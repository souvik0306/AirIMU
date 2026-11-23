"""Full encoder gradients/freezing, independent clipping, early stopping and eval metrics."""

import math
import unittest
from unittest.mock import patch

import torch

from ekf.gradients import (clipping_groups, clip_gradients, component_gradient_diagnostics,
                          parameter_update_diagnostics)
from ekf.runner import FlightTrainer
from ekf.stability import ValidationEarlyStopping
from ekf.training import AirIMUESKF
from ekf.tests.test_training_improvements import network, flight_fixture
from evaluation.eskf_metrics import covariance_distribution, orientation_metrics


class ExperimentCTests(unittest.TestCase):
    def test_full_encoder_updates_and_correction_heads_remain_exact(self):
        ekf,flight = flight_fixture();wrapper = AirIMUESKF(network(),ekf,configuration='C')
        groups = wrapper.optimizer_groups(cnn_lr=3e-7,gru1_lr=1e-6,gru2_lr=3e-6)
        self.assertEqual({g['name']:g['lr'] for g in groups},
            {'cnn':3e-7,'gru1':1e-6,'gru2':3e-6,'covariance_heads':1e-5,'bias_gains':1e-5})
        before = {name:value.detach().clone() for name,value in wrapper.network.state_dict().items()}
        logs = []
        with patch('ekf.runner.prepare_flight',return_value=flight):
            with FlightTrainer(wrapper,torch.optim.Adam(groups),gradient_clipping='separate',
                    max_relative_update=.01,require_encoder_gradients=True,
                    loss_options={'gyro_weight':1e-4,'calibration_window':2.}) as trainer:
                trainer.run_split([{'id':'fixture'}],{'fixture':1.5},epoch=1,phase='train',epoch_start=0.,log=logs.append)
        for name in ('cnn','gru1','gru2','covariance_heads'):
            self.assertGreater(logs[0]['gradient_norms_by_group'][name]['total'],0.)
            self.assertGreater(logs[0]['parameter_updates_by_group'][name]['l2'],0.)
        current = wrapper.network.state_dict()
        for name in ('cnn.','gru1.','gru2.','acccov_decoder.','gyrocov_decoder.'):
            self.assertTrue(any(key.startswith(name) and not torch.equal(value,before[key]) for key,value in current.items()))
        for key,value in current.items():
            if key.startswith(('accdecoder.','gyrodecoder.')):torch.testing.assert_close(value,before[key],atol=0,rtol=0)
        self.assertFalse(wrapper.cacheable_features)

    def test_each_encoder_clips_independently(self):
        parameters = [torch.nn.Parameter(torch.ones(2,dtype=torch.double)) for _ in range(5)]
        groups = [{'name':name,'params':[p]} for name,p in zip(
            ('cnn','gru1','gru2','covariance_heads','bias_gains'),parameters)]
        for p,scale in zip(parameters,(100.,.1,20.,.1,.1)):p.grad = torch.tensor([scale,0.],dtype=torch.double)
        result = clip_gradients(clipping_groups(groups,mode='separate'))
        self.assertEqual(set(result),{'cnn','gru1','gru2','heads_and_bias_gains'})
        self.assertAlmostEqual(float(parameters[0].grad.norm()),1.,places=6)
        self.assertEqual(float(parameters[1].grad[0]),.1)
        self.assertAlmostEqual(float(parameters[2].grad.norm()),5.,places=6)
        self.assertEqual(float(parameters[3].grad[0]),.1)

    def test_early_stopping_uses_eight_consecutive_epochs_and_cumulative_delta(self):
        guard = ValidationEarlyStopping(8,1e-4)
        self.assertFalse(guard.observe(4.))
        for _ in range(7):self.assertFalse(guard.observe(4.))
        self.assertTrue(guard.observe(4.))
        guard = ValidationEarlyStopping(8,1e-4);guard.observe(4.)
        guard.observe(3.99996);self.assertEqual(guard.bad_epochs,1)
        guard.observe(3.99992);self.assertEqual(guard.bad_epochs,2)
        self.assertFalse(guard.observe(3.99985));self.assertEqual(guard.bad_epochs,0)
        with self.assertRaises(ValueError):guard.observe(float('nan'))

    def test_covariance_distribution_preserves_units_and_percentiles(self):
        q = torch.tensor([[.01]*3+[.0001]*3,[.09]*3+[.0009]*3],dtype=torch.double)
        stats = covariance_distribution(q)
        self.assertAlmostEqual(stats['acc_cov']['mean'][0],.05)
        self.assertAlmostEqual(stats['gyro_cov']['median'][0],.0005)
        self.assertAlmostEqual(stats['acc_cov']['sqrt_mean_variance'][0],math.sqrt(.05))
        self.assertAlmostEqual(stats['acc_cov']['mean_predicted_std'][0],.2)
        self.assertLess(stats['gyro_cov']['p05'][0],stats['gyro_cov']['p95'][0])

    def test_orientation_sign_invariance_and_known_angle(self):
        _,flight = flight_fixture();flight['vicon_time_gt'] = flight['imu_time']
        mask = torch.ones(len(flight['imu_time']),dtype=torch.bool)
        estimate = -flight['vicon_quaternion_gt'].clone()
        result = orientation_metrics({'quaternion':estimate},flight,mask,mask)
        self.assertAlmostEqual(result['geodesic_quality_filtered']['rmse_degrees'],0.,places=10)
        estimate[:] = torch.tensor([math.cos(math.pi/12),0.,0.,math.sin(math.pi/12)])
        result = orientation_metrics({'quaternion':estimate},flight,mask,mask)
        self.assertAlmostEqual(result['geodesic_quality_filtered']['rmse_degrees'],30.,places=5)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_full_encoder_repeated_vjps_and_updates(self):
        wrapper = AirIMUESKF(network().cuda(),configuration='C');wrapper.train()
        groups = wrapper.optimizer_groups(cnn_lr=3e-7,gru1_lr=1e-6,gru2_lr=3e-6)
        optimizer = torch.optim.Adam(groups)
        q = wrapper.predict_variance(torch.randn(150,3),torch.randn(150,3)*.02)
        parameters = [p for p in wrapper.network.parameters() if p.requires_grad]
        velocity = (q-.001).square().mean();gyro = (q[:,3:].log()+.001/q[:,3:]).mean()
        vg, = torch.autograd.grad(velocity,q,retain_graph=True)
        gg, = torch.autograd.grad(gyro,q,retain_graph=True)
        diag = component_gradient_diagnostics([q],[{'velocity_gradient':vg,'gyro_calibration_gradient':gg}],
            parameters,groups,1e-4)
        q.backward(vg+1e-4*gg)
        for name in ('cnn','gru1','gru2'):
            self.assertGreater(diag['gradient_norms_by_group'][name]['velocity'],0.)
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                for p in getattr(wrapper.network,name).parameters()))
        self.assertTrue(all(p.grad is None for p in wrapper.network.accdecoder.parameters()))
        clip_gradients(clipping_groups(groups,mode='separate'))
        before = {id(p):p.detach().clone() for p in parameters}
        optimizer.step()
        updated = parameter_update_diagnostics(groups,before,.01)
        for name in ('cnn','gru1','gru2'):self.assertGreater(updated[name]['l2'],0.)


if __name__=='__main__':unittest.main()
