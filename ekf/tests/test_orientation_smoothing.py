"""Noise suppression, rapid-motion fidelity, segmentation and unchanged EKF paths."""

import unittest
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from ekf.orientation_smoothing import smooth_rotations
from ekf.regularization import gyro_reference, compare_gyro_references
from ekf.parallel import replay_velocity
from ekf.tests.test_training_improvements import flight_fixture, network


def quaternion(rotations):
    return rotations.as_quat()[:,[3,0,1,2]]


class OrientationSmoothingTests(unittest.TestCase):
    def test_filtered_training_cli_penalty_gradients_and_milestones(self):
        import train_eskf
        _,fixture=flight_fixture();model=network(raw_cov_adapter=True)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'network.conf';checkpoint=root/'initial.ckpt';out=root/'training'
            config.write_text(json.dumps({'train':{'network':'codenet','propcov':True,'gtrot':False,
                'raw_cov_adapter':True,'use_cov_limits':False}}))
            torch.save({'model_state_dict':model.state_dict()},checkpoint)
            records=[{'id':'train','split':'train'},{'id':'validation','split':'validation'}]
            arguments=['train_eskf.py','--config',str(config),'--ckpt',str(checkpoint),'--dataset-config','unused',
                '--sensor-metadata-root',str(root),'--output',str(out),'--configuration','Adapter',
                '--adapter-weight','.01','--orientation-smoothing-window','2','--calibration-window','2',
                '--epochs','2','--evaluate-initial','--allow-reduced-split','--replay-backend','eager',
                '--milestone-epochs','1','2','--milestone-report-root',str(root)]
            with patch.object(sys,'argv',arguments),patch('train_eskf.read_dataset_flights',return_value=[]),\
                    patch('train_eskf.apply_sensor_metadata',return_value=(records,{'train':1.5,'validation':1.5})),\
                    patch('ekf.runner.prepare_flight',return_value=fixture),\
                    patch('ekf.scripts.strong_regularization.milestone') as hook,contextlib.redirect_stdout(io.StringIO()):
                train_eskf.main()
            self.assertEqual([call.args[1] for call in hook.call_args_list],[1,2])
            for epoch in (1,2):
                saved=torch.load(out/'milestones'/f'epoch_{epoch:03d}.ckpt',weights_only=True)
                self.assertEqual(saved['epoch'],epoch)
                self.assertEqual(saved['ekf_metadata']['adapter_weight'],.01)
                self.assertEqual(saved['ekf_metadata']['loss_options']['orientation_smoothing_window'],2.)
            rows=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
            self.assertGreater(rows[-1]['train_gradients']['gradient_norm_adapter_regularization']['mean'],0)
            for sensor in ('acc_cov','gyro_cov'):
                self.assertEqual(len(rows[-1]['validation_covariance'][sensor]['median']),3)
                self.assertEqual(len(rows[-1]['validation_covariance'][sensor]['p95']),3)
    def test_noise_reduction_and_nonzero_reference_uncertainty(self):
        t=np.arange(801)*.005;axis=np.array([.3,-.2,.8]);axis/=np.linalg.norm(axis)
        angle=8*t+2/(2*np.pi)*(1-np.cos(2*np.pi*t))
        truth=Rotation.from_rotvec(angle[:,None]*axis)
        rng=np.random.default_rng(7)
        observed=truth*Rotation.from_rotvec(rng.normal(0,.001,(len(t),3)))
        raw=gyro_reference(t,quaternion(observed));filtered=gyro_reference(t,quaternion(observed),smoothing_window_seconds=.05)
        expected=gyro_reference(t,quaternion(truth));mask=filtered['mask']&(torch.tensor(t)>.1)&(torch.tensor(t)<3.9)
        self.assertTrue(torch.equal(raw['mask'],filtered['mask']))
        original_rms=float((raw['rate'][mask]-expected['rate'][mask]).square().mean().sqrt())
        filtered_rms=float((filtered['rate'][mask]-expected['rate'][mask]).square().mean().sqrt())
        self.assertLess(filtered_rms,.8*original_rms)
        self.assertTrue(bool((filtered['variance']>=raw['variance']-1e-15).all()))
        self.assertTrue(bool((filtered['variance']>0).all()))

    def test_fast_rotation_sign_invariance_and_varying_rate_fidelity(self):
        t=np.arange(801)*.005;angle=8*t+2/(2*np.pi)*(1-np.cos(2*np.pi*t))
        q=quaternion(Rotation.from_rotvec(angle[:,None]*np.array([0.,0.,1.])))
        q[::2]*=-1
        raw=gyro_reference(t,q);filtered=gyro_reference(t,q,smoothing_window_seconds=.05)
        mask=raw['mask']&(torch.tensor(t)>.1)&(torch.tensor(t)<3.9)
        torch.testing.assert_close(filtered['rate'][mask],raw['rate'][mask],atol=5e-4,rtol=1e-4)
        self.assertLess(float(filtered['rate'][mask].norm(dim=1).max()),10.01)
        uniform=quaternion(Rotation.from_rotvec(t[:,None]*np.array([0.,0.,12.])))
        a=gyro_reference(t,uniform,smoothing_window_seconds=.05)
        torch.testing.assert_close(a['rate'][a['mask'],2],torch.full((int(a['mask'].sum()),),12.,dtype=torch.double),atol=1e-10,rtol=1e-10)

    def test_no_smoothing_across_breaks_invalids_or_gaps(self):
        t=np.arange(121)*.005;rot=Rotation.from_rotvec(np.c_[np.zeros(len(t)),np.zeros(len(t)),.5*t])
        good=np.ones(len(t),dtype=bool);good[60]=False
        breaks=np.zeros(len(t)-1,dtype=bool);breaks[29]=True;breaks[89]=True
        full,_,_=smooth_rotations(t,rot,good,breaks,.05)
        changed=rot.as_quat();changed[30:60]=Rotation.from_rotvec([1.,.4,.2]).as_quat()
        second,_,_=smooth_rotations(t,Rotation.from_quat(changed),good,breaks,.05)
        np.testing.assert_allclose(full.as_matrix()[:30],second.as_matrix()[:30],atol=1e-14)
        np.testing.assert_allclose(full.as_matrix()[61:],second.as_matrix()[61:],atol=1e-14)
        q=quaternion(rot);q[60]=np.nan;q[90:]=quaternion(Rotation.from_rotvec([[2.,0.,0.]]))[0]
        t[30:]+=.1
        a=gyro_reference(t,q);b=gyro_reference(t,q,smoothing_window_seconds=.05)
        self.assertTrue(torch.equal(a['mask'],b['mask']))
        self.assertEqual(a['quality'],b['quality'])
        self.assertTrue(bool(torch.isfinite(b['rate']).all()))

    def test_filter_only_changes_supervision_not_velocity_or_inputs(self):
        ekf,flight=flight_fixture();q=torch.full((len(flight['imu_time']),6),.001,dtype=torch.double)
        before=flight['vicon_quaternion_gt'].clone()
        a=replay_velocity(ekf,flight,q,backward=True,loss_options={'gyro_weight':1e-4,'calibration_window':2.})
        b=replay_velocity(ekf,flight,q,backward=True,loss_options={'gyro_weight':1e-4,'calibration_window':2.,'orientation_smoothing_window':2.})
        self.assertTrue(torch.equal(before,flight['vicon_quaternion_gt']))
        self.assertEqual(a['loss'],b['loss'])
        torch.testing.assert_close(a['velocity_gradient'],b['velocity_gradient'],atol=0,rtol=0)
        self.assertIsNotNone(b['gyro_reference_comparison'])
        self.assertGreater(float(b['gyro_calibration_gradient'].abs().sum()),0.)

    def test_reference_report_and_disabled_filter_legacy_parity(self):
        t=np.arange(101)*.005;q=quaternion(Rotation.from_rotvec(t[:,None]*np.array([0.,0.,1.])))
        a=gyro_reference(t,q);b=gyro_reference(t,q,smoothing_window_seconds=0.)
        for key in ('rate','variance','mask'):self.assertTrue(torch.equal(a[key],b[key]))
        filtered=gyro_reference(t,q,smoothing_window_seconds=.05)
        report=compare_gyro_references(torch.ones(len(t),6)*.001,torch.ones(len(t),3),torch.zeros(len(t),3),filtered)
        self.assertEqual(report['valid_samples']+report['rejected_samples'],len(t))
        self.assertEqual(len(report['filtered']['residual_rms_xyz']),3)


if __name__=='__main__':unittest.main()
