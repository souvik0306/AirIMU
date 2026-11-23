"""Independent clipping, actual Adam updates, and instability stops for B Strong."""

import math
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from ekf.gradients import (clipping_groups, clip_gradients, gradient_norm,
                           parameter_update_diagnostics, summarize_batches)
from ekf.runner import FlightTrainer
from ekf.stability import LossGrowthGuard
from ekf.training import AirIMUESKF
from ekf.tests.test_training_improvements import network, flight_fixture
from ekf.scripts.finish_strong_b_run import make_report, verify_comparable


class SeparateClippingTests(unittest.TestCase):
    def groups(self):
        params = [torch.nn.Parameter(torch.tensor([3., 4.], dtype=torch.double)) for _ in range(3)]
        return [{'name':name,'params':[p],'lr':lr} for name,p,lr in zip(
            ('covariance_heads','bias_gains','gru2'),params,(1e-5,1e-5,3e-6))]

    def test_large_gru_gradient_does_not_shrink_unclipped_heads(self):
        groups = self.groups()
        for group, values in zip(groups, ([.3,.4],[0.,.2],[60.,80.])):
            group['params'][0].grad = torch.tensor(values,dtype=torch.double)
        originals = [group['params'][0].grad.clone() for group in groups]
        diagnostics = clip_gradients(clipping_groups(groups,mode='separate'))
        self.assertEqual(diagnostics['heads_and_bias_gains']['scale'],1.)
        for group, original in zip(groups[:2], originals[:2]):
            torch.testing.assert_close(group['params'][0].grad,original,atol=0,rtol=0)
        self.assertAlmostEqual(gradient_norm([groups[2]['params'][0].grad]),5.,places=6)
        self.assertGreater(gradient_norm(group['params'][0].grad for group in groups),5.)

    def test_heads_and_gains_share_one_limit_and_no_global_reclip(self):
        groups = self.groups()
        for group, values in zip(groups, ([3.,0.],[0.,4.],[6.,8.])):
            group['params'][0].grad = torch.tensor(values,dtype=torch.double)
        diagnostics = clip_gradients(clipping_groups(groups,mode='separate'))
        self.assertAlmostEqual(diagnostics['heads_and_bias_gains']['norm_after'],1.,places=6)
        self.assertAlmostEqual(diagnostics['gru2']['norm_after'],5.,places=6)
        torch.testing.assert_close(groups[0]['params'][0].grad,torch.tensor([.6,0.],dtype=torch.double))
        self.assertAlmostEqual(gradient_norm(group['params'][0].grad for group in groups),math.sqrt(26),places=5)

    def test_global_default_is_unchanged(self):
        groups = self.groups()
        for group in groups:
            group['params'][0].grad = torch.ones(2,dtype=torch.double)*10
        result = clip_gradients(clipping_groups(groups))
        self.assertEqual(set(result),{'global'})
        self.assertAlmostEqual(gradient_norm(group['params'][0].grad for group in groups),1.,places=6)

    def test_unknown_and_duplicate_parameters_are_rejected(self):
        groups = self.groups()
        groups.append({'name':'unrecognized','params':[torch.nn.Parameter(torch.ones(2))]})
        with self.assertRaises(ValueError):clipping_groups(groups,mode='separate')
        with self.assertRaises(ValueError):clipping_groups(groups[:3]+[groups[0]],mode='separate')

    def test_update_norms_measure_actual_adam_change_and_guard(self):
        groups = self.groups()
        optimizer = torch.optim.Adam(groups)
        for group in groups:
            group['params'][0].grad = torch.tensor([10.,-4.],dtype=torch.double)
        before = {id(p):p.detach().clone() for group in groups for p in group['params']}
        optimizer.step()
        result = parameter_update_diagnostics(optimizer.param_groups,before,.01)
        for group in groups:
            delta = group['params'][0].detach()-before[id(group['params'][0])]
            self.assertAlmostEqual(result[group['name']]['l2'],float(delta.norm()),places=14)
            self.assertAlmostEqual(result[group['name']]['relative_l2'],float(delta.norm())/5.,places=14)
        with self.assertRaisesRegex(RuntimeError,'relative Adam update'):
            parameter_update_diagnostics(optimizer.param_groups,before,1e-12)
        with torch.no_grad():groups[2]['params'][0][0] = float('nan')
        with self.assertRaisesRegex(RuntimeError,'non-finite Adam update'):
            parameter_update_diagnostics(optimizer.param_groups,before)

    def test_loss_guard_requires_sustained_growth_and_resets(self):
        guard = LossGrowthGuard(2.,2)
        for value in (4.,3.,7.,4.,7.):guard.observe('validation',value)
        with self.assertRaisesRegex(RuntimeError,'consecutive epochs'):guard.observe('validation',8.)
        with self.assertRaisesRegex(RuntimeError,'non-finite'):LossGrowthGuard().observe('train',float('nan'))

    def test_zero_initialized_gain_update_uses_unit_normalization(self):
        p = torch.nn.Parameter(torch.zeros(3,dtype=torch.double))
        optimizer = torch.optim.Adam([{'name':'bias_gains','params':[p],'lr':1e-5}])
        before = {id(p):p.detach().clone()};p.grad = torch.ones_like(p)
        optimizer.step()
        result = parameter_update_diagnostics(optimizer.param_groups,before,.01)['bias_gains']
        self.assertEqual(result['normalization_norm'],1.)
        self.assertLess(result['relative_l2'],.01)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_separate_clipping_and_adam_updates(self):
        groups = self.groups()
        for group in groups:
            group['params'] = [torch.nn.Parameter(group['params'][0].detach().cuda())]
            group['params'][0].grad = torch.tensor([60.,80.],device='cuda',dtype=torch.double)
        optimizer = torch.optim.Adam(groups)
        clipped = clip_gradients(clipping_groups(optimizer.param_groups,mode='separate'))
        before = {id(p):p.detach().clone() for group in optimizer.param_groups for p in group['params']}
        optimizer.step()
        result = parameter_update_diagnostics(optimizer.param_groups,before,.01)
        self.assertLessEqual(clipped['gru2']['norm_after'],5.00001)
        self.assertLessEqual(clipped['heads_and_bias_gains']['norm_after'],1.00001)
        self.assertGreater(result['gru2']['l2'],0.)

    def test_trainer_records_independent_clips_and_actual_updates(self):
        ekf, flight = flight_fixture()
        wrapper = AirIMUESKF(network(),ekf,configuration='B')
        optimizer = torch.optim.Adam(wrapper.optimizer_groups(gru2_lr=3e-6))
        frozen = {name:p.detach().clone() for name,p in wrapper.network.named_parameters() if not p.requires_grad}
        logs = [];batches = []
        with patch('ekf.runner.prepare_flight',return_value=flight):
            with FlightTrainer(wrapper,optimizer,gradient_clipping='separate',max_relative_update=.01,
                               loss_options={'gyro_weight':1e-4,'calibration_window':2.}) as trainer:
                trainer.run_split([{'id':'fixture','split':'train'}],{'fixture':1.5},epoch=1,
                                  phase='train',epoch_start=0.,log=logs.append,log_batch=batches.append)
                summary = trainer.last_split_diagnostics['gradients']
        self.assertIsNone(batches[0]['gradient_clip_scale'])
        for name,limit in (('gru2',5.),('heads_and_bias_gains',1.)):
            self.assertLessEqual(batches[0]['gradient_clipping_groups'][name]['norm_after'],limit+1e-6)
            self.assertIn('clip_fraction',summary['clipping_groups'][name])
        self.assertGreater(batches[0]['parameter_updates_by_group']['gru2']['l2'],0.)
        self.assertEqual(summary['parameter_updates_by_group']['gru2']['l2']['mean'],
                         batches[0]['parameter_updates_by_group']['gru2']['l2'])
        for name,p in wrapper.network.named_parameters():
            if name in frozen:torch.testing.assert_close(p,frozen[name],atol=0,rtol=0)


class StrongReportTests(unittest.TestCase):
    def test_report_separates_epoch25_from_best_of_50_and_rejects_changed_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [root/'controls/A',root/'controls/B',root/'training']
            covariance = {sensor:{'mean':[.001]*3,'min':[.0001]*3,'max':[.01]*3}
                          for sensor in ('acc_cov','gyro_cov')}
            for run,count,step,configuration in zip(runs,(25,25,50),(.02,.01,.02),('A','B','B')):
                run.mkdir(parents=True)
                metadata = {'configuration':configuration,'loss_options':{'gyro_weight':1e-4},
                    'flights':[{'id':'fixture','sensor_metadata':{'source_csv':str(run/'bias.csv')}}]}
                (run/'eskf_metadata.json').write_text(json.dumps(metadata))
                (run/'initial_validation.json').write_text(json.dumps({'validation_xy_rmse':math.sqrt(4.1)}))
                (run/'initial_validation_flights.jsonl').write_text(json.dumps({
                    'flight':'fixture','validation_xy_rmse':math.sqrt(4.1)})+'\n')
                epochs = [];flights = []
                for epoch in range(1,count+1):
                    mse = 4.-step*epoch
                    epochs.append({'epoch':epoch,'validation_xy_mse':mse,'validation_xy_rmse':math.sqrt(mse),
                        'train_xy_rmse':math.sqrt(mse),'epoch_seconds':1.,'validation_covariance':covariance,
                        'train_gradients':{'clipping_groups':{name:{'clip_fraction':.5}
                            for name in ('gru2','heads_and_bias_gains')},
                            'parameter_updates_by_group':{name:{'relative_l2':{'median':1e-6}}
                                for name in ('gru2','covariance_heads','bias_gains')}}})
                    flights.append({'epoch':epoch,'phase':'validation','flight':'fixture',
                        'validation_xy_rmse':math.sqrt(mse),'gyro_calibration_loss':1.,'smoothness_loss':0.,
                        'gyro_diagnostics':{},'correction_drift_rmse':[0.]*6})
                for name,rows in (('metrics',epochs),('flights',flights)):
                    (run/f'{name}.jsonl').write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
            report = make_report(root)
            self.assertEqual(report['strong_epoch25']['epoch'],25)
            self.assertEqual(report['strong_best_epochs_1_to_25']['epoch'],25)
            self.assertEqual(report['strong_best_epochs_26_to_50']['epoch'],50)
            self.assertEqual(report['first_strong_epoch_beating_a'],26)
            self.assertTrue((root/'comparison/comparison.png').is_file())
            self.assertTrue((root/'comparison/comparison_flights.csv').is_file())
            metadata = json.loads((runs[2]/'eskf_metadata.json').read_text())
            metadata['loss_options']['gyro_weight'] = 1e-3
            (runs[2]/'eskf_metadata.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'loss_options'):verify_comparable(runs)


if __name__ == '__main__':unittest.main()
