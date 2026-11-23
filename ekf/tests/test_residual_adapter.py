"""Frozen-parent identity, EKF gradients and adapter-only checkpoint regression tests."""

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from ekf.experiment import load_airimu, save_checkpoint
from ekf.runner import FlightTrainer
from ekf.training import AirIMUESKF
from ekf.tests.test_training_improvements import network, flight_fixture
from model.checkpoints import load_covariance_checkpoint


class ResidualAdapterTests(unittest.TestCase):
    def test_training_cli_selects_best_including_parent_and_preserves_frozen_state(self):
        import train_eskf
        _,fixture=flight_fixture();parent=network()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'network.conf';checkpoint=root/'parent.ckpt';out=root/'training'
            config.write_text(json.dumps({'train':{'network':'codenet','propcov':True,'gtrot':False,
                'use_cov_limits':False,'raw_cov_adapter':True}}))
            torch.save({'model_state_dict':parent.state_dict()},checkpoint)
            records=[{'id':'train_1','split':'train'},{'id':'train_2','split':'train'},
                     {'id':'validation_1','split':'validation'}]
            cutoffs={row['id']:1.5 for row in records}
            argv=['train_eskf.py','--config',str(config),'--ckpt',str(checkpoint),
                '--dataset-config',str(root/'unused.conf'),'--sensor-metadata-root',str(root),
                '--output',str(out),'--configuration','Adapter','--initialize-adapter',
                '--adapter-weight','1e-3','--evaluate-initial','--epochs','2','--flight-batch-size','2',
                '--allow-reduced-split','--replay-backend','eager','--calibration-window','2.']
            with patch.object(sys,'argv',argv),patch('train_eskf.read_dataset_flights',return_value=[]),\
                    patch('train_eskf.apply_sensor_metadata',return_value=(records,cutoffs)),\
                    patch('ekf.runner.prepare_flight',return_value=fixture),contextlib.redirect_stdout(io.StringIO()):
                train_eskf.main()
            initial=json.loads((out/'initial_validation.json').read_text())
            metrics=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
            best=torch.load(out/'best_model.ckpt',weights_only=True)
            self.assertEqual(best['best_loss'],min([initial['validation_xy_mse']]+[row['validation_xy_mse'] for row in metrics]))
            self.assertTrue(all(row['frozen_parent_weights_and_buffers_unchanged'] for row in metrics))
            newest=torch.load(out/'newest.ckpt',weights_only=True)
            for name,value in parent.state_dict().items():self.assertTrue(torch.equal(value,newest['model_state_dict'][name]))
            # Compare uninterrupted epoch 3 with an Adam-restored continuation.
            out_resume=root/'resumed';out_full=root/'full'
            resume_args=list(argv);resume_args[resume_args.index('--output')+1]=str(out_resume)
            resume_args[resume_args.index('--ckpt')+1]=str(out/'newest.ckpt')
            resume_args[resume_args.index('--epochs')+1]='1';resume_args.remove('--initialize-adapter')
            resume_args+=['--resume','--early-stopping-patience','5']
            full_args=list(argv);full_args[full_args.index('--output')+1]=str(out_full)
            full_args[full_args.index('--epochs')+1]='3'
            for arguments in (resume_args,full_args):
                with patch.object(sys,'argv',arguments),patch('train_eskf.read_dataset_flights',return_value=[]),\
                        patch('train_eskf.apply_sensor_metadata',side_effect=lambda *a:(list(records),dict(cutoffs))),\
                        patch('ekf.runner.prepare_flight',return_value=fixture),contextlib.redirect_stdout(io.StringIO()):
                    train_eskf.main()
            resumed=torch.load(out_resume/'newest.ckpt',weights_only=True)
            full=torch.load(out_full/'newest.ckpt',weights_only=True)
            self.assertEqual(resumed['epoch'],3)
            for name,value in full['model_state_dict'].items():
                self.assertTrue(torch.equal(value,resumed['model_state_dict'][name]),name)
            for key,state in full['optimizer_state_dict']['state'].items():
                for name,value in state.items():self.assertTrue(torch.equal(value,resumed['optimizer_state_dict']['state'][key][name]))
            summary=json.loads((out_resume/'training_summary.json').read_text())
            self.assertEqual(summary['restored_adam_steps'],2)
            self.assertEqual(summary['total_optimizer_steps'],3)

    def test_identity_migration_freezing_and_two_ekf_updates(self):
        torch.manual_seed(0)
        parent = network().eval()
        adapted = network(raw_cov_adapter=True)
        load_covariance_checkpoint(adapted,parent.state_dict(),initialize_adapter=True)
        ekf,flight = flight_fixture()
        baseline = AirIMUESKF(parent,ekf,configuration='A').eval()
        wrapper = AirIMUESKF(adapted,ekf,configuration='Adapter').train()
        self.assertTrue(wrapper.cacheable_features)
        self.assertFalse(wrapper.network.cnn.training)
        self.assertTrue(all(name.startswith('raw_cov_adapter.') for name,p in adapted.named_parameters() if p.requires_grad))
        original = {key:value.clone() for key,value in adapted.state_dict().items()}
        with torch.no_grad():
            q0 = baseline.predict_variance(flight['raw_acceleration'],flight['raw_gyroscope'])
            q1 = wrapper.predict_variance(flight['raw_acceleration'],flight['raw_gyroscope'])
        self.assertTrue(torch.equal(q0,q1))
        groups = wrapper.optimizer_groups(adapter_lr=1e-4)
        self.assertEqual([(g['name'],g['lr']) for g in groups],[('raw_cov_adapter',1e-4)])
        optimizer = torch.optim.Adam(groups)
        logs = []
        with patch('ekf.runner.prepare_flight',return_value=flight):
            with FlightTrainer(wrapper,optimizer,adapter_weight=1e-3,
                    loss_options={'gyro_weight':1e-4,'calibration_window':2.},max_relative_update=.01) as trainer:
                for epoch in (1,2):
                    trainer.run_split([{'id':'fixture'}],{'fixture':1.5},epoch=epoch,phase='train',epoch_start=0.,log=logs.append)
        self.assertGreater(logs[-1]['gradient_norm_velocity'],0.)
        self.assertGreater(logs[-1]['gradient_norm_adapter_regularization'],0.)
        self.assertGreater(logs[-1]['parameter_updates_by_group']['raw_cov_adapter']['l2'],0.)
        for name,value in adapted.state_dict().items():
            if not name.startswith('raw_cov_adapter.'):
                self.assertTrue(torch.equal(value,original[name]),name)
        for index in (0,2):
            self.assertFalse(torch.equal(adapted.raw_cov_adapter[index].weight,original[f'raw_cov_adapter.{index}.weight']))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.ckpt'
            save_checkpoint(path,wrapper,optimizer,2,1.,{'configuration':'Adapter'})
            restored = network(raw_cov_adapter=True)
            load_covariance_checkpoint(restored,torch.load(path,weights_only=True)['model_state_dict'])
            for name,value in restored.state_dict().items():self.assertTrue(torch.equal(value,adapted.state_dict()[name]))

    def test_migration_is_explicit_and_rejects_missing_parent_tensor(self):
        parent = network();adapted = network(raw_cov_adapter=True)
        with self.assertRaises(ValueError):load_covariance_checkpoint(adapted,parent.state_dict())
        state = dict(parent.state_dict());state.pop('gru2.weight_ih_l0')
        with self.assertRaises(ValueError):load_covariance_checkpoint(adapted,state,initialize_adapter=True)
        with self.assertRaises(ValueError):AirIMUESKF(parent,configuration='Adapter')


if __name__=='__main__':unittest.main()
