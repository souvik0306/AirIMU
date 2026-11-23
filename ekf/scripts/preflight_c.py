"""Local real-checkpoint checks and a validation-only sample evaluation before PBS."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import torch

from ekf import DifferentiableESKF
from ekf.data import apply_sensor_metadata, prepare_flight, read_dataset_flights
from ekf.experiment import load_airimu, save_checkpoint, file_sha256
from ekf.runner import FlightTrainer
from ekf.tests.test_training_improvements import flight_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',required=True)
    args = parser.parse_args();root = Path(args.run_root).resolve();output = root/'preflight';output.mkdir(exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(0)
    config = root/'configs/stage2.conf';dataset = root/'configs/dataset.conf';bias = root/'configs/initial_sensor_biases'
    all_flights,cutoffs = apply_sensor_metadata(read_dataset_flights(dataset),bias)
    counts = {split:sum(row['split']==split for row in all_flights) for split in ('train','validation','test')}
    assert counts=={'train':82,'validation':23,'test':12}
    train = next(row for row in all_flights if row['split']=='train')
    flight = prepare_flight(train,cutoffs[train['id']],DifferentiableESKF())
    checked = {}
    for label,name in (('Stage1','initial_airimu.ckpt'),('A','A_best_model.ckpt'),('BStrong','BStrong_best_model.ckpt')):
        wrapper = load_airimu(config,root/'inputs'/name,DifferentiableESKF(),'cpu');wrapper.eval()
        before = {key:value.clone() for key,value in wrapper.network.state_dict().items()}
        with torch.no_grad():q = wrapper.predict_variance(flight['raw_acceleration'][:256],flight['raw_gyroscope'][:256])
        assert q.shape==(256,6) and not q.requires_grad and bool(torch.isfinite(q).all()) and bool((q>0).all())
        assert all(torch.equal(value,before[key]) for key,value in wrapper.network.state_dict().items())
        checked[label] = {'strict_load':True,'prediction_shape':[256,6],'finite_positive_variances':True,
            'checkpoint_sha256':file_sha256(root/'inputs'/name)}
    ekf,fixture = flight_fixture()
    wrapper = load_airimu(config,root/'inputs/initial_airimu.ckpt',ekf,'cpu',configuration='C')
    original = {key:value.clone() for key,value in wrapper.network.state_dict().items()}
    groups = wrapper.optimizer_groups(cnn_lr=3e-7,gru1_lr=1e-6,gru2_lr=3e-6,head_lr=1e-5,bias_lr=1e-5)
    optimizer = torch.optim.Adam(groups);logs = []
    with patch('ekf.runner.prepare_flight',return_value=fixture):
        with FlightTrainer(wrapper,optimizer,gradient_clipping='separate',require_encoder_gradients=True,
                max_relative_update=.01,loss_options={'gyro_weight':1e-4,'calibration_window':2.}) as trainer:
            trainer.run_split([{'id':'synthetic_preflight_only'}],{'synthetic_preflight_only':1.5},epoch=1,
                phase='train',epoch_start=0.,log=logs.append)
    for name in ('cnn','gru1','gru2','covariance_heads'):
        assert logs[0]['gradient_norms_by_group'][name]['total']>0
        assert logs[0]['parameter_updates_by_group'][name]['l2']>0
    for name,value in wrapper.network.state_dict().items():
        if name.startswith(('accdecoder.','gyrodecoder.')):assert torch.equal(value,original[name])
    c_sample = output/'C_sample_model.ckpt'
    save_checkpoint(c_sample,wrapper,optimizer,1,logs[0]['train_xy_mse'],
                    {'configuration':'C','height_updates':True,'preflight_synthetic_update_only':True})
    restored = load_airimu(config,c_sample,DifferentiableESKF(),'cpu');restored.eval()
    with torch.no_grad():q = restored.predict_variance(flight['raw_acceleration'][:256],flight['raw_gyroscope'][:256])
    assert q.shape==(256,6) and bool(torch.isfinite(q).all())
    # A real validation flight exercises inference, full replay, artifacts and
    # orientation/calibration diagnostics. No reserved eval flight is inspected.
    command = [sys.executable,'-s','-u',str(Path(__file__).resolve().parents[2]/'jobs/utils/container_python.py'),
        '-m','evaluation.evaluate_eskf','--config',str(config),'--ckpt',str(c_sample),
        '--dataset-config',str(dataset),'--sensor-metadata-root',str(bias),
        '--output',str(output/'sample_validation_evaluation'),'--split','validation','--limit-flights','1',
        '--full-sequence','--no-plots','--device','cpu','--ekf-device','cpu','--cpu-threads','1','--height-updates']
    with (output/'sample_evaluation.log').open('w') as handle:
        subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True)
    verification = json.loads((output/'sample_validation_evaluation/evaluation_verification.json').read_text())
    assert verification['optimizer_steps']==0 and verification['weights_and_buffers_unchanged']
    assert verification['flights_completed']==1 and verification['full_sequence']
    result = {'split_counts':counts,'checkpoint_checks':checked,'C_sample_checkpoint_reload':True,
        'C_encoder_gradients_and_updates':{name:{'gradient_norm':logs[0]['gradient_norms_by_group'][name]['total'],
            **logs[0]['parameter_updates_by_group'][name]} for name in ('cnn','gru1','gru2','covariance_heads','bias_gains')},
        'correction_heads_unchanged':True,'sample_full_sequence_evaluation':verification,
        'sample_evaluation_split':'validation','reserved_eval_flights_used':0,
        'synthetic_update_not_used_for_production_initialization':True}
    (output/'preflight_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
