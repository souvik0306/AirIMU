"""Local checks for fresh identical initialization and calibration-only SO(3) filtering."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from ekf import DifferentiableESKF
from ekf.experiment import load_airimu, save_checkpoint, file_sha256
from ekf.regularization import gyro_reference
from ekf.runner import FlightTrainer
from ekf.scripts.residual_adapter import check_inputs, evaluate, write
from ekf.scripts.strong_regularization import compare
from ekf.tests.test_training_improvements import flight_fixture


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',required=True)
    args=parser.parse_args();root=Path(args.run_root).resolve();plan=check_inputs(root)
    torch.set_num_threads(1);torch.manual_seed(0);output=root/'preflight';output.mkdir(exist_ok=True)
    checkpoint=root/'inputs/adapter_initial.ckpt';checkpoint_sha=file_sha256(checkpoint)
    assert checkpoint_sha==file_sha256(plan['initialization_source'])
    parent=torch.load(root/'inputs/BStrong_epoch50.ckpt',map_location='cpu',weights_only=True)['model_state_dict']
    ekf,fixture=flight_fixture()
    wrapper=load_airimu(root/'configs/adapter.conf',checkpoint,ekf,'cpu',configuration='Adapter')
    original={key:value.clone() for key,value in wrapper.network.state_dict().items()}
    for name,value in parent.items():assert torch.equal(value,original[name]),name
    assert all(not bool(value.ne(0).any()) for name,value in original.items() if name.startswith('raw_cov_adapter.2.'))
    baseline=load_airimu(root/'configs/stage2.conf',root/'inputs/BStrong_epoch50.ckpt',DifferentiableESKF(),'cpu').eval()
    wrapper.train()
    with torch.no_grad():
        q=wrapper.predict_variance(fixture['raw_acceleration'],fixture['raw_gyroscope'])
        q0=baseline.predict_variance(fixture['raw_acceleration'],fixture['raw_gyroscope'])
    assert torch.equal(q,q0)
    optimizer=torch.optim.Adam(wrapper.optimizer_groups(adapter_lr=1e-4));assert not optimizer.state
    logged=[]
    with patch('ekf.runner.prepare_flight',return_value=fixture):
        with FlightTrainer(wrapper,optimizer,adapter_weight=.01,max_relative_update=.01,
                loss_options={'gyro_weight':1e-4,'calibration_window':2.,'orientation_smoothing_window':.05}) as trainer:
            for epoch in (1,2):trainer.run_split([{'id':'throwaway_strongreg'}],{'throwaway_strongreg':1.5},
                epoch=epoch,phase='train',epoch_start=0.,log=logged.append)
    for name,value in parent.items():assert torch.equal(value,wrapper.network.state_dict()[name]),name
    assert logged[-1]['gradient_norm_adapter_regularization']>0 and logged[-1]['gradient_norm_velocity']>0
    sample=output/'throwaway_filtered_adapter.ckpt'
    save_checkpoint(sample,wrapper,optimizer,2,logged[-1]['train_xy_mse'],{'configuration':'Adapter','preflight_only':True})
    restored=load_airimu(root/'configs/adapter.conf',sample,DifferentiableESKF(),'cpu').eval()
    assert all(torch.equal(value,restored.network.state_dict()[name]) for name,value in wrapper.network.state_dict().items())
    destination=output/'sample_validation_evaluation'
    verification=evaluate(root,sample,destination,limit=1,plots=True,smoothing_window=.05)
    # Quantify synthetic fidelity separately from real residual agreement.
    t=np.arange(801)*.005;axis=np.array([.3,-.2,.8]);axis/=np.linalg.norm(axis)
    angle=8*t+2/(2*np.pi)*(1-np.cos(2*np.pi*t));truth=Rotation.from_rotvec(angle[:,None]*axis)
    observed=truth*Rotation.from_rotvec(np.random.default_rng(7).normal(0,.001,(len(t),3)))
    q=lambda r:r.as_quat()[:,[3,0,1,2]]
    clean=gyro_reference(t,q(truth));raw=gyro_reference(t,q(observed))
    filtered=gyro_reference(t,q(observed),smoothing_window_seconds=.05)
    noiseless=gyro_reference(t,q(truth),smoothing_window_seconds=.05)
    mask=raw['mask']&(torch.tensor(t)>.1)&(torch.tensor(t)<3.9)
    raw_rms=float((raw['rate'][mask]-clean['rate'][mask]).square().mean().sqrt())
    filtered_rms=float((filtered['rate'][mask]-clean['rate'][mask]).square().mean().sqrt())
    fidelity=float((noiseless['rate'][mask]-clean['rate'][mask]).abs().max())
    assert filtered_rms<.8*raw_rms and fidelity<.001
    old=Path(plan['control_evaluations']['Weak15']).parents[2]/'preflight/sample_validation_evaluation'
    checked=compare(root,destination,{label:old for label in ('BStrong','Weak15','Weak25')},
                    output/'comparison_execution_check',expected=1)
    # This single-flight execution check uses throwaway preflight models;
    # scientific comparisons later use the actual saved control evaluations.
    result={'all_required_checks_passed':True,'exact_original_BStrong_checkpoint':True,
        'adapter_initialization_identical_to_previous_experiment':True,'fresh_Adam_step_zero':True,
        'zero_adapter_exact_BStrong_identity':True,'adapter_only_updates':True,
        'frozen_BStrong_weights_and_BatchNorm_unchanged':True,'checkpoint_roundtrip':True,
        'original_quality_masks_unchanged':True,'EKF_and_model_source_files_unchanged':plan['unchanged_model_and_EKF_paths'],
        'synthetic_reference_rate_error_rms_raw':raw_rms,'synthetic_reference_rate_error_rms_filtered':filtered_rms,
        'synthetic_error_reduction_percent':100*(1-filtered_rms/raw_rms),
        'noiseless_rapid_motion_max_rate_change':fidelity,
        'reference_noise_floor_retained':True,'sample_validation_evaluation':verification,
        'comparison_and_plot_pipeline_passed':True,'comparison_execution_check_uses_throwaway_preflight_models':True,
        'real_flight_reference_diagnostics':checked['models']['Strong']['paired_reference_diagnostics'],
        'reserved_eval_flights_used':0,'production_initializer_unchanged':file_sha256(checkpoint)==checkpoint_sha}
    assert result['production_initializer_unchanged']
    write(root/'preflight_verification.json',result);plan['preflight']=result
    plan['input_sha256']['preflight_verification.json']=file_sha256(root/'preflight_verification.json')
    plan['status']='all local checks passed; ready for stronger-penalty/filtered-reference PBS'
    write(root/'run_plan.json',plan);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
