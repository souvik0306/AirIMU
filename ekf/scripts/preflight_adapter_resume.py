"""Verify exact epoch-15 Adam restoration and one throwaway EKF step before PBS."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from ekf import DifferentiableESKF
from ekf.experiment import load_airimu, save_checkpoint, file_sha256
from ekf.runner import FlightTrainer
from ekf.scripts.residual_adapter import check_inputs, evaluate, write
from ekf.tests.test_training_improvements import flight_fixture
from train_eskf import restore_adapter_optimizer


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',required=True)
    args=parser.parse_args();root=Path(args.run_root).resolve();plan=check_inputs(root)
    torch.set_num_threads(1);torch.manual_seed(0)
    checkpoint=root/'inputs/adapter_initial.ckpt';original_sha=file_sha256(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['epoch']==plan['resume_epoch']==15
    parent=torch.load(root/'inputs/BStrong_epoch50.ckpt',map_location='cpu',weights_only=True)['model_state_dict']
    ekf,fixture=flight_fixture()
    wrapper=load_airimu(root/'configs/adapter.conf',checkpoint,ekf,'cpu',configuration='Adapter').train()
    trainable=[name for name,p in wrapper.network.named_parameters() if p.requires_grad]
    assert trainable and all(name.startswith('raw_cov_adapter.') for name in trainable)
    assert not wrapper.network.cnn.training
    for name,value in parent.items():assert torch.equal(value,wrapper.network.state_dict()[name]),name
    optimizer=torch.optim.Adam(wrapper.optimizer_groups(adapter_lr=1e-4))
    steps=restore_adapter_optimizer(optimizer,saved);assert steps==315
    output=root/'preflight';output.mkdir(exist_ok=True)
    logs=[]
    with patch('ekf.runner.prepare_flight',return_value=fixture):
        with FlightTrainer(wrapper,optimizer,adapter_weight=1e-3,max_relative_update=.01,
                loss_options={'gyro_weight':1e-4,'calibration_window':2.}) as trainer:
            trainer.run_split([{'id':'throwaway_resume_check'}],{'throwaway_resume_check':1.5},
                epoch=16,phase='train',epoch_start=0.,log=logs.append)
    assert {int(state['step']) for state in optimizer.state_dict()['state'].values()}=={316}
    assert logs[0]['parameter_updates_by_group']['raw_cov_adapter']['l2']>0
    for name,value in parent.items():assert torch.equal(value,wrapper.network.state_dict()[name]),name
    sample=output/'throwaway_resumed_adapter.ckpt'
    save_checkpoint(sample,wrapper,optimizer,16,logs[0]['train_xy_mse'],{'configuration':'Adapter','preflight_only':True})
    restored=load_airimu(root/'configs/adapter.conf',sample,DifferentiableESKF(),'cpu').eval()
    assert all(torch.equal(value,restored.network.state_dict()[name]) for name,value in wrapper.network.state_dict().items())
    verification=evaluate(root,sample,output/'sample_validation_evaluation',limit=1,plots=True)
    assert file_sha256(checkpoint)==original_sha
    result={'all_required_checks_passed':True,'resume_epoch':15,'adam_moments_and_steps_restored_exactly':True,
        'restored_adam_steps':315,'throwaway_adam_step':316,'learning_rate':1e-4,
        'trainable_parameters':sum(p.numel() for p in wrapper.network.parameters() if p.requires_grad),
        'all_BStrong_weights_and_BatchNorm_buffers_unchanged':True,'strict_checkpoint_roundtrip':True,
        'production_initial_checkpoint_unchanged':True,'throwaway_updates_excluded_from_training':True,
        'sample_validation_evaluation':verification,'reserved_eval_flights_used':0}
    write(root/'preflight_verification.json',result)
    plan['preflight']=result;plan['status']='local Adam restoration and sample replay passed; ready for PBS'
    plan['input_sha256']['preflight_verification.json']=file_sha256(root/'preflight_verification.json')
    write(root/'run_plan.json',plan);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
