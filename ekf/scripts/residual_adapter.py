"""Preflight, audit and fresh inference for the frozen B Strong residual adapter."""

import argparse
from datetime import datetime
import html
import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import torch

from ekf import DifferentiableESKF
from ekf.data import apply_sensor_metadata, prepare_flight, read_dataset_flights
from ekf.experiment import load_airimu, save_checkpoint, file_sha256
from ekf.runner import FlightTrainer
from ekf.tests.test_training_improvements import flight_fixture
from evaluation.batchnorm_diagnostic import motion_summary
from evaluation.compare_eskf_models import summarize_evaluation, canonical_flights, improvement, write_csv
from evaluation.evaluate_eskf import write_json_atomic
from evaluation.plot_eskf import verify_plot_outputs


def write(path, value):
    write_json_atomic(path,value)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_inputs(root):
    plan=json.loads((root/'run_plan.json').read_text())
    for name,expected in plan['input_sha256'].items():
        assert file_sha256(root/name)==expected,name
    return plan


def evaluate(root, checkpoint, destination, *, parent=False, split='validation', limit=None, plots=True, device='cpu',
             smoothing_window=0.):
    destination.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-s','-u',str(Path(__file__).resolve().parents[2]/'jobs/utils/container_python.py'),
        '-m','evaluation.evaluate_eskf','--config',str(root/'configs'/('stage2.conf' if parent else 'adapter.conf')),
        '--ckpt',str(checkpoint),'--dataset-config',str(root/'configs/dataset.conf'),
        '--sensor-metadata-root',str(root/'configs/initial_sensor_biases'),'--output',str(destination),
        '--split',split,'--expected-flights',str(23 if split=='validation' else 12),'--full-sequence',
        '--device',device,'--ekf-device','cpu','--cpu-threads','1','--height-updates',
        '--calibration-window','.05','--orientation-std','.001',
        '--calibration-max-step-deg','45','--calibration-max-rate','20','--plots' if plots else '--no-plots']
    if limit is not None:command+=['--limit-flights',str(limit)]
    if smoothing_window:command+=['--orientation-smoothing-window',str(smoothing_window)]
    write(destination/'command.json',{'command':command})
    print(f'Fresh inference/replay: {destination}; log={destination/"job.log"}',flush=True)
    with (destination/'job.log').open('w') as handle:
        subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True)
    verification=json.loads((destination/'evaluation_verification.json').read_text())
    assert verification['optimizer_steps']==0 and verification['weights_and_buffers_unchanged']
    if plots:verify_plot_outputs(destination)
    return verification


def preflight(root):
    plan=check_inputs(root);torch.set_num_threads(1);torch.manual_seed(0)
    out=root/'preflight';out.mkdir(exist_ok=True)
    all_flights,cutoffs=apply_sensor_metadata(read_dataset_flights(root/'configs/dataset.conf'),root/'configs/initial_sensor_biases')
    assert {split:sum(row['split']==split for row in all_flights) for split in ('train','validation','test')}==plan['split_counts']
    flight=next(row for row in all_flights if row['split']=='train')
    real=prepare_flight(flight,cutoffs[flight['id']],DifferentiableESKF())
    parent=root/'inputs/BStrong_epoch50.ckpt'
    assert torch.load(parent,map_location='cpu',weights_only=True)['epoch']==50
    baseline=load_airimu(root/'configs/stage2.conf',parent,DifferentiableESKF(),'cpu').eval()
    ekf,fixture=flight_fixture()
    model=load_airimu(root/'configs/adapter.conf',parent,ekf,'cpu',configuration='Adapter',initialize_adapter=True).train()
    assert model.network.raw_cov_adapter is not None and not model.network.cnn.training
    trainable=[name for name,p in model.network.named_parameters() if p.requires_grad]
    assert trainable and all(name.startswith('raw_cov_adapter.') for name in trainable)
    assert sum(p.numel() for p in model.network.parameters() if p.requires_grad)==8806
    for source in (real,fixture):
        with torch.no_grad():
            q0=baseline.predict_variance(source['raw_acceleration'][:256],source['raw_gyroscope'][:256])
            q1=model.predict_variance(source['raw_acceleration'][:256],source['raw_gyroscope'][:256])
        assert torch.equal(q0,q1),'Zero adapter differs from B Strong'
    original={name:value.clone() for name,value in model.network.state_dict().items()}
    optimizer=torch.optim.Adam(model.optimizer_groups(adapter_lr=1e-4))
    initial=root/'inputs/adapter_initial.ckpt'
    save_checkpoint(initial,model,optimizer,0,math.inf,{'configuration':'Adapter','parent_epoch':50,
        'parent_checkpoint_sha256':file_sha256(parent),'adapter_zero_initialized':True})
    logged=[]
    with patch('ekf.runner.prepare_flight',return_value=fixture):
        with FlightTrainer(model,optimizer,adapter_weight=1e-3,max_relative_update=.01,
                loss_options={'gyro_weight':1e-4,'calibration_window':2.}) as trainer:
            for epoch in (1,2):
                trainer.run_split([{'id':'synthetic_preflight_only'}],{'synthetic_preflight_only':1.5},epoch=epoch,
                    phase='train',epoch_start=0.,log=logged.append)
    for name,value in model.network.state_dict().items():
        if not name.startswith('raw_cov_adapter.'):assert torch.equal(value,original[name]),name
    for index in (0,2):
        assert not torch.equal(model.network.raw_cov_adapter[index].weight,original[f'raw_cov_adapter.{index}.weight'])
    assert logged[-1]['gradient_norm_velocity']>0 and logged[-1]['gradient_norm_adapter_regularization']>0
    sample=out/'throwaway_adapter.ckpt'
    save_checkpoint(sample,model,optimizer,2,logged[-1]['train_xy_mse'],{'configuration':'Adapter','preflight_only':True})
    restored=load_airimu(root/'configs/adapter.conf',sample,DifferentiableESKF(),'cpu',configuration='Adapter').eval()
    assert all(torch.equal(value,restored.network.state_dict()[name]) for name,value in model.network.state_dict().items())
    # Fresh inference with a trained throwaway adapter proves loading, replay,
    # metric/artifact creation and all five velocity/uncertainty plots work.
    verification=evaluate(root,sample,out/'sample_validation_evaluation',limit=1,plots=True)
    result={'all_required_checks_passed':True,'parent_epoch':50,'zero_adapter_exact_identity':True,
        'frozen_weights_and_batchnorm_buffers_unchanged':True,'trainable_parameters':8806,
        'both_adapter_layers_update_after_two_steps':True,'strict_checkpoint_round_trip':True,
        'velocity_gradient_norm':logged[-1]['gradient_norm_velocity'],
        'adapter_regularizer_gradient_norm':logged[-1]['gradient_norm_adapter_regularization'],
        'sample_validation_evaluation':verification,'reserved_eval_flights_used':0,
        'production_initial_checkpoint':str(initial),'throwaway_updates_excluded_from_training':True}
    write(root/'preflight_verification.json',result)
    plan['preflight']=result;plan['status']='local checks passed; ready for PBS'
    plan['input_sha256']['inputs/adapter_initial.ckpt']=file_sha256(initial)
    plan['input_sha256']['preflight_verification.json']=file_sha256(root/'preflight_verification.json')
    write(root/'run_plan.json',plan);print(json.dumps(result,indent=2),flush=True)


def verify_training(root):
    plan=check_inputs(root);training=root/'training';metrics=rows(training/'metrics.jsonl');flights=rows(training/'flights.jsonl')
    start_epoch=plan.get('resume_epoch',0)
    summary=json.loads((training/'training_summary.json').read_text())
    initial=json.loads((training/'initial_validation.json').read_text())
    best=torch.load(training/'best_model.ckpt',map_location='cpu',weights_only=True)
    newest=torch.load(training/'newest.ckpt',map_location='cpu',weights_only=True)
    parent=torch.load(root/'inputs/BStrong_epoch50.ckpt',map_location='cpu',weights_only=True)['model_state_dict']
    selected=min([(start_epoch,initial['validation_xy_mse'])]+[(row['epoch'],row['validation_xy_mse']) for row in metrics],key=lambda x:x[1])
    assert best['epoch']==selected[0] and math.isclose(best['best_loss'],selected[1],rel_tol=1e-12)
    assert newest['epoch']==summary['epochs_completed'] and 1<=len(metrics)<=plan['epochs']
    assert [row['epoch'] for row in metrics]==list(range(start_epoch+1,start_epoch+len(metrics)+1))
    for epoch in range(start_epoch+1,start_epoch+len(metrics)+1):
        assert sum(row['epoch']==epoch and row['phase']=='train' for row in flights)==82
        assert sum(row['epoch']==epoch and row['phase']=='validation' for row in flights)==23
    for checkpoint in (best,newest):
        state=checkpoint['model_state_dict']
        assert all(torch.equal(value,state[name]) for name,value in parent.items())
        assert all(bool(torch.isfinite(value).all()) for value in state.values())
        assert checkpoint['ekf_metadata']['configuration']=='Adapter'
        assert checkpoint['ekf_metadata']['optimizer_groups']==[{'name':'raw_cov_adapter','lr':1e-4}]
    batches=[row for row in rows(training/'batches.jsonl') if row['phase']=='train']
    assert len(batches)==21*len(metrics)
    assert all(set(row['parameter_updates_by_group'])=={'raw_cov_adapter'} for row in batches)
    assert all(row['gradient_norm_before_clip']>=0 and row['gradient_norm_after_clip']<=1.00002 for row in batches)
    if start_epoch:
        resumed=torch.load(root/'inputs/adapter_initial.ckpt',map_location='cpu',weights_only=True)
        old_steps={int(state['step']) for state in resumed['optimizer_state_dict']['state'].values()}
        steps={int(state['step']) for state in newest['optimizer_state_dict']['state'].values()}
        assert old_steps=={summary['restored_adam_steps']}
        assert steps=={summary['restored_adam_steps']+len(batches)}
    result={'epochs_completed':start_epoch+len(metrics),'additional_epochs_completed':len(metrics),'starting_epoch':start_epoch,'best_epoch':best['epoch'],
        'best_validation_xy_rmse':math.sqrt(selected[1]),'initial_BStrong_validation_xy_rmse':initial['validation_xy_rmse'],
        'parent_weights_and_buffers_unchanged':True,'optimizer_updates':len(batches),
        'selected_checkpoint_sha256':file_sha256(training/'best_model.ckpt'),
        'checkpoint_selection_split':'validation','held_out_eval_used':False}
    if start_epoch:
        result.update(initial_adapter_validation_xy_rmse=initial['validation_xy_rmse'],
            restored_adam_steps=summary['restored_adam_steps'],optimizer_state_restored_and_advanced=True)
        result.pop('initial_BStrong_validation_xy_rmse')
    write(root/'training_verification.json',result)
    write_csv(root/'epoch_timing.csv',[{key:row[key] for key in ('epoch','train_seconds','validation_seconds',
        'epoch_seconds','train_xy_rmse','validation_xy_rmse')} for row in metrics])
    print(json.dumps(result,indent=2),flush=True)
    return result


def comparison(root, paths, split, expected):
    loaded={label:summarize_evaluation(path,expected,required_split='validation' if split=='validation' else 'test')
            for label,path in paths.items()}
    base=loaded['BStrong'];adapted=loaded['Adapter']
    policy=('dataset_config_sha256','internal_split','dataset_split_counts','full_sequence','evaluation_interval',
            'height_updates','height_measurement_std','calibration_options')
    assert all(base[1].get(key)==adapted[1].get(key) for key in policy)
    assert canonical_flights(base[1])==canonical_flights(adapted[1]) and base[3]==adapted[3]
    pairs=[];by_id={row['id']:row for row in adapted[2]}
    for row in base[2]:
        other=by_id[row['id']];assert row['gyro_reference_quality']==other['gyro_reference_quality']
        a=row['ekf_vs_vicon']['xy_rmse'];b=other['ekf_vs_vicon']['xy_rmse']
        pairs.append({'flight':row['id'],'BStrong_xy_rmse':a,'Adapter_xy_rmse':b,'improvement_percent_vs_BStrong':improvement(a,b)})
    summaries={label:value[0] for label,value in loaded.items()}
    previous_run=json.loads((root/'run_plan.json').read_text()).get('previous_run')
    previous_summary=None
    if previous_run:
        previous=summarize_evaluation(Path(previous_run)/'evaluation'/split/'Adapter',expected,
                                     required_split='validation' if split=='validation' else 'test')
        assert previous[3]==adapted[3] and canonical_flights(previous[1])==canonical_flights(adapted[1])
        assert all(previous[1].get(key)==adapted[1].get(key) for key in policy)
        previous_by_id={row['id']:row for row in previous[2]}
        for row in pairs:
            value=previous_by_id[row['flight']]['ekf_vs_vicon']['xy_rmse']
            row['Adapter15_xy_rmse']=value
            row['improvement_percent_vs_Adapter15']=improvement(value,row['Adapter_xy_rmse'])
        previous_summary={'checkpoint_epoch':15,
            'aggregate_horizontal_velocity_rmse':previous[0]['aggregate_horizontal_velocity_rmse'],
            'aggregate_improvement_percent':improvement(previous[0]['aggregate_horizontal_velocity_rmse'],
                                                      adapted[0]['aggregate_horizontal_velocity_rmse']),
            'flights_improved':sum(row['Adapter_xy_rmse']<row['Adapter15_xy_rmse'] for row in pairs),
            'worst_flight_rmse':previous[0]['worst_flight_rmse']}
    for label,record in summaries.items():
        record['motion_covariance']=motion_summary(paths[label])
        record['flights_over_2_mps']=sum(row['ekf_vs_vicon']['xy_rmse']>2. for row in loaded[label][2])
    inflation={sensor:[b/a for a,b in zip(base[0]['covariance_full_sequence'][sensor]['mean'],
                                       adapted[0]['covariance_full_sequence'][sensor]['mean'])] for sensor in ('acc_cov','gyro_cov')}
    report={'split':split,'flight_count':expected,'models':summaries,'paired_flights':pairs,
        'flights_improved_over_BStrong':sum(row['Adapter_xy_rmse']<row['BStrong_xy_rmse'] for row in pairs),
        'aggregate_improvement_percent_vs_BStrong':improvement(base[0]['aggregate_horizontal_velocity_rmse'],adapted[0]['aggregate_horizontal_velocity_rmse']),
        'mean_variance_ratio_Adapter_over_BStrong_xyz':inflation,
        'lower_aggregate_rmse':adapted[0]['aggregate_horizontal_velocity_rmse']<base[0]['aggregate_horizontal_velocity_rmse'],
        'fewer_flights_over_2_mps':adapted[0]['flights_over_2_mps']<base[0]['flights_over_2_mps'],
        'uncertainty_assessment':'Inspect covariance ratios/percentiles and quiet/rapid variation; velocity gain alone does not establish calibration.',
        'checkpoint_selection':'validation only, including epoch-0 frozen parent; eval never used for selection'}
    if previous_summary is not None:
        report['epoch15_comparison']=previous_summary
        report['checkpoint_selection']='validation only, including resumed epoch-15 adapter; eval never used for selection'
    output=root/'comparison'/split;output.mkdir(parents=True,exist_ok=True)
    write(output/'comparison.json',report);write_csv(output/'comparison_flights.csv',pairs)
    covariance_rows=[]
    for label,record in summaries.items():
        scopes={'full_sequence':record['covariance_full_sequence'],'scored_interval':record['covariance_scored_interval']}
        for scope,groups in record['motion_covariance'].items():
            scopes.update({scope+'_'+motion:value for motion,value in groups.items() if value['samples']})
        for scope,value in scopes.items():
            for sensor in ('acc_cov','gyro_cov'):
                for axis in range(3):
                    covariance_rows.append({'model':label,'scope':scope,'sensor':sensor,'axis':'xyz'[axis],
                        **{key:values[axis] for key,values in value[sensor].items()}})
    write_csv(output/'comparison_covariance.csv',covariance_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for label,record in summaries.items():
        axes[0,0].bar(label,record['aggregate_horizontal_velocity_rmse'])
        axes[0,1].plot([row[label+'_xy_rmse'] for row in pairs],label=label,marker='.')
        for ax,sensor in zip(axes[1],('acc_cov','gyro_cov')):
            groups=record['motion_covariance']['full_sequence']
            ax.plot(['quiet','moderate','rapid'],[math.sqrt(sum(groups[motion][sensor]['mean'])/3)
                    if groups[motion]['samples'] else math.nan for motion in ('quiet','moderate','rapid')],label=label,marker='.')
    for ax,title in zip(axes.flat,('Aggregate XY RMSE (m/s)','Per-flight XY RMSE (m/s)',
                                 'Accel sqrt(mean variance) (m/s²)','Gyro sqrt(mean variance) (rad/s)')):
        ax.set_title(title);ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:ax.legend()
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    import os
    manifests={label:verify_plot_outputs(path) for label,path in paths.items()}
    parts=[f'<!doctype html><meta charset="utf-8"><h1>B Strong residual adapter: {split}</h1>',
        '<p>Velocity bands: ±2√Pvv. IMU uncertainty panels: √variance. Scoring: cutoff+5 to cutoff+25 seconds.</p>',
        '<p><a href="comparison.json">Metrics</a> · <a href="comparison_flights.csv">Per-flight CSV</a> · <a href="comparison_covariance.csv">Covariance CSV</a></p>',
        '<img src="comparison.png" style="max-width:100%">']
    for row in pairs:
        parts.append(f'<details><summary>{html.escape(row["flight"])}; improvement {row["improvement_percent_vs_BStrong"]:+.2f}%</summary>')
        if previous_summary is not None:
            parts.append(f'<p>Epoch 15 RMSE: {row["Adapter15_xy_rmse"]:.4f} m/s; '
                         f'change vs epoch 15: {row["improvement_percent_vs_Adapter15"]:+.2f}% improvement.</p>')
        for label,path in paths.items():
            image=path/'plots'/manifests[label][row['flight']]['scoring_window']
            parts.append(f'<h3>{label}; RMSE {row[label+"_xy_rmse"]:.4f} m/s</h3><img loading="lazy" src="{html.escape(os.path.relpath(image,output))}" style="max-width:100%">')
        parts.append('</details>')
    (output/'index.html').write_text('\n'.join(parts))
    return report


def finish(root, device):
    verification=verify_training(root)
    selected=root/'training/best_model.ckpt';selected_hash=verification['selected_checkpoint_sha256']
    plan=check_inputs(root);plan.update(status='training verified; fresh validation/eval inference and plots',training_verification=verification)
    write(root/'run_plan.json',plan)
    for split,expected in (('validation',23),('eval',12)):
        paths={label:root/'evaluation'/split/label for label in ('BStrong','Adapter')}
        for label,path in paths.items():
            evaluate(root,root/'inputs/BStrong_epoch50.ckpt' if label=='BStrong' else selected,path,
                     parent=label=='BStrong',split=split,device=device)
        report=comparison(root,paths,split,expected)
        assert file_sha256(selected)==selected_hash
        print(f'{split}: aggregate improvement vs B Strong {report["aggregate_improvement_percent_vs_BStrong"]:+.4f}%; '
              f'improved flights {report["flights_improved_over_BStrong"]}/{expected}',flush=True)
    check_inputs(root)
    plan.update(status='complete; adapter training, validation and held-out eval with plots verified',
                completed_sgt=datetime.now(ZoneInfo('Asia/Singapore')).isoformat())
    write(root/'run_plan.json',plan);(root/'COMPLETE').write_text(plan['completed_sgt']+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',required=True)
    parser.add_argument('--phase',choices=('preflight','finish'),required=True);parser.add_argument('--device',default='cpu')
    args=parser.parse_args();root=Path(args.run_root).resolve();torch.set_num_threads(1)
    if args.phase=='preflight':preflight(root)
    else:finish(root,args.device)


if __name__=='__main__':main()
