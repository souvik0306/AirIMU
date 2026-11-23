"""Validation milestones and final selected inference for SO(3)/strong-penalty adapter."""

import argparse
from datetime import datetime
import hashlib
import html
import json
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from ekf import DifferentiableESKF
from ekf.experiment import file_sha256, load_airimu
from ekf.regularization import gyro_reference
from ekf.scripts.residual_adapter import check_inputs, evaluate, verify_training, write
from evaluation.batchnorm_diagnostic import motion_summary
from evaluation.compare_eskf_models import summarize_evaluation, canonical_flights, write_csv, improvement
from evaluation.plot_eskf import verify_plot_outputs


def reference_summary(path, cache):
    """Re-score saved trajectories, with identical Q and biases for both references."""
    cache.mkdir(parents=True,exist_ok=True)
    residuals={label:[] for label in ('original','filtered')};effective={label:[] for label in residuals}
    changes=[];samples=rejected=0;per_flight=[];roughness={label:[] for label in residuals}
    for row in json.loads((path/'metrics.json').read_text()):
        saved=torch.load(path/row['trajectory_file'],map_location='cpu',weights_only=True)
        pred=torch.load(path/row['prediction_file'],map_location='cpu',weights_only=True)
        digest=hashlib.sha256()
        for key in ('imu_time','vicon_time_gt','vicon_quaternion_gt'):
            digest.update(saved[key].contiguous().numpy().tobytes())
        target=cache/(digest.hexdigest()+'.pt')
        if target.exists():reference=torch.load(target,weights_only=True)
        else:
            reference=gyro_reference(saved['imu_time'],saved['vicon_quaternion_gt'],orientation_time=saved['vicon_time_gt'],
                                     smoothing_window_seconds=.05)
            torch.save(reference,target)
        mask=reference['mask'];samples+=int(mask.sum());rejected+=int((~mask).sum())
        assert torch.equal(mask,saved['gyro_reference_mask'])
        q=saved['imu_variance'][:,3:].double();measured=pred['raw_gyroscope'].double()-saved['trajectory']['gyroscope_bias'].double()
        delta=(reference['rate']-reference['original_rate'])[mask];changes.append(delta)
        record={'flight':row['id'],'valid_samples':int(mask.sum()),'rejected_samples':int((~mask).sum())}
        for label,key,var in (('original','original_rate','original_variance'),('filtered','rate','variance')):
            error=(measured-reference[key])[mask];eq=(q+reference[var])[mask]
            residuals[label].append(error);effective[label].append(eq)
            consecutive=mask[1:]&mask[:-1]
            roughness[label].append((reference[key].diff(dim=0)/saved['imu_time'].diff()[:,None])[consecutive])
            if len(error):
                record[label+'_rms_xyz']=error.square().mean(0).sqrt().tolist()
                record[label+'_coverage_xyz']=(error.abs()<=eq.sqrt()).double().mean(0).tolist()
                record[label+'_loss']=float((error.square()/eq+eq.log()).mean())
        per_flight.append(record)
    report={'valid_samples':samples,'rejected_samples':rejected,'same_raw_quality_mask':True,
        'scope':'full flight; original and filtered references use identical predicted Q and saved EKF biases',
        'reference_noise_policy':'retain original endpoint-noise floor; max with linearized correlated-endpoint bound',
        'smoothing_window_seconds':.05,'systematic_errors_removed':False,'per_flight':per_flight}
    for label in residuals:
        error=torch.cat(residuals[label]);eq=torch.cat(effective[label])
        report[label]={'residual_rms_xyz':error.square().mean(0).sqrt().tolist(),
            'residual_abs_p95_xyz':error.abs().quantile(.95,dim=0).tolist(),
            'residual_abs_max_xyz':error.abs().amax(0).tolist(),
            'one_sigma_coverage_xyz':(error.abs()<=eq.sqrt()).double().mean(0).tolist(),
            'sample_weighted_calibration_loss':float((error.square()/eq+eq.log()).mean())}
        derivative=torch.cat(roughness[label])
        report[label]['reference_rate_change_rms_xyz']=derivative.square().mean(0).sqrt().tolist()
        report[label]['roughness_scope']='adjacent valid samples only; real angular acceleration also contributes'
    change=torch.cat(changes);report['reference_change_rms_xyz']=change.square().mean(0).sqrt().tolist()
    report['reference_change_abs_p95_xyz']=change.abs().quantile(.95,dim=0).tolist()
    report['reference_change_abs_max_xyz']=change.abs().amax(0).tolist()
    return report


def compare(root, destination, controls, output, expected=23, split='validation'):
    paths={**controls,'Strong':destination}
    loaded={label:summarize_evaluation(path,expected,required_split='validation' if split=='validation' else 'test')
            for label,path in paths.items()}
    baseline=loaded['BStrong'];policy=('dataset_config_sha256','internal_split','dataset_split_counts','full_sequence',
        'evaluation_interval','height_updates','height_measurement_std')
    cache=root/'diagnostics/reference_cache';models={}
    for label,(summary,metadata,flights,fingerprints) in loaded.items():
        assert all(metadata.get(key)==baseline[1].get(key) for key in policy)
        assert canonical_flights(metadata)==canonical_flights(baseline[1]) and fingerprints==baseline[3]
        for key in ('window_seconds','orientation_std','max_step_degrees','max_angular_rate'):
            assert metadata['calibration_options'][key]==baseline[1]['calibration_options'][key]
        old_quality={row['id']:row['gyro_reference_quality'] for row in baseline[2]}
        assert all(row['gyro_reference_quality']==old_quality[row['id']] for row in flights)
        models[label]={**summary,'motion_covariance':motion_summary(paths[label]),
            'native_calibration_reference':'filtered' if metadata['calibration_options'].get('orientation_smoothing_window',0.) else 'original',
            'recommended_calibration_comparison':'paired_reference_diagnostics.filtered for all models',
            'paired_reference_diagnostics':reference_summary(paths[label],cache),
            'flights_over_2_mps':sum(row['ekf_vs_vicon']['xy_rmse']>2 for row in flights)}
    by_id={label:{row['id']:row for row in result[2]} for label,result in loaded.items()}
    paired=[]
    for flight in sorted(by_id['BStrong']):
        row={'flight':flight}
        for label in loaded:row[label+'_rmse']=by_id[label][flight]['ekf_vs_vicon']['xy_rmse']
        for label in controls:row['Strong_improvement_percent_vs_'+label]=improvement(row[label+'_rmse'],row['Strong_rmse'])
        paired.append(row)
    changes={}
    for label in controls:
        changes[label]={'aggregate_improvement_percent':improvement(models[label]['aggregate_horizontal_velocity_rmse'],
            models['Strong']['aggregate_horizontal_velocity_rmse']),
            'flights_improved':sum(row['Strong_rmse']<row[label+'_rmse'] for row in paired),
            'flights_deteriorated':sum(row['Strong_rmse']>row[label+'_rmse'] for row in paired),
            'mean_variance_ratio_xyz':{sensor:[a/b for a,b in zip(models['Strong']['covariance_full_sequence'][sensor]['mean'],
                models[label]['covariance_full_sequence'][sensor]['mean'])] for sensor in ('acc_cov','gyro_cov')}}
    report={'split':split,'flight_count':expected,'models':models,'paired_flights':paired,'comparisons':changes,
        'controls_reused_without_rerunning_inference_or_EKF':True,
        'selection':'minimum validation MSE; individual regressions reported, reserved eval never selects checkpoint',
        'interpretation':'penalty and filtering change together; paired same-Q/same-bias reference diagnostics isolate reference preprocessing only'}
    output.mkdir(parents=True,exist_ok=True);write(output/'comparison.json',report);write_csv(output/'comparison_flights.csv',paired)
    cov=[]
    for label,summary in models.items():
        scopes={'full_sequence':summary['covariance_full_sequence'],'scored_interval':summary['covariance_scored_interval']}
        for scope,groups in summary['motion_covariance'].items():
            for motion,values in groups.items():
                if values['samples']:scopes[scope+'_'+motion]=values
        for scope,values in scopes.items():
            for sensor in ('acc_cov','gyro_cov'):
                for axis in range(3):cov.append({'model':label,'scope':scope,'sensor':sensor,'axis':'xyz'[axis],
                    **{key:value[axis] for key,value in values[sensor].items()}})
    write_csv(output/'comparison_covariance.csv',cov)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for label,summary in models.items():
        axes[0,0].bar(label,summary['aggregate_horizontal_velocity_rmse'])
        axes[0,1].plot([row[label+'_rmse'] for row in paired],label=label,marker='.')
        for ax,sensor in zip(axes[1],('acc_cov','gyro_cov')):
            groups=summary['motion_covariance']['full_sequence']
            ax.plot(['quiet','moderate','rapid'],[math.sqrt(sum(groups[key][sensor]['mean'])/3)
                if groups[key]['samples'] else math.nan for key in ('quiet','moderate','rapid')],label=label,marker='.')
    for ax,title in zip(axes.flat,('Aggregate XY RMSE (m/s)','Per-flight XY RMSE (m/s)',
                                 'Accel sqrt(mean variance) (m/s²)','Gyro sqrt(mean variance) (rad/s)')):
        ax.set_title(title);ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:ax.legend()
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    parts=[f'<!doctype html><meta charset="utf-8"><h1>Stronger regularization: {html.escape(split)}</h1>',
        '<p>Validation selects checkpoints. Velocity bands: ±2√Pvv; IMU uncertainty: √variance.</p>',
        '<p><a href="comparison.json">Metrics and paired reference diagnostics</a> · <a href="comparison_flights.csv">Per-flight CSV</a> · <a href="comparison_covariance.csv">Covariance CSV</a></p>',
        '<img src="comparison.png" style="max-width:100%">']
    manifests={label:verify_plot_outputs(path) for label,path in paths.items()}
    for row in paired:
        parts.append(f'<details><summary>{html.escape(row["flight"])}</summary>')
        for label,path in paths.items():
            image=path/'plots'/manifests[label][row['flight']]['scoring_window']
            parts.append(f'<h3>{label}: {row[label+"_rmse"]:.4f} m/s</h3><img loading="lazy" src="{html.escape(os.path.relpath(image,output))}" style="max-width:100%">')
        parts.append('</details>')
    (output/'index.html').write_text('\n'.join(parts))
    return report


def milestone(root, epoch, device):
    plan=check_inputs(root);checkpoint=root/'training/milestones'/f'epoch_{epoch:03d}.ckpt'
    destination=root/'milestones'/f'epoch_{epoch:03d}'/'validation'
    evaluate(root,checkpoint,destination,smoothing_window=.05,device=device)
    report=compare(root,destination,{label:Path(path) for label,path in plan['control_evaluations'].items()},
                   destination.parent/'comparison')
    metrics=[json.loads(line) for line in (root/'training/metrics.jsonl').read_text().splitlines()]
    record=next(row for row in metrics if row['epoch']==epoch)
    write(destination.parent/'training_diagnostics.json',record)
    print(f'Milestone epoch={epoch}: validation RMSE={report["models"]["Strong"]["aggregate_horizontal_velocity_rmse"]:.9g}; '
          f'comparison={destination.parent/"comparison/index.html"}',flush=True)


def finish(root, device):
    plan=check_inputs(root);verification=verify_training(root)
    checkpoint=root/'training/best_model.ckpt';selected=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert selected['ekf_metadata']['adapter_weight']==.01
    assert selected['ekf_metadata']['loss_options']['orientation_smoothing_window']==.05
    initial=torch.load(root/'inputs/adapter_initial.ckpt',map_location='cpu',weights_only=True)['model_state_dict']
    newest=torch.load(root/'training/newest.ckpt',map_location='cpu',weights_only=True)['model_state_dict']
    assert any(not torch.equal(value,newest[name]) for name,value in initial.items() if name.startswith('raw_cov_adapter.'))
    selection={'selected_epoch':selected['epoch'],'checkpoint_sha256':file_sha256(checkpoint),
        'best_validation_xy_mse':selected['best_loss'],'selection_split':'validation',
        'reserved_eval_inspected_before_selection':False}
    write(root/'selection_locked.json',selection)
    validation=root/'evaluation/validation/Strong'
    evaluate(root,checkpoint,validation,smoothing_window=.05,device=device)
    compare(root,validation,{label:Path(path) for label,path in plan['control_evaluations'].items()},root/'comparison/validation')
    # Only the locked selected configuration receives new reserved-eval inference.
    destination=root/'evaluation/eval/Strong'
    evaluate(root,checkpoint,destination,split='eval',smoothing_window=.05,device=device)
    controls={label:Path(path.replace('/validation/','/eval/')) for label,path in plan['control_evaluations'].items()}
    compare(root,destination,controls,root/'comparison/eval',expected=12,split='eval')
    assert file_sha256(checkpoint)==selection['checkpoint_sha256'];check_inputs(root)
    plan.update(status='complete; stronger-penalty/filtered-reference training, validation and selected final eval verified',
        completed_sgt=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),training_verification=verification,
        selected_checkpoint=selection,milestones_reached=[epoch for epoch in plan['milestones']
            if (root/'training/milestones'/f'epoch_{epoch:03d}.ckpt').exists()])
    write(root/'run_plan.json',plan);(root/'COMPLETE').write_text(plan['completed_sgt']+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',required=True)
    parser.add_argument('--device',default='cpu');args=parser.parse_args();torch.set_num_threads(1)
    finish(Path(args.run_root).resolve(),args.device)


if __name__=='__main__':main()
