"""Validation-only Stage 1/C BatchNorm swaps, with no training or optimizer."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

import torch

from ekf import DifferentiableESKF
from ekf.experiment import file_sha256, load_airimu
from evaluation.compare_eskf_models import canonical_flights, improvement, summarize_evaluation, write_csv
from evaluation.eskf_metrics import covariance_distribution
from evaluation.plot_eskf import verify_plot_outputs
from model.checkpoints import covariance_policy


CASES = {
    'Test1': ('Stage1', 'Stage1'),
    'Test2': ('C27', 'C27'),
    'Test3': ('C27', 'Stage1'),
    'Test4': ('Stage1', 'C27'),
}


def now():
    return datetime.now(ZoneInfo('Asia/Singapore')).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def batchnorm_keys(network):
    """Find actual BatchNorm buffers; affine weights/biases are excluded."""
    keys = []
    for name, module in network.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if not module.track_running_stats:
                raise ValueError(f'BatchNorm has no tracked running statistics: {name}')
            keys.extend(f'{name}.{suffix}' for suffix in
                        ('running_mean', 'running_var', 'num_batches_tracked'))
    if not keys:
        raise ValueError('No BatchNorm running statistics were found.')
    return keys


def swap_batchnorm_state(weights, donor, keys):
    if set(weights) != set(donor):
        raise ValueError('Parent checkpoint state keys differ.')
    result = {name: value.detach().cpu().clone() for name, value in weights.items()}
    for name in keys:
        if name not in donor or weights[name].shape != donor[name].shape:
            raise ValueError(f'Invalid BatchNorm buffer: {name}')
        result[name] = donor[name].detach().cpu().clone()
    verify_swap(result, weights, donor, keys)
    return result


def verify_swap(state, weights, donor, keys):
    if set(state) != set(weights) or set(weights) != set(donor):
        raise ValueError('Swapped checkpoint state keys differ.')
    for name, value in state.items():
        expected = donor[name] if name in keys else weights[name]
        if value.dtype != expected.dtype or not torch.equal(value.cpu(), expected.cpu()):
            raise ValueError(f'Unexpected tensor change: {name}')


def prepare(root):
    plan = json.loads((root/'run_plan.json').read_text())
    source = root/'inputs'
    wrappers = {label: load_airimu(root/'configs/stage2.conf', source/name,
                DifferentiableESKF(), 'cpu').eval().requires_grad_(False)
                for label, name in (('Stage1', 'Stage1.ckpt'), ('C27', 'C27.ckpt'))}
    states = {label: wrapper.network.state_dict() for label, wrapper in wrappers.items()}
    keys = batchnorm_keys(wrappers['Stage1'].network)
    if keys != batchnorm_keys(wrappers['C27'].network):
        raise ValueError('Parent BatchNorm modules differ.')
    (root/'checkpoints').mkdir()
    checks = {}
    for case, (weights, donor) in CASES.items():
        state = swap_batchnorm_state(states[weights], states[donor], keys)
        path = root/'checkpoints'/f'{case}.ckpt'
        provenance = {'diagnostic_only': True, 'test': case, 'weights_source': weights,
            'BatchNorm_source': donor, 'swapped_buffer_keys': keys, 'optimizer_steps': 0,
            'source_checkpoint_sha256': {name:file_sha256(source/filename)
                for name,filename in (('Stage1','Stage1.ckpt'),('C27','C27.ckpt'))}}
        torch.save({'model_state_dict':state, 'covariance_policy':covariance_policy(wrappers[weights].network),
                    'batchnorm_diagnostic':provenance}, path)
        loaded = load_airimu(root/'configs/stage2.conf',path,DifferentiableESKF(),'cpu').eval().requires_grad_(False)
        verify_swap(loaded.network.state_dict(),states[weights],states[donor],keys)
        checks[case] = {**provenance, 'checkpoint_sha256':file_sha256(path),
            'all_trainable_weights_equal_weights_source':True,
            'all_non_BatchNorm_buffers_equal_weights_source':True,
            'all_BatchNorm_buffers_equal_statistics_source':True, 'strict_reload_passed':True}
    plan['tests'] = checks
    write_json(root/'run_plan.json', plan)
    write_json(root/'checkpoint_verification.json', checks)


def verify_inputs(root, plan):
    for name, expected in plan['input_sha256'].items():
        if file_sha256(root/name) != expected:
            raise ValueError(f'Frozen diagnostic input changed: {name}')
    for case, item in plan['tests'].items():
        if file_sha256(root/'checkpoints'/f'{case}.ckpt') != item['checkpoint_sha256']:
            raise ValueError(f'Diagnostic checkpoint changed: {case}')


def evaluate_case(root, case, destination, limit=None, plots=True):
    wrapper = Path(__file__).resolve().parents[1]/'jobs/utils/container_python.py'
    command = [sys.executable,'-s','-u',str(wrapper),'-m','evaluation.evaluate_eskf',
        '--config',str(root/'configs/stage2.conf'),'--ckpt',str(root/'checkpoints'/f'{case}.ckpt'),
        '--dataset-config',str(root/'configs/dataset.conf'),
        '--sensor-metadata-root',str(root/'configs/initial_sensor_biases'),
        '--output',str(destination),'--split','validation','--expected-flights','23','--full-sequence',
        '--device','cpu','--ekf-device','cpu','--cpu-threads','1','--height-updates',
        '--calibration-window','.05','--orientation-std','.001',
        '--calibration-max-step-deg','45','--calibration-max-rate','20',
        '--plots' if plots else '--no-plots']
    if limit is not None:
        command += ['--limit-flights',str(limit)]
    destination.mkdir(parents=True)
    write_json(destination/'command.json',{'command':command,'started_sgt':now()})
    print(f'{now()} {case}: fresh validation inference/replay; log={destination/"job.log"}',flush=True)
    with (destination/'job.log').open('w') as handle:
        subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True)


def motion_summary(path):
    """Use existing fixed gyro-rate/angular-acceleration categories for all six axes."""
    pools = {scope:{label:[] for label in ('quiet','moderate','rapid')}
             for scope in ('full_sequence','scored_interval')}
    for row in json.loads((path/'metrics.json').read_text()):
        saved = torch.load(path/row['prediction_file'],map_location='cpu',weights_only=True)
        raw = saved['raw_gyroscope'].double();time = saved['imu_time'].double()
        angular_acceleration = torch.cat((torch.zeros(1),raw.diff(dim=0).norm(dim=1)/time.diff()))
        rate = raw.norm(dim=1)
        masks = {'quiet':(rate<.1)&(angular_acceleration<1.),
                 'rapid':(rate>=1.)|(angular_acceleration>=10.)}
        masks['moderate'] = ~(masks['quiet']|masks['rapid'])
        scored = (time>=row['evaluation_start'])&(time<=row['evaluation_end'])
        for scope in pools:
            for label,mask in masks.items():
                if scope=='scored_interval':mask = mask&scored
                if bool(mask.any()):pools[scope][label].append(saved['imu_variance'][mask])
    return {scope:{label:covariance_distribution(torch.cat(values)) if values else {'samples':0}
                   for label,values in groups.items()} for scope,groups in pools.items()}


def compare(root, evaluation_root=None, expected=23, plots=True):
    evaluation_root = evaluation_root or root/'evaluation'
    loaded = {case:summarize_evaluation(evaluation_root/case,expected,required_split='validation') for case in CASES}
    policy = ('dataset_config_sha256','internal_split','dataset_split_counts','full_sequence',
              'evaluation_interval','height_updates','height_measurement_std','calibration_options')
    base = loaded['Test1']
    for case, (summary,metadata,flights,fingerprints) in loaded.items():
        if any(metadata.get(key)!=base[1].get(key) for key in policy):
            raise ValueError(f'Diagnostic evaluation policy differs: {case}')
        if canonical_flights(metadata)!=canonical_flights(base[1]) or fingerprints!=base[3]:
            raise ValueError(f'Diagnostic raw flights/biases/cutoffs differ: {case}')
        base_quality = {row['id']:row['gyro_reference_quality'] for row in base[2]}
        if any(row['gyro_reference_quality']!=base_quality[row['id']] for row in flights):
            raise ValueError(f'Diagnostic reference masks differ: {case}')
        summary['motion_covariance'] = motion_summary(evaluation_root/case)
        if plots:
            if len(verify_plot_outputs(evaluation_root/case))!=expected:
                raise ValueError(f'Incomplete plots: {case}')
    by_flight = {case:{row['id']:row for row in item[2]} for case,item in loaded.items()}
    paired = []
    for flight in sorted(by_flight['Test1']):
        row = {'flight':flight}
        for case in CASES:
            row[case+'_xy_rmse'] = by_flight[case][flight]['ekf_vs_vicon']['xy_rmse']
        row['Test3_improvement_percent_vs_Test2'] = improvement(row['Test2_xy_rmse'],row['Test3_xy_rmse'])
        row['Test3_improvement_percent_vs_Test1'] = improvement(row['Test1_xy_rmse'],row['Test3_xy_rmse'])
        paired.append(row)
    t2,t3 = loaded['Test2'][0],loaded['Test3'][0]
    decision = {'Test3_improvement_percent_vs_Test2':improvement(t2['aggregate_horizontal_velocity_rmse'],t3['aggregate_horizontal_velocity_rmse']),
        'Test3_flights_improved_over_Test2':sum(row['Test3_xy_rmse']<row['Test2_xy_rmse'] for row in paired),
        'Test2_flights_improved_over_Test1':sum(row['Test2_xy_rmse']<row['Test1_xy_rmse'] for row in paired),
        'Test3_flights_improved_over_Test1':sum(row['Test3_xy_rmse']<row['Test1_xy_rmse'] for row in paired),
        'Test3_preserves_Test2_improved_flights_over_Test1':sum(row['Test2_xy_rmse']<row['Test1_xy_rmse']
            and row['Test3_xy_rmse']<row['Test1_xy_rmse'] for row in paired),
        'Test3_reduces_Test2_regressions_over_Test1':sum(row['Test2_xy_rmse']>row['Test1_xy_rmse']
            and row['Test3_xy_rmse']<row['Test2_xy_rmse'] for row in paired),
        'Test3_worst_rmse_change_vs_Test2':t3['worst_flight_rmse']-t2['worst_flight_rmse'],
        'training_started':False,'interpretation':'Diagnostic only; assess aggregate, worst flights and covariance motion dependence before a short training experiment.'}
    report = {'flight_count':expected,'split':'validation; dataset.test; reserved dataset.eval excluded',
        'tests':{case:{'weights_source':sources[0],'BatchNorm_source':sources[1],**loaded[case][0]} for case,sources in CASES.items()},
        'paired_flights':paired,'decision':decision,'optimizer_steps':0}
    output = evaluation_root.parent/'comparison';output.mkdir(exist_ok=True)
    write_json(output/'comparison.json',report);write_csv(output/'comparison_flights.csv',paired)
    write_csv(output/'comparison_summary.csv',[{'test':case,**{key:item[key] for key in
        ('aggregate_horizontal_velocity_rmse','mean_flight_rmse','median_flight_rmse','worst_flight_rmse','worst_flight_id')}}
        for case,item in report['tests'].items()])
    covariance_rows = []
    for case,item in report['tests'].items():
        scopes = {'full_sequence':item['covariance_full_sequence'],'scored_interval':item['covariance_scored_interval']}
        for scope,groups in item['motion_covariance'].items():
            scopes.update({scope+'_'+label:stats for label,stats in groups.items() if stats['samples']})
        for scope,stats in scopes.items():
            for sensor in ('acc_cov','gyro_cov'):
                for axis in range(3):
                    covariance_rows.append({'test':case,'scope':scope,'sensor':sensor,'axis':'XYZ'[axis],
                        'samples':stats['samples'],**{key:value[axis] for key,value in stats[sensor].items()}})
    write_csv(output/'comparison_covariance.csv',covariance_rows)
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes = plt.subplots(2,2,figsize=(15,10),layout='constrained')
        labels=list(CASES)
        axes[0,0].bar(labels,[report['tests'][case]['aggregate_horizontal_velocity_rmse'] for case in labels])
        axes[0,0].set(ylabel='Validation horizontal velocity RMSE (m/s)')
        axes[0,1].bar(labels,[report['tests'][case]['worst_flight_rmse'] for case in labels]);axes[0,1].set(ylabel='Worst validation flight RMSE (m/s)')
        for axis,sensor in ((axes[1,0],'acc_cov'),(axes[1,1],'gyro_cov')):
            for case in labels:
                motion=report['tests'][case]['motion_covariance']['full_sequence']
                axis.plot(['quiet','moderate','rapid'],[math.sqrt(sum(motion[label][sensor]['mean'])/3)
                    if motion[label]['samples'] else math.nan for label in ('quiet','moderate','rapid')],marker='o',label=case)
            axis.set(ylabel='Accel √mean variance (m/s²)' if sensor=='acc_cov' else 'Gyro √mean variance (rad/s)');axis.legend()
        for axis in axes.flat:axis.grid(alpha=.2)
        fig.suptitle('Validation-only BatchNorm diagnostic; no training')
        fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)
        sections=['<!doctype html><meta charset="utf-8"><title>Validation BatchNorm diagnostic</title>',
            '<h1>Validation-only BatchNorm diagnostic</h1><p>Fresh full-flight inference/EKF replay; no training. '
            'Test1: Stage1/Stage1; Test2: C27/C27; Test3: C27/Stage1; Test4: Stage1/C27 (weights/BN).</p>',
            '<p><a href="comparison_summary.csv">Summary CSV</a> · <a href="comparison_flights.csv">Per-flight CSV</a> · '
            '<a href="comparison_covariance.csv">Covariance and motion CSV</a></p><img src="comparison.png" style="max-width:100%;width:1400px">']
        for case in CASES:
            link=html.escape(os.path.relpath(evaluation_root/case/'plots/index.html',output),quote=True)
            sections.append(f'<p><a href="{link}">{case}: velocity and uncertainty plots for {expected} validation flights</a></p>')
        (output/'index.html').write_text(''.join(sections))
    print(json.dumps(decision,indent=2),flush=True)
    (output/'COMPLETE').write_text(now()+'\n')
    return report


def run(root, preflight=False):
    plan=json.loads((root/'run_plan.json').read_text());verify_inputs(root,plan)
    if not preflight and not plan.get('preflight',{}).get('passed'):
        raise ValueError('Local checkpoint swap and sample replay checks must pass first.')
    destination=root/'preflight/evaluation' if preflight else root/'evaluation'
    plan.update(status='running validation-only preflight' if preflight else 'running all four tests on 23 validation flights',started_sgt=now())
    write_json(root/'run_plan.json',plan)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(evaluate_case,root,case,destination/case,1 if preflight else None):case for case in CASES}
        for future in as_completed(futures):
            future.result();print(f'{now()} completed {futures[future]}',flush=True)
    report=compare(root,destination,1 if preflight else 23)
    verify_inputs(root,plan)
    if preflight:
        plan.update(status='local preflight passed; ready for submission',preflight={'passed':True,
            'strict_checkpoint_swaps':True,'sample_validation_flights_per_test':1,'fresh_forward_and_full_replay':True,
            'zero_optimizer_steps':True,'four_test_comparison_and_plots':True,'reserved_eval_flights_used':0})
        write_json(root/'preflight_verification.json',plan['preflight'])
    else:
        plan.update(status='complete; all four BatchNorm tests evaluated on 23 validation flights',completed_sgt=now(),decision=report['decision'])
        (root/'COMPLETE').write_text(now()+'\n')
    write_json(root/'run_plan.json',plan)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--phase',choices=('prepare','preflight','run','compare'),required=True)
    parser.add_argument('--expected-flights',type=int,default=23,help='Flight count for standalone comparison only; production replay always requires 23.')
    args=parser.parse_args();torch.set_num_threads(1);root=args.run_root.resolve()
    try:
        if args.phase=='prepare':prepare(root)
        elif args.phase=='compare':compare(root,expected=args.expected_flights)
        else:run(root,args.phase=='preflight')
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        try:
            plan_path=root/'run_plan.json'
            if plan_path.exists():
                plan=json.loads(plan_path.read_text())
                plan.update(status='failed; inspect job.log and per-test logs',failed_phase=args.phase,
                            error=f'{type(error).__name__}: {error}',failed_sgt=now())
                write_json(plan_path,plan)
            (root/('PREFLIGHT_FAILED' if args.phase=='preflight' else 'FAILED')).write_text(f'{type(error).__name__}: {error}\n')
        except OSError as marker_error:
            print(f'Could not write failure record: {marker_error}',file=sys.stderr,flush=True)
        raise


if __name__=='__main__':main()
