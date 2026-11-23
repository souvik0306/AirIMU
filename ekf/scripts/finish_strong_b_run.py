"""Verify B Strong and compare its validation results with frozen A/B controls."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from ekf.experiment import file_sha256
from ekf.gradients import summarize_batches
from ekf.scripts.compare_training_runs import summarize


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def canonical_flights(metadata):
    flights = json.loads(json.dumps(metadata['flights']))
    for flight in flights:
        flight.get('sensor_metadata',{}).pop('source_csv',None)
    return sorted(flights,key=lambda row:row['id'])


def verify_comparable(runs):
    metadata = [json.loads((run/'eskf_metadata.json').read_text()) for run in runs]
    for row in metadata[1:]:
        for key in ('initial_checkpoint_sha256','seed','loss_options','height_updates',
                    'height_measurement_std','covariance_policy','flight_batch_size',
                    'replay_backend','replay_workers','feature_cache_mb','gyro_reference_quality_policy',
                    'gravity','evaluation_start_offset','evaluation_duration','correction_weight'):
            if row.get(key) != metadata[0].get(key):
                raise ValueError(f'Control comparison differs in {key}.')
        if canonical_flights(row) != canonical_flights(metadata[0]):
            raise ValueError('Control comparison has different flights, biases or cutoffs.')


def write_csv(path, rows):
    with path.open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def make_report(root):
    runs = [root/'controls/A',root/'controls/B',root/'training']
    labels = ('A_25','B_25','B_Strong_50')
    verify_comparable(runs)
    records = [summarize(run) for run in runs]
    initial = records[0]['initial_validation']['validation_xy_rmse']
    for label,record in zip(labels,records):
        if not math.isclose(record['initial_validation']['validation_xy_rmse'],initial,rel_tol=1e-7):
            raise ValueError('The untouched Stage 1 validation baseline did not reproduce.')
        record.update(label=label,improvement_percent_vs_initial=100*(1-record['validation_xy_rmse']/initial))
    ids = set(records[0]['flights'])
    if any(set(record['flights']) != ids for record in records):
        raise ValueError('Comparison requires identical validation flights.')
    baseline = {row['flight']:row['validation_xy_rmse'] for row in
                read_rows(runs[0]/'initial_validation_flights.jsonl')}
    paired = []
    for flight in sorted(ids):
        values = [record['flights'][flight] for record in records]
        if any(value['reference_quality'] != values[0]['reference_quality'] for value in values):
            raise ValueError(f'Gyro reference validity differs for {flight}.')
        a,b,strong = [value['xy_rmse'] for value in values]
        paired.append({'flight':flight,'initial_xy_rmse':baseline[flight],
            'a_xy_rmse':a,'b_xy_rmse':b,'b_strong_xy_rmse':strong,
            'strong_improvement_percent_vs_a':100*(1-strong/a),
            'strong_improvement_percent_vs_b':100*(1-strong/b),
            'strong_improvement_percent_vs_initial':100*(1-strong/baseline[flight])})
    epochs = [read_rows(run/'metrics.jsonl') for run in runs]
    strong_epochs = epochs[-1]
    endpoint25 = next(row for row in strong_epochs if row['epoch']==25)
    def selected(rows):
        best = min(rows,key=lambda row:row['validation_xy_mse'])
        return {key:best[key] for key in ('epoch','train_xy_rmse','validation_xy_rmse')}
    report = {'runs':records,'initial_validation_xy_rmse':initial,
        'strong_epoch25':{key:endpoint25[key] for key in ('epoch','train_xy_rmse','validation_xy_rmse')},
        'strong_best_epochs_1_to_25':selected(strong_epochs[:25]),
        'strong_best_epochs_26_to_50':selected(strong_epochs[25:]),
        'strong_improvement_percent_vs_a':100*(1-records[-1]['validation_xy_rmse']/records[0]['validation_xy_rmse']),
        'strong_improvement_percent_vs_b':100*(1-records[-1]['validation_xy_rmse']/records[1]['validation_xy_rmse']),
        'first_strong_epoch_beating_a':next((row['epoch'] for row in strong_epochs
            if row['validation_xy_rmse']<records[0]['validation_xy_rmse']),None),
        'strong_better_flights_vs_a':sum(row['b_strong_xy_rmse']<row['a_xy_rmse'] for row in paired),
        'strong_better_flights_vs_b':sum(row['b_strong_xy_rmse']<row['b_xy_rmse'] for row in paired),
        'paired_flights':paired,
        'interpretation':'Single-seed experiment changing GRU2 LR, clipping and training duration together; '
            'A/B controls have 25 epochs. Report epoch-25 and best-of-50 results separately. '
            'This does not isolate the effect of each change. Held-out test data are not evaluated.'}
    output = root/'comparison';output.mkdir(exist_ok=True)
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    write_csv(output/'comparison_flights.csv',paired)
    curves = [{'run':label,**{key:row[key] for key in
        ('epoch','train_xy_rmse','validation_xy_rmse','epoch_seconds')}}
        for label,rows in zip(labels,epochs) for row in rows]
    write_csv(output/'comparison_epochs.csv',curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for label,rows,record in zip(labels,epochs,records):
        for axis,key in zip(axes[0],('validation_xy_rmse','train_xy_rmse')):
            axis.plot([row['epoch'] for row in rows],[row[key] for row in rows],label=label)
        if label != labels[-1]:
            axes[0,0].axhline(record['validation_xy_rmse'],linestyle=':',alpha=.5)
    axes[0,0].set(ylabel='Validation XY RMSE (m/s)')
    axes[0,1].set(ylabel='Training XY RMSE (m/s)')
    for name in ('gru2','heads_and_bias_gains'):
        axes[1,0].plot([row['epoch'] for row in strong_epochs],
            [row['train_gradients']['clipping_groups'][name]['clip_fraction'] for row in strong_epochs],label=name)
    axes[1,0].set(ylabel='B Strong clipped batch fraction',ylim=(0,1.05))
    for name in ('gru2','covariance_heads','bias_gains'):
        axes[1,1].plot([row['epoch'] for row in strong_epochs],
            [row['train_gradients']['parameter_updates_by_group'][name]['relative_l2']['median']
             for row in strong_epochs],label=name)
    axes[1,1].set(ylabel='B Strong median relative Adam update',yscale='log')
    for axis in axes.flat:
        axis.set(xlabel='Epoch');axis.grid(alpha=.2);axis.legend()
    fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve();training = root/'training'
    plan = json.loads((root/'run_plan.json').read_text())
    metrics = read_rows(training/'metrics.jsonl')
    flights = read_rows(training/'flights.jsonl')
    batches = read_rows(training/'batches.jsonl')
    assert [row['epoch'] for row in metrics] == list(range(1,51))
    for epoch in range(1,51):
        assert Counter(row['phase'] for row in flights if row['epoch']==epoch)=={'train':82,'validation':23}
        assert Counter(row['phase'] for row in batches if row['epoch']==epoch)=={'train':21,'validation':6}
    original = torch.load(root/'inputs/initial_airimu.ckpt',map_location='cpu',weights_only=True)['model_state_dict']
    newest = torch.load(training/'newest.ckpt',map_location='cpu',weights_only=True)
    best = torch.load(training/'best_model.ckpt',map_location='cpu',weights_only=True)
    selected = min(metrics,key=lambda row:row['validation_xy_mse'])
    assert newest['epoch']==50 and best['epoch']==selected['epoch']
    assert math.isclose(best['best_loss'],selected['validation_xy_mse'],rel_tol=1e-12)
    metadata = newest['ekf_metadata']
    assert metadata['configuration']=='B' and metadata['gradient_clip'] is None
    assert metadata['gradient_clipping']==plan['gradient_clipping']
    assert {group['name']:group['lr'] for group in metadata['optimizer_groups']} == {
        'covariance_heads':1e-5,'bias_gains':1e-5,'gru2':3e-6}
    prefixes = ('acccov_decoder.','gyrocov_decoder.','acc_bias_gain.','gyro_bias_gain.','gru2.')
    assert newest['model_state_dict'].keys()==original.keys()
    changed = [name for name,value in newest['model_state_dict'].items() if not torch.equal(value,original[name].to(value))]
    frozen = [name for name in original if not name.startswith(prefixes)]
    assert not set(changed).intersection(frozen)
    assert all(any(name.startswith(prefix) for name in changed) for prefix in prefixes)
    for checkpoint in (newest,best):
        assert all(bool(torch.isfinite(value).all()) for value in checkpoint['model_state_dict'].values())
    initial = json.loads((training/'initial_validation.json').read_text())
    assert initial['optimizer_steps']==0 and initial['weights_unchanged']
    train_batches = [row for row in batches if row['phase']=='train']
    assert len(train_batches)==1050
    for row in train_batches:
        assert row['gradient_clip_scale'] is None
        for name,limit in (('gru2',5.),('heads_and_bias_gains',1.)):
            assert row['gradient_clipping_groups'][name]['norm_after'] <= limit+2e-5
        for group in row['parameter_updates_by_group'].values():
            assert all(math.isfinite(value) for value in group.values())
            assert group['relative_l2'] <= metadata['stability_policy']['max_relative_update']
    for name,expected in plan['input_sha256'].items():assert file_sha256(root/name)==expected,name
    verification = {'epochs_completed':50,'best_epoch':best['epoch'],
        'best_validation_xy_rmse':math.sqrt(selected['validation_xy_mse']),
        'initial_validation_xy_rmse':initial['validation_xy_rmse'],
        'frozen_tensors_checked':len(frozen),'frozen_tensors_changed':[],
        'all_weights_finite':True,'optimizer_steps':len(train_batches),
        'gradient_clipping':metadata['gradient_clipping'],'gradient_summary':summarize_batches(train_batches)}
    (root/'completion_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    write_csv(root/'epoch_timing.csv',[{key:row[key] for key in ('epoch','train_seconds',
        'validation_seconds','checkpoint_seconds','epoch_seconds','train_xy_rmse','validation_xy_rmse')} for row in metrics])
    report = make_report(root)
    completed = datetime.now(ZoneInfo('Asia/Singapore')).isoformat()
    (root/'COMPLETE').write_text(completed+'\n')
    (root/'comparison/COMPLETE').write_text(completed+'\n')
    plan.update(status='complete; B Strong verified and comparison generated',completed_sgt=completed,
                comparison=str(root/'comparison/comparison.json'))
    (root/'run_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key not in ('runs','paired_flights')},indent=2),flush=True)


if __name__=='__main__':main()
