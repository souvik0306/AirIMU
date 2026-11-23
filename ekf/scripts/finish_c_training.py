"""Verify full-encoder C and freeze its validation-selected checkpoint before eval."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import torch

from ekf.experiment import file_sha256
from ekf.gradients import summarize_batches
from ekf.stability import ValidationEarlyStopping


def rows(path):return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',required=True)
    parser.add_argument('--allow-reduced-split',action='store_true',help='Synthetic integration fixtures only')
    args = parser.parse_args();root = Path(args.run_root).resolve();training = root/'training'
    plan = json.loads((root/'run_plan.json').read_text())
    summary = json.loads((training/'training_summary.json').read_text())
    run_metadata = json.loads((training/'eskf_metadata.json').read_text())
    metrics = rows(training/'metrics.jsonl');flights = rows(training/'flights.jsonl');batches = rows(training/'batches.jsonl')
    epochs = len(metrics)
    maximum = summary['maximum_epochs'] if args.allow_reduced_split else 50
    counts = ({'train':run_metadata['split_counts']['train'],'validation':run_metadata['split_counts']['validation']}
              if args.allow_reduced_split else {'train':82,'validation':23})
    batch_size = run_metadata['flight_batch_size']
    batch_counts = {name:math.ceil(count/batch_size) for name,count in counts.items()}
    assert 1<=epochs<=maximum and summary['epochs_completed']==epochs
    assert [row['epoch'] for row in metrics]==list(range(1,epochs+1))
    assert summary['checkpoint_selection_split']=='validation' and not summary['held_out_eval_used']
    assert epochs==maximum or summary['early_stopped']
    for epoch in range(1,epochs+1):
        assert Counter(row['phase'] for row in flights if row['epoch']==epoch)==counts
        assert Counter(row['phase'] for row in batches if row['epoch']==epoch)==batch_counts
    initial_validation = json.loads((training/'initial_validation.json').read_text())
    assert initial_validation['optimizer_steps']==0 and initial_validation['weights_unchanged']
    stopper = ValidationEarlyStopping(8,1e-4);stopper.observe(initial_validation['validation_xy_mse'])
    for row in metrics:
        triggered = stopper.observe(row['validation_xy_mse'])
        assert triggered==row['early_stopping_triggered']
        assert stopper.bad_epochs==row['early_stopping_bad_epochs']
    original = torch.load(root/'inputs/initial_airimu.ckpt',map_location='cpu',weights_only=True)['model_state_dict']
    newest = torch.load(training/'newest.ckpt',map_location='cpu',weights_only=True)
    best = torch.load(training/'best_model.ckpt',map_location='cpu',weights_only=True)
    selected = min(metrics,key=lambda row:row['validation_xy_mse'])
    assert newest['epoch']==epochs and best['epoch']==selected['epoch']
    assert math.isclose(best['best_loss'],selected['validation_xy_mse'],rel_tol=1e-12)
    metadata = newest['ekf_metadata']
    assert metadata['initial_checkpoint_sha256']==file_sha256(root/'inputs/initial_airimu.ckpt')
    assert metadata['configuration_definition']=='full_encoder_cnn_gru1_gru2'
    assert metadata['gradient_clip'] is None and metadata['gradient_clipping']==plan['gradient_clipping']
    assert {group['name']:group['lr'] for group in metadata['optimizer_groups']}==plan['optimizer_learning_rates']
    assert metadata['early_stopping']=={'patience':8,'min_delta':1e-4,'metric':'validation_xy_mse','mode':'min'}
    prefixes = ('cnn.','gru1.','gru2.','acccov_decoder.','gyrocov_decoder.','acc_bias_gain.','gyro_bias_gain.')
    frozen = [name for name in original if not name.startswith(prefixes)]
    changed_counts = {}
    for label,checkpoint in (('newest',newest),('best',best)):
        current = checkpoint['model_state_dict'];assert current.keys()==original.keys()
        assert all(bool(torch.isfinite(value).all()) for value in current.values())
        changed = [name for name,value in current.items() if not torch.equal(value,original[name].to(value))]
        assert not set(changed).intersection(frozen)
        changed_counts[label] = {prefix:sum(name.startswith(prefix) for name in changed) for prefix in prefixes}
        assert all(changed_counts[label][prefix]>0 for prefix in prefixes)
    train_batches = [row for row in batches if row['phase']=='train']
    assert len(train_batches)==epochs*batch_counts['train']
    for row in train_batches:
        assert row['gradient_clip_scale'] is None
        for name,limit in (('cnn',1.),('gru1',1.),('gru2',5.),('heads_and_bias_gains',1.)):
            assert row['gradient_clipping_groups'][name]['norm_after']<=limit+2e-5
        for name in ('cnn','gru1','gru2'):
            assert row['gradient_norms_by_group'][name]['total']>0
        assert all(value['relative_l2']<=.01 for value in row['parameter_updates_by_group'].values())
    for name,expected in plan['input_sha256'].items():assert file_sha256(root/name)==expected,name
    verification = {'configuration':'C_full_encoder','epochs_completed':epochs,'early_stopped':summary['early_stopped'],
        'best_epoch':best['epoch'],'best_validation_xy_rmse':math.sqrt(selected['validation_xy_mse']),
        'initial_validation_xy_rmse':initial_validation['validation_xy_rmse'],
        'optimizer_steps':len(train_batches),'frozen_tensors_checked':len(frozen),'frozen_tensors_changed':[],
        'correction_heads_unchanged':True,'all_weights_finite':True,'updated_tensors_by_module':changed_counts,
        'gradient_clipping':metadata['gradient_clipping'],'gradient_summary':summarize_batches(train_batches),
        'checkpoint_selection_split':'validation','held_out_eval_used':False,
        'selected_checkpoint':str(training/'best_model.ckpt'),
        'selected_checkpoint_sha256':file_sha256(training/'best_model.ckpt')}
    (root/'training_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    fields = ('epoch','train_seconds','validation_seconds','checkpoint_seconds','epoch_seconds','train_xy_rmse','validation_xy_rmse')
    with (root/'epoch_timing.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        writer.writerows({key:row[key] for key in fields} for row in metrics)
    (root/'TRAINING_COMPLETE').write_text(json.dumps({key:value for key,value in verification.items() if key!='gradient_summary'},indent=2)+'\n')
    plan.update(status='C training verified; checkpoint selected using validation; held-out evaluation pending',
        epochs_completed=epochs,best_epoch=best['epoch'],selected_c_checkpoint_sha256=verification['selected_checkpoint_sha256'])
    (root/'run_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({key:value for key,value in verification.items() if key!='gradient_summary'},indent=2),flush=True)


if __name__=='__main__':main()
