"""Verify one complete matched experiment and compare once both runs finish."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import torch

from ekf.experiment import file_sha256
from ekf.gradients import summarize_batches


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True)
    parser.add_argument('--configuration',choices=('A','B'),required=True)
    args=parser.parse_args()
    root=Path(args.run_root).resolve()
    run=root/f'configuration_{args.configuration}'
    training=run/'training'
    plan=json.loads((root/'run_plan.json').read_text())
    epochs=plan['epochs']
    metrics=[json.loads(line) for line in (training/'metrics.jsonl').read_text().splitlines()]
    flights=[json.loads(line) for line in (training/'flights.jsonl').read_text().splitlines()]
    batches=[json.loads(line) for line in (training/'batches.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in metrics]==list(range(1,epochs+1))
    for epoch in range(1,epochs+1):
        assert Counter(row['phase'] for row in flights if row['epoch']==epoch)=={'train':82,'validation':23}
        assert Counter(row['phase'] for row in batches if row['epoch']==epoch)=={'train':21,'validation':6}
    initial=torch.load(root/'inputs/initial_airimu.ckpt',map_location='cpu',weights_only=True)['model_state_dict']
    newest=torch.load(training/'newest.ckpt',map_location='cpu',weights_only=True)
    best=torch.load(training/'best_model.ckpt',map_location='cpu',weights_only=True)
    expected=min(metrics,key=lambda row:row['validation_xy_mse'])
    assert newest['epoch']==epochs and best['epoch']==expected['epoch']
    assert math.isclose(best['best_loss'],expected['validation_xy_mse'],rel_tol=1e-12)
    assert newest['ekf_metadata']['configuration']==args.configuration
    assert newest['ekf_metadata']['gradient_clip']==1.
    assert newest['ekf_metadata']['gyro_reference_quality_policy']=='vicon_so3_quality_v1'
    assert newest['ekf_metadata']['initial_checkpoint_sha256']==plan['input_sha256']['inputs/initial_airimu.ckpt']
    prefixes=['acccov_decoder.','gyrocov_decoder.','acc_bias_gain.','gyro_bias_gain.']
    if args.configuration=='B':prefixes.append('gru2.')
    current=newest['model_state_dict']
    assert current.keys()==initial.keys()
    changed=[key for key,value in current.items() if not torch.equal(value,initial[key].to(value))]
    frozen=[key for key in current if not any(key.startswith(prefix) for prefix in prefixes)]
    assert not set(frozen).intersection(changed)
    updated={prefix:sum(key.startswith(prefix) for key in changed) for prefix in prefixes}
    assert all(updated.values())
    assert all(bool(torch.isfinite(value).all()) for value in current.values())
    initial_validation=json.loads((training/'initial_validation.json').read_text())
    assert initial_validation['optimizer_steps']==0 and initial_validation['weights_unchanged']
    train_batches=[row for row in batches if row['phase']=='train']
    assert len(train_batches)==epochs*21
    assert all(row['gradient_norm_after_clip']<=1.00002 for row in train_batches)
    for name,expected_hash in plan['input_sha256'].items():
        assert file_sha256(root/name)==expected_hash,name
    gradient_summary=summarize_batches(train_batches)
    verification={'configuration':args.configuration,'epochs_completed':epochs,
        'best_epoch':best['epoch'],'best_validation_xy_rmse':math.sqrt(expected['validation_xy_mse']),
        'initial_validation_xy_rmse':initial_validation['validation_xy_rmse'],
        'flight_replays':len(flights),'optimizer_steps':len(train_batches),
        'frozen_tensors_checked':len(frozen),'frozen_tensors_changed':[],
        'updated_tensors_by_module':updated,'all_weights_finite':True,
        'gradient_clip':1.,'gradient_summary':gradient_summary,
        'initial_checkpoint_sha256':plan['input_sha256']['inputs/initial_airimu.ckpt']}
    (run/'completion_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    fields=['epoch','train_seconds','validation_seconds','checkpoint_seconds','epoch_seconds',
            'train_xy_rmse','validation_xy_rmse']
    with (run/'epoch_timing.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows({key:row[key] for key in fields} for row in metrics)
    completed=datetime.now(ZoneInfo('Asia/Singapore')).isoformat()
    (run/'COMPLETE').write_text(completed+'\n')
    print(json.dumps(verification,indent=2),flush=True)
    # The last completed job performs the comparison; the first never waits
    # for another training job while holding a GPU allocation.
    with (root/'.comparison.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ready=[name for name in ('A','B') if (root/f'configuration_{name}/COMPLETE').exists()]
        plan=json.loads((root/'run_plan.json').read_text())
        plan['completed_configurations']=ready
        plan['status']='one configuration complete; awaiting matched run'
        if len(ready)==2:
            comparison=root/'comparison/comparison.json'
            subprocess.run([sys.executable,'-m','ekf.scripts.compare_training_runs',
                str(root/'configuration_A/training'),str(root/'configuration_B/training'),
                '--require-matched-ab','--expected-epochs',str(epochs),'--plots','--output',str(comparison)],check=True)
            (root/'comparison/COMPLETE').write_text(completed+'\n')
            plan.update(status='complete; both configurations verified and matched comparison generated',
                        completed_sgt=completed,comparison=str(comparison))
            print('Matched A/B comparison ready:',comparison,flush=True)
        else:
            print('Matched comparison will run when the other configuration finishes.',flush=True)
        (root/'run_plan.json').write_text(json.dumps(plan,indent=2)+'\n')


if __name__=='__main__':main()
