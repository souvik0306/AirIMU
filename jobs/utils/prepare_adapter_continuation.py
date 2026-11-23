"""Freeze a separate ten-epoch continuation of the selected epoch-15 adapter."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[2];root=Path(args.run_root).resolve()
    previous=repo/'experiments/20261122_223523_SGT_BStrong_residual_adapter'
    checkpoint=previous/'training/best_model.ckpt'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    expected='25e68ecd19cc10fdb31594c7c201af22e68fab6a9cf5974d9f2a54e297d53e96'
    assert digest(checkpoint)==expected
    root.mkdir();(root/'inputs').mkdir();(root/'provenance').mkdir()
    shutil.copytree(previous/'configs',root/'configs')
    shutil.copy2(checkpoint,root/'inputs/adapter_initial.ckpt')
    shutil.copy2(previous/'inputs/BStrong_epoch50.ckpt',root/'inputs/BStrong_epoch50.ckpt')
    shutil.copy2(previous/'training/metrics.jsonl',root/'inputs/epochs_1_to_15_metrics.jsonl')
    shutil.copy2(previous/'run_plan.json',root/'provenance/previous_run_plan.json')
    files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=repo).decode().split('\0')
    files=sorted({name for name in files if name and (repo/name).is_file() and not name.startswith(('experiments/','data/','.git/'))
                 and Path(name).suffix in ('.py','.conf','.pbs','.sh','.md','.txt','.json')})
    for name in files:
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
    for cmd,name in ((['git','diff','--binary','HEAD'],'submitted_tracked_changes.patch'),
                     (['git','status','--short'],'submitted_git_status.txt'),(['git','rev-parse','HEAD'],'git_HEAD.txt')):
        (root/'provenance'/name).write_bytes(subprocess.check_output(cmd,cwd=repo))
    plan={'status':'prepared; local resume checks required before submission',
        'created_sgt':datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
        'configuration':'Adapter','previous_run':str(previous),'source_snapshot':str(root/'source'),
        'resume_checkpoint':str(checkpoint),'resume_checkpoint_sha256':expected,
        'resume_epoch':15,'epochs':10,'maximum_total_epochs':25,'restore_adam':True,'restored_adam_steps':315,
        'optimizer':{'raw_cov_adapter':{'lr':1e-4,'gradient_clip':1.}},'adapter_weight':1e-3,
        'early_stopping':{'patience':5,'min_delta':1e-4,'metric':'validation_xy_mse','checkpoint_selection':'minimum validation RMSE, including epoch 15'},
        'split_counts':{'train':82,'validation':23,'test':12},'gyro_weight':1e-4,'height_updates':True,
        'BStrong_parent_completely_frozen':True,'post_training':'verify then fresh validation/eval and velocity/uncertainty plots'}
    paths=[p for folder in ('source','configs','inputs','provenance') for p in (root/folder).rglob('*') if p.is_file()]
    plan['input_sha256']={str(p.relative_to(root)):digest(p) for p in paths}
    (root/'run_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(root,flush=True)


if __name__=='__main__':main()
