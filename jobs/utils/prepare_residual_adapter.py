"""Freeze the B Strong parent, exact split/bias settings and adapter run sources."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[2];root=Path(args.run_root).resolve()
    parent=repo/'experiments/20261121_074452_SGT_stage1_ekf_B_strong_50epochs'
    ckpt=parent/'training/best_model.ckpt'
    expected='13d38c300c860a09d1f02bccfdaf9f14e13876650f9538d6a558eaeb4a64a2da'
    assert digest(ckpt)==expected
    assert not (root/'source').exists() and not (root/'training').exists()
    (root/'inputs').mkdir();shutil.copy2(ckpt,root/'inputs/BStrong_epoch50.ckpt')
    shutil.copytree(parent/'configs',root/'configs')
    config=json.loads((root/'configs/stage2.conf').read_text())
    config['train'].update(raw_cov_adapter=True,gtrot=False)
    (root/'configs/adapter.conf').write_text(json.dumps(config,indent=2)+'\n')
    files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=repo).decode().split('\0')
    files=sorted({name for name in files if name and (repo/name).is_file()
        and not name.startswith(('experiments/','data/','.git/'))
        and Path(name).suffix in ('.py','.conf','.pbs','.sh','.md','.txt','.json')})
    for name in files:
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
    provenance=root/'provenance';provenance.mkdir(exist_ok=True)
    (provenance/'submitted_tracked_changes.patch').write_bytes(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=repo))
    (provenance/'submitted_git_status.txt').write_bytes(subprocess.check_output(['git','status','--short'],cwd=repo))
    source_files=[p for folder in ('source','configs','inputs','source_before_adapter','provenance')
                  if (root/folder).exists() for p in (root/folder).rglob('*') if p.is_file()]
    plan={'created_sgt':datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
        'status':'prepared; local preflight required before PBS submission','configuration':'Adapter',
        'parent_checkpoint':str(ckpt),'parent_checkpoint_sha256':expected,'parent_epoch':50,
        'git_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip(),
        'source_snapshot':str(root/'source'),'pre_change_snapshot':str(root/'source_before_adapter'),
        'epochs':15,'optimizer':{'raw_cov_adapter':{'lr':1e-4,'gradient_clip':1.}},
        'adapter':{'inputs':'frozen GRU2 features, log parent covariance, frozen corrections',
            'hidden_units':32,'parameters':8806,'formula':'Q_new=max(floor,Q_B*exp(delta))',
            'zero_output_initialization':True,'delta_l2_weight':1e-3},
        'split_counts':{'train':82,'validation':23,'test':12},
        'gyro_calibration_weight':1e-4,'height_updates':True,'gtrot':False,
        'checkpoint_selection':'minimum validation MSE including unchanged epoch-0 parent',
        'large_flight_error_threshold_mps':2.,
        'uncertainty_review':'report per-axis inflation ratios, percentiles, variation and quiet/rapid motion; no claim of calibration from velocity alone',
        'post_training':'fresh validation and held-out eval of BStrong and selected adapter; velocity/uncertainty plots',
        'input_sha256':{str(p.relative_to(root)):digest(p) for p in sorted(source_files)}}
    (root/'run_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(root,flush=True)


if __name__=='__main__':main()
