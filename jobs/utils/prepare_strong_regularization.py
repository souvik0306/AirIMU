"""Freeze original B Strong, identical fresh adapter initialization and existing controls."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[2];root=Path(args.run_root).resolve()
    root.mkdir()
    subprocess.run([sys.executable,str(repo/'jobs/utils/prepare_residual_adapter.py'),'--run-root',str(root)],check=True)
    old15=repo/'experiments/20261122_223523_SGT_BStrong_residual_adapter'
    old25=repo/'experiments/20261123_103117_SGT_adapter_resume_epoch15_to25'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    copies={'adapter_initial.ckpt':old15/'inputs/adapter_initial.ckpt',
            'weak_epoch15.ckpt':old15/'training/best_model.ckpt','weak_epoch25.ckpt':old25/'training/best_model.ckpt'}
    for name,path in copies.items():shutil.copy2(path,root/'inputs'/name)
    plan=json.loads((root/'run_plan.json').read_text())
    plan.update(experiment='stronger_adjustment_penalty_and_SO3_reference_filter',epochs=50,
        orientation_smoothing_window=.05,early_stopping={'patience':5,'min_delta':1e-4,'metric':'validation_xy_mse'},
        milestones=[15,25,35,50],fresh_optimizer=True,seed=0,
        initialization='bit-identical original zero-output adapter and frozen B Strong; fresh Adam state',
        initialization_source=str(copies['adapter_initial.ckpt']),
        control_evaluations={'BStrong':str(old15/'evaluation/validation/BStrong'),
                             'Weak15':str(old15/'evaluation/validation/Adapter'),
                             'Weak25':str(old25/'evaluation/validation/Adapter')},
        checkpoint_selection='exact minimum validation horizontal velocity MSE including epoch 0; per-flight regressions reported separately',
        post_training='validation comparisons/milestones reuse old controls; reserved eval only once for the selected checkpoint',
        reference_uncertainty='unfiltered endpoint-noise variance floor plus conservative linearized correlated-endpoint bound; systematic errors not eliminated',
        attribution='two settings changed together; per-checkpoint paired raw/filtered diagnostics isolate reference change, not separate training effects')
    plan['adapter']['delta_l2_weight']=1e-2
    for name in copies:plan['input_sha256']['inputs/'+name]=sha(root/'inputs'/name)
    for label,directory in plan['control_evaluations'].items():
        for filename in ('metrics.json','evaluation_metadata.json','evaluation_verification.json','COMPLETE'):
            path=Path(directory)/filename;plan['input_sha256'][str(path)]=sha(path)
    for name in ('ekf/data.py','ekf/eskf.py','ekf/compiled.py','model/code.py'):
        assert sha(root/'source'/name)==sha(old15/'source'/name),name
    plan['unchanged_model_and_EKF_paths']=['ekf/data.py','ekf/eskf.py','ekf/compiled.py','model/code.py']
    (root/'run_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(root,flush=True)


if __name__=='__main__':main()
