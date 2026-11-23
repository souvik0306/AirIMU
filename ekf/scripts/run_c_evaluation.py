"""Run fresh held-out inference for frozen Stage 1/A/B Strong/C selected checkpoints."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from ekf.experiment import file_sha256
from evaluation.compare_eskf_models import compare
from evaluation.plot_eskf import verify_plot_outputs, write_comparison_index


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',required=True)
    args = parser.parse_args();root = Path(args.run_root).resolve()
    plan = json.loads((root/'run_plan.json').read_text())
    verified = json.loads((root/'training_verification.json').read_text())
    assert (root/'TRAINING_COMPLETE').exists() and not verified['held_out_eval_used']
    models = {'Stage1':root/'inputs/initial_airimu.ckpt','A':root/'inputs/A_best_model.ckpt',
              'BStrong':root/'inputs/BStrong_best_model.ckpt','C':root/'training/best_model.ckpt'}
    assert file_sha256(models['C'])==verified['selected_checkpoint_sha256']
    for name,expected in plan['input_sha256'].items():assert file_sha256(root/name)==expected,name
    output = root/'evaluation';output.mkdir(exist_ok=True)
    wrapper = Path(__file__).resolve().parents[2]/'jobs/utils/container_python.py'
    evaluations = {}
    plan.update(status='C checkpoint frozen; fresh inference and EKF evaluation on 12 reserved flights',evaluations_completed=[])
    (root/'run_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for label,checkpoint in models.items():
        destination = output/label;destination.mkdir(exist_ok=True);evaluations[label] = destination
        command = [sys.executable,'-s','-u',str(wrapper),'-m','evaluation.evaluate_eskf',
            '--config',str(root/'configs/stage2.conf'),'--ckpt',str(checkpoint),
            '--dataset-config',str(root/'configs/dataset.conf'),
            '--sensor-metadata-root',str(root/'configs/initial_sensor_biases'),
            '--output',str(destination),'--split','eval','--expected-flights','12','--full-sequence','--plots',
            '--device','cuda:0','--ekf-device','cpu','--cpu-threads','1','--height-updates',
            '--calibration-window','.05','--orientation-std','.001',
            '--calibration-max-step-deg','45','--calibration-max-rate','20']
        print(f'Fresh inference/evaluation starting: {label}; log={destination/"job.log"}',flush=True)
        with (destination/'job.log').open('w') as handle:
            subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True)
        verification = json.loads((destination/'evaluation_verification.json').read_text())
        assert verification['flights_completed']==12 and verification['optimizer_steps']==0
        assert len(verify_plot_outputs(destination))==12
        plan['evaluations_completed'].append(label)
        (root/'run_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        print(f'Fresh inference/evaluation completed: {label}; 12 flights, zero optimizer steps.',flush=True)
    comparison = compare(evaluations,root/'comparison',expected_flights=12,plots=True)
    gallery = write_comparison_index(evaluations,comparison,root/'comparison')
    for name,expected in plan['input_sha256'].items():assert file_sha256(root/name)==expected,name
    assert file_sha256(models['C'])==verified['selected_checkpoint_sha256']
    completed = datetime.now(ZoneInfo('Asia/Singapore')).isoformat()
    plan.update(status='complete; C trained and Stage1/A/BStrong/C evaluated on all 12 reserved flights',
        completed_sgt=completed,comparison=str(root/'comparison/comparison.json'),plot_gallery=str(gallery),
        plot_sets_completed=48,plot_images_completed=240)
    (root/'run_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (root/'COMPLETE').write_text(completed+'\n')
    for label,row in comparison['models'].items():
        print(f"{label}: held_out_eval_xy_rmse={row['aggregate_horizontal_velocity_rmse']:.9g} "
              f"improved_flights_vs_stage1={row['flights_improved_over_Stage1']}/12",flush=True)


if __name__=='__main__':main()
