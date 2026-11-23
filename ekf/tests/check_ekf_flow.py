"""Run from the repository root to check the complete EKF pipeline on CPU."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil

import torch
from pyhocon import ConfigFactory
from model import net_dict

repo = Path.cwd()
configuration = os.environ.get('EKF_FLOW_CONFIGURATION','B')
python_entry = [sys.executable, '-s', str(repo/'jobs/utils/container_python.py')]
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")
with tempfile.TemporaryDirectory(prefix="airimu_ekf_flow_") as directory:
    root = Path(directory)
    data_root = root / "T-Lab_31st_July_dataset"
    for number in range(1, 7):
        flight = data_root / "RAW" / f"flight_{number}"
        flight.mkdir(parents=True)
        with (flight / "imu.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow("time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z".split(","))
            writer.writerows((1000+k*.5, .12, -.04, 9.80665, .005, -.008, .01) for k in range(61))
        with (flight / "data.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow("time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,quat_w,quat_x,quat_y,quat_z".split(","))
            writer.writerows((1000+k*.5, .1*k, -.01*k, 0., .2, -.02, 0., 1., 0., 0., 0.) for k in range(61))
    config = root / "dataset.conf"
    config.write_text("\n".join(
        f'{split}: {{mode: evaluate, gravity: 9.80665, data_list: [{{name: TLab, '
        f'window_size: 20, step_size: 20, data_root: "{data_root}", '
        f'data_drive: [{", ".join(f"RAW/flight_{n}" for n in numbers)}]}}]}}'
        for split, numbers in (("train", (1, 2, 3, 4)), ("test", (5,)),
                               ("eval", (6,)), ("inference", (6,)))))
    network_config = root / "network.conf"
    network_config.write_text(
        f'include "{repo}/configs/exp/TLab/4th_Nov_Scratch_stage2.conf"\n'
        f'dataset {{ include "{config}" }}\n')
    metadata = root / "metadata" / "2026-07-31"
    metadata.mkdir(parents=True)
    with (metadata / "initial_sensor_bias_RAW.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow("bag_file,ulg_file,group,flight_number,cutoff_rosbag_s,cutoff_ros_epoch_s,bias_samples,accel_bias_x_mps2,accel_bias_y_mps2,accel_bias_z_mps2,gyro_bias_x_radps,gyro_bias_y_radps,gyro_bias_z_radps".split(","))
        writer.writerows((f"flight_{number}.bag",f"log_{number}.ulg","RAW",number,2.,1002.,100,0.,0.,0.,0.,0.,0.) for number in range(1, 7))
    output = root / "trained"
    checkpoint = root / "synthetic_stage2.ckpt"
    conf = ConfigFactory.parse_file(str(network_config))
    network = net_dict[conf.train.network](conf.train)
    torch.save({"model_state_dict": network.state_dict()}, checkpoint)
    common = ["--config", str(network_config), "--dataset-config", str(config), "--sensor-metadata-root", str(root/"metadata"), "--device", "cpu", "--ekf-device", "cpu", "--cpu-threads", "1", "--calibration-window", "2."]
    with (root / "train.log").open("w") as handle:
        result = subprocess.run([*python_entry,"train_eskf.py","--replay-workers","4","--flight-batch-size","4", "--allow-reduced-split", '--evaluate-initial',
            '--configuration',configuration,'--gru2-lr','3e-6','--gru1-lr','1e-6','--cnn-lr','3e-7',
            '--gradient-clipping','separate','--early-stopping-patience','8','--early-stopping-min-delta','1e-4',
            '--require-encoder-gradients',
            '--gru2-gradient-clip','5','--head-gradient-clip','1','--max-relative-update','.01',
            '--loss-growth-factor','2','--loss-growth-patience','2',
            *common,"--ckpt",str(checkpoint),"--output",str(output),"--epochs","1"],stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode:
        print((root/"train.log").read_text()[-10000:]); raise SystemExit(result.returncode)
    flights = [json.loads(line) for line in (output/"flights.jsonl").read_text().splitlines()]
    epoch = json.loads((output/"metrics.jsonl").read_text())
    assert len(flights)==5 and {row["phase"] for row in flights}=={"train","validation"}
    assert len({row["worker_pid"] for row in flights if row["phase"]=="train"})==4
    for row in flights:
        for name in ("acc_cov","gyro_cov"):
            assert all(len(row["covariance"][name][stat])==3 for stat in ("mean","min","max"))
    assert epoch["train_covariance"]["samples"]==220
    assert epoch["validation_covariance"]["samples"]==55
    assert json.loads((output/"eskf_metadata.json").read_text())["replay_workers"] == 4
    assert epoch["optimizer_steps"] == 1
    initial=json.loads((output/'initial_validation.json').read_text())
    assert initial['optimizer_steps']==0 and initial['weights_unchanged']
    batches=[json.loads(line) for line in (output/'batches.jsonl').read_text().splitlines()]
    assert len(batches)==2 and epoch['train_gradients']['batches']==1
    assert epoch['train_gradients']['gradient_norm_velocity']['mean']>0
    assert epoch['train_gradients']['clipping_groups']['gru2']['norm_after']['max']<=5.00001
    assert epoch['train_gradients']['clipping_groups']['heads_and_bias_gains']['norm_after']['max']<=1.00001
    assert epoch['train_gradients']['parameter_updates_by_group']['gru2']['relative_l2']['max']<.01
    if configuration=='C':
        for name in ('cnn','gru1'):
            assert epoch['train_gradients']['parameter_updates_by_group'][name]['l2']['min']>0
            assert epoch['train_gradients']['by_group'][name]['total']['min']>0
        before = torch.load(checkpoint,weights_only=True)['model_state_dict']
        after = torch.load(output/'newest.ckpt',weights_only=True)['model_state_dict']
        for name,value in before.items():
            if name.startswith(('accdecoder.','gyrodecoder.')):assert torch.equal(value,after[name])
    assert all(key in batches[0] for key in ('gradient_norm_before_clip','gradient_norm_velocity',
                                           'gradient_norm_gyro_calibration','gradient_norm_gyro_calibration_unweighted'))
    assert (output/"best_model.ckpt").is_file() and (output/"newest.ckpt").is_file()
    if configuration=='C':
        verified_root = root/'C_finish_fixture';(verified_root/'inputs').mkdir(parents=True)
        shutil.copy2(checkpoint,verified_root/'inputs/initial_airimu.ckpt')
        shutil.copytree(output,verified_root/'training')
        c_metadata = json.loads((output/'eskf_metadata.json').read_text())
        (verified_root/'run_plan.json').write_text(json.dumps({'gradient_clipping':c_metadata['gradient_clipping'],
            'optimizer_learning_rates':{group['name']:group['lr'] for group in c_metadata['optimizer_groups']},'input_sha256':{}}))
        result = subprocess.run([*python_entry,'-m','ekf.scripts.finish_c_training','--run-root',str(verified_root),
            '--allow-reduced-split'],capture_output=True,text=True)
        if result.returncode:print(result.stdout,result.stderr);raise SystemExit(result.returncode)
        verified = json.loads((verified_root/'training_verification.json').read_text())
        assert verified['correction_heads_unchanged'] and verified['epochs_completed']==1
    inference_output = root / "inference_ekf"
    state_output = root / "evaluation_results_ekf"
    for name, command in (
        ("inference", ["inference.py", "--config", str(network_config), "--load",
                       str(output/"best_model.ckpt"), "--exp_dir", str(inference_output),
                       "--device", "cpu", "--window_size", "20", "--step_size", "20"]),
        ("state_eval", ["evaluation/evaluate_state.py", "--device", "cpu", "--dataconf",
                        str(config), "--exp", str(inference_output), "--savedir", str(state_output),
                        "--seqlen", "20"]),
    ):
        with (root/f"{name}.log").open("w") as handle:
            result = subprocess.run([*python_entry, *command], stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            print((root/f"{name}.log").read_text()[-10000:]); raise SystemExit(result.returncode)
    assert (inference_output/"net_output.pickle").stat().st_size > 0
    assert len(json.loads((state_output/"loss_result.json").read_text())) == 1
    assert list(state_output.rglob("*.png"))
    eval_output = root / "evaluated"
    with (root / "eval.log").open("w") as handle:
        result = subprocess.run([*python_entry,"-m","evaluation.evaluate_eskf",*common,"--ckpt",str(output/"best_model.ckpt"),"--output",str(eval_output),"--split","eval",'--full-sequence'],stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode:
        print((root/"eval.log").read_text()[-10000:]); raise SystemExit(result.returncode)
    report = json.loads((eval_output/"metrics.json").read_text())
    assert len(report)==1 and "covariance" in report[0] and report[0]["elapsed_seconds"]>0
    import torch
    artifact = torch.load(eval_output/report[0]["trajectory_file"],weights_only=True)
    assert artifact["trajectory"]["covariance"].shape==(61,15,15)
    assert float(artifact['imu_time'][-1])>1027.
    assert (eval_output/report[0]['prediction_file']).exists()
    assert json.loads((eval_output/'evaluation_verification.json').read_text())['weights_and_buffers_unchanged']
    evaluated = {'Stage1':root/'eval_Stage1','A':root/'eval_A','BStrong':root/'eval_BStrong','C':eval_output}
    for label,destination in evaluated.items():
        if label=='C':continue
        selected = checkpoint if label=='Stage1' else output/'best_model.ckpt'
        with (root/f'eval_{label}.log').open('w') as handle:
            result = subprocess.run([*python_entry,'-m','evaluation.evaluate_eskf',*common,
                '--ckpt',str(selected),'--output',str(destination),'--split','eval',
                '--expected-flights','1','--full-sequence','--plots'],stdout=handle,stderr=subprocess.STDOUT)
        if result.returncode:
            print((root/f'eval_{label}.log').read_text()[-10000:]);raise SystemExit(result.returncode)
    comparison = root/'four_model_comparison'
    result = subprocess.run([*python_entry,'-m','evaluation.compare_eskf_models','--evaluations',
        *(f'{label}={path}' for label,path in evaluated.items()),'--expected-flights','1','--output',str(comparison)],capture_output=True,text=True)
    if result.returncode:print(result.stdout,result.stderr);raise SystemExit(result.returncode)
    compared = json.loads((comparison/'comparison.json').read_text())
    assert compared['flight_count']==1 and set(compared['models'])=={'Stage1','A','BStrong','C'}
    assert (comparison/'comparison.png').is_file() and (comparison/'comparison_covariance.csv').is_file()
    from evaluation.plot_eskf import verify_plot_outputs, write_comparison_index
    for destination in evaluated.values():
        assert len(verify_plot_outputs(destination))==1
    gallery = write_comparison_index(evaluated,compared,comparison)
    assert gallery.is_file() and gallery.read_text().count('loading="lazy"')==4
    assert sum(len(list((destination/'plots').rglob('*.png'))) for destination in evaluated.values())==20
    # The final gallery must reject missing deliverables instead of silently omitting a flight plot.
    manifest = verify_plot_outputs(eval_output)
    preview = eval_output/'plots'/next(iter(manifest.values()))['scoring_window']
    temporary_preview = preview.with_suffix('.hidden')
    preview.rename(temporary_preview)
    try:
        try:verify_plot_outputs(eval_output)
        except ValueError as error:assert 'Missing or invalid plot' in str(error)
        else:raise AssertionError('A missing uncertainty plot was accepted.')
    finally:temporary_preview.rename(preview)
    # A same-name flight with different raw samples must never enter a paired comparison.
    prediction = eval_output/report[0]['prediction_file']
    changed = torch.load(prediction,weights_only=True);changed['raw_acceleration'][0,0] += 1.
    torch.save(changed,prediction)
    result = subprocess.run([*python_entry,'-m','evaluation.compare_eskf_models','--evaluations',
        *(f'{label}={path}' for label,path in evaluated.items()),'--expected-flights','1',
        '--no-plots','--output',str(root/'must_reject')],capture_output=True,text=True)
    assert result.returncode and 'Raw IMU samples/timestamps differ' in result.stderr
    print(json.dumps({"flow_check":"passed", "training_flights":4, "replay_workers":4,
                     "epoch":epoch["epoch"], "covariance_samples":epoch["train_covariance"]["samples"],
                     "inference_saved":True, "state_evaluation_and_plots_saved":True,
                     "evaluation_flights":len(report),"configuration":configuration,
                     'full_sequence_inference_samples':61,'four_model_comparison_saved':True,
                     'four_model_per_flight_plot_gallery_saved':True,'plot_images_verified':20,
                     "evaluation_xy_rmse":report[0]["ekf_vs_vicon"]["xy_rmse"]}))
