"""Run from the repository root to check the complete EKF pipeline on CPU."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

repo = Path.cwd()
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
            writer.writerows((1000+k*.5, .12, -.04, 9.80665, .005, -.008, .01) for k in range(55))
        with (flight / "data.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow("time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,quat_w,quat_x,quat_y,quat_z".split(","))
            writer.writerows((1000+k*.5, .1*k, -.01*k, 0., .2, -.02, 0., 1., 0., 0., 0.) for k in range(55))
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
    checkpoint = repo / "experiments/20261115_184615_SGT_job1558928/joint_ekf/airimu_stage2/ckpt/best_model.ckpt"
    common = ["--config", str(network_config), "--dataset-config", str(config), "--sensor-metadata-root", str(root/"metadata"), "--device", "cpu", "--ekf-device", "cpu", "--cpu-threads", "1"]
    with (root / "train.log").open("w") as handle:
        result = subprocess.run([sys.executable,"-u","train_eskf.py","--replay-workers","4","--flight-batch-size","4",*common,"--ckpt",str(checkpoint),"--output",str(output),"--epochs","1"],stdout=handle,stderr=subprocess.STDOUT)
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
    assert (output/"best_model.ckpt").is_file() and (output/"newest.ckpt").is_file()
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
            result = subprocess.run([sys.executable, "-u", *command], stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            print((root/f"{name}.log").read_text()[-10000:]); raise SystemExit(result.returncode)
    assert (inference_output/"net_output.pickle").stat().st_size > 0
    assert len(json.loads((state_output/"loss_result.json").read_text())) == 1
    assert list(state_output.rglob("*.png"))
    eval_output = root / "evaluated"
    with (root / "eval.log").open("w") as handle:
        result = subprocess.run([sys.executable,"-u","-m","evaluation.evaluate_eskf",*common,"--ckpt",str(output/"best_model.ckpt"),"--output",str(eval_output),"--split","test"],stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode:
        print((root/"eval.log").read_text()[-10000:]); raise SystemExit(result.returncode)
    report = json.loads((eval_output/"metrics.json").read_text())
    assert len(report)==1 and "covariance" in report[0] and report[0]["elapsed_seconds"]>0
    import torch
    artifact = torch.load(eval_output/report[0]["trajectory_file"],weights_only=True)
    assert artifact["trajectory"]["covariance"].shape==(55,15,15)
    print(json.dumps({"flow_check":"passed", "training_flights":4, "replay_workers":4,
                     "epoch":epoch["epoch"], "covariance_samples":epoch["train_covariance"]["samples"],
                     "inference_saved":True, "state_evaluation_and_plots_saved":True,
                     "evaluation_flights":len(report), "evaluation_xy_rmse":report[0]["ekf_vs_vicon"]["xy_rmse"]}))
