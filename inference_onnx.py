#!/usr/bin/env python3
"""Run offline inference using an exported ONNX AirIMU model."""
import argparse
import os
import pickle
import numpy as np
import torch
import onnxruntime as ort
from pyhocon import ConfigFactory
from model import net_dict


def load_imu_csv(csv_path: str):
    imu = np.loadtxt(csv_path, delimiter=",", dtype=float)
    time = imu[:, 0] / 1e9
    gyro = imu[:, 1:4]
    acc = imu[:, 4:7]
    return time, acc, gyro


def run(config: str, onnx_path: str, outfile: str) -> None:
    conf = ConfigFactory.parse_file(config)
    conf.train.gtrot = False
    dataset_conf = conf.dataset.inference

    # Instantiate network to obtain interval length
    net = net_dict[conf.train.network](conf.train)
    interval = getattr(net, "interval", 0)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    results = {}

    for data_conf in dataset_conf.data_list:
        root = data_conf["data_root"]
        for seq in data_conf["data_drive"]:
            csv_path = os.path.join(root, seq, "mav0", "imu0", "data.csv")
            time, acc_all, gyro_all = load_imu_csv(csv_path)
            dt = time[1:] - time[:-1]
            acc = acc_all[:-1].astype(np.float32)
            gyro = gyro_all[:-1].astype(np.float32)

            acc_np = acc[np.newaxis, ...]
            gyro_np = gyro[np.newaxis, ...]
            corr_acc, corr_gyro = sess.run(None, {"acc": acc_np, "gyro": gyro_np})
            corrected_acc = acc_np[:, interval:, :] + corr_acc
            corrected_gyro = gyro_np[:, interval:, :] + corr_gyro
            dt_trimmed = dt[interval:, None]

            results[seq] = {
                "correction_acc": torch.from_numpy(corr_acc[0]).to(dtype=torch.float64),
                "correction_gyro": torch.from_numpy(corr_gyro[0]).to(dtype=torch.float64),
                "corrected_acc": torch.from_numpy(corrected_acc[0]).to(dtype=torch.float64),
                "corrected_gyro": torch.from_numpy(corrected_gyro[0]).to(dtype=torch.float64),
                "dt": torch.tensor(dt_trimmed, dtype=torch.float64),
            }

    with open(outfile, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AirIMU ONNX model on IMU CSVs")
    parser.add_argument("--config", default="configs/exp/EuRoC/codenet.conf",
                        help="Model config file")
    parser.add_argument("--onnx", default="codenet.onnx", help="ONNX model file")
    parser.add_argument("--out", default="net_output_onnx.pickle", help="Output pickle")
    args = parser.parse_args()

    run(args.config, args.onnx, args.out)


if __name__ == "__main__":
    main()