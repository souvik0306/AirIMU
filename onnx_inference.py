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


def run(config: str, onnx_path: str, outfile: str, seqlen: int, whole: bool) -> None:
    """Run ONNX inference on sequences listed in ``config``.

    Parameters
    ----------
    config : str
        Path to the model configuration.
    onnx_path : str
        Location of the exported ONNX model.
    outfile : str
        Destination pickle file for corrected IMU data.
    seqlen : int
        Sequence length used when ``whole`` is ``False``. The IMU stream will be
        processed in non-overlapping chunks of this size, mimicking the behaviour
        of ``inference.py``.
    whole : bool
        If ``True`` the entire sequence is processed in one shot.
    """

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
            dt_all = time[1:] - time[:-1]
            acc_all = acc_all[:-1].astype(np.float32)
            gyro_all = gyro_all[:-1].astype(np.float32)

            if whole:
                splits = [(acc_all, gyro_all, dt_all)]
            else:
                splits = []
                start = 0
                total = acc_all.shape[0]
                while start < total:
                    end = min(start + seqlen, total)
                    splits.append((acc_all[start:end], gyro_all[start:end], dt_all[start:end]))
                    start = end

            corr_acc_all, corr_gyro_all = [], []
            corrected_acc_all, corrected_gyro_all, dt_all_trim = [], [], []

            for acc, gyro, dt in splits:
                acc_np = acc[np.newaxis, ...]
                gyro_np = gyro[np.newaxis, ...]
                corr_acc, corr_gyro = sess.run(None, {"acc": acc_np, "gyro": gyro_np})
                corrected_acc = acc_np[:, interval:, :] + corr_acc
                corrected_gyro = gyro_np[:, interval:, :] + corr_gyro
                dt_trimmed = dt[interval:, None]

                corr_acc_all.append(torch.from_numpy(corr_acc[0]).to(dtype=torch.float64))
                corr_gyro_all.append(torch.from_numpy(corr_gyro[0]).to(dtype=torch.float64))
                corrected_acc_all.append(torch.from_numpy(corrected_acc[0]).to(dtype=torch.float64))
                corrected_gyro_all.append(torch.from_numpy(corrected_gyro[0]).to(dtype=torch.float64))
                dt_all_trim.append(torch.tensor(dt_trimmed, dtype=torch.float64))

            results[seq] = {
                "correction_acc": torch.cat(corr_acc_all, dim=0),
                "correction_gyro": torch.cat(corr_gyro_all, dim=0),
                "corrected_acc": torch.cat(corrected_acc_all, dim=0),
                "corrected_gyro": torch.cat(corrected_gyro_all, dim=0),
                "dt": torch.cat(dt_all_trim, dim=0),
            }

    with open(outfile, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AirIMU ONNX model on IMU CSVs")
    parser.add_argument("--config", default="configs/exp/EuRoC/codenet.conf",
                        help="Model config file")
    parser.add_argument("--onnx", default="codenet.onnx", help="ONNX model file")
    parser.add_argument("--out", default="net_output.pickle", help="Output pickle")
    parser.add_argument("--seqlen", type=int, default=1000,
                        help="IMU sequence length for chunked inference")
    parser.add_argument("--whole", action="store_true",
                        help="Process entire sequences in one shot")
    args = parser.parse_args()

    run(args.config, args.onnx, args.out, args.seqlen, args.whole)


if __name__ == "__main__":
    main()
