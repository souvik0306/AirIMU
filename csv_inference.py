import argparse
import os
import pickle
import numpy as np
import torch
from pyhocon import ConfigFactory

from model import net_dict


def load_imu_csv(csv_path: str):
    """Load IMU data from a EuRoC-style CSV file."""
    imu = np.loadtxt(csv_path, delimiter=",", dtype=float)
    time = imu[:, 0] / 1e9  # ns -> s
    gyro = imu[:, 1:4]
    acc = imu[:, 4:7]
    return time, acc, gyro


def run(config: str, ckpt: str, device: str, outfile: str):
    """Run inference on sequences specified in the config."""
    # Load configuration and disable ground-truth orientation
    conf = ConfigFactory.parse_file(config)
    conf.train.gtrot = False
    conf.train.device = device

    # Build network and load weights
    net = net_dict[conf.train.network](conf.train).to(device).double().eval()
    checkpoint = torch.load(ckpt, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])

    dataset_conf = conf.dataset.inference
    results = {}

    # Iterate over all sequences defined in the dataset config
    for data_conf in dataset_conf.data_list:
        root = data_conf["data_root"]
        for seq in data_conf["data_drive"]:
            csv_path = os.path.join(root, seq, "mav0", "imu0", "data.csv")
            time, acc_all, gyro_all = load_imu_csv(csv_path)
            dt = time[1:] - time[:-1]

            # Drop last sample to match dt length
            acc = acc_all[:-1]
            gyro = gyro_all[:-1]

            # Prepare batch tensors
            acc_t = torch.tensor(acc, dtype=torch.float64).unsqueeze(0)
            gyro_t = torch.tensor(gyro, dtype=torch.float64).unsqueeze(0)
            batch = {"acc": acc_t, "gyro": gyro_t}

            # Forward pass
            with torch.no_grad():
                out = net.inference(batch)
                corrected_acc = acc_t[:, net.interval:, :] + out["correction_acc"]
                corrected_gyro = gyro_t[:, net.interval:, :] + out["correction_gyro"]

            # Align dt with corrected signals
            dt_t = torch.tensor(dt[net.interval:], dtype=torch.float64)[:, None]

            results[seq] = {
                "correction_acc": out["correction_acc"][0],
                "correction_gyro": out["correction_gyro"][0],
                "corrected_acc": corrected_acc[0],
                "corrected_gyro": corrected_gyro[0],
                "dt": dt_t,
            }

    # Save results for downstream evaluation/integration
    with open(outfile, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AirIMU on IMU CSVs listed in the config.")
    parser.add_argument("--config", default="configs/exp/EuRoC/codenet.conf", help="Model config file")
    parser.add_argument("--ckpt", default="experiments/EuRoC/codenet/ckpt/best_model.ckpt", help="Model checkpoint")
    parser.add_argument("--device", default="cpu", help="Device to run on")
    parser.add_argument("--out", default="net_output.pickle", help="Output pickle path")
    args = parser.parse_args()

    run(args.config, args.ckpt, args.device, args.out)
