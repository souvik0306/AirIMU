# AirIMU : Learning Uncertainty Propagation for Inertial Odometry

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-yellow.svg)](./LICENSE)
[![YouTube](https://img.shields.io/badge/YouTube-b31b1b?style=flat&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=fTX1u-e7wtU)
[![arXiv](https://img.shields.io/badge/arXiv-AirIMU-orange.svg)](https://arxiv.org/abs/2310.04874)
[![githubio](https://img.shields.io/badge/-homepage-blue?logo=Github&color=FF0000)](https://airimu.github.io/)

![AirIMU](./doc/model.png)

## 📢 Latest News

- [2025-02-01] Introducing Our New Project!`<br>`
  🚀 [**AirIO : Learning Inertial Odometry with Enhanced IMU Feature Observability**](https://github.com/Air-IO/Air-IO)`<br>`

```
AirIO achieves up to 86.6% performance boost over SOTA methods:

- ✅ Tailored specifically for drones
- ✅ No external sensors or control inputs required
- ✅ Generalizes to unseen trajectories
- ✅ Explicitly encodes UAV attitude and predicts velocity in body-frame representation
```

## Installation

This work is based on pypose. Follow the instruction and install the newest realase of pypose:
https://github.com/pypose/pypose

## Dataset

> **Note**: Remember to reset the `data_root` in `configs/datasets/${DATASET}/${DATASET}.conf`.

Download the Euroc dataset from:
https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

Download the TUM VI dataset from:
https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset

Download the KITTI dataset from:
https://www.cvlibs.net/datasets/kitti/

Download the SubT-MRS dataset: [Download](https://github.com/sleepycan/AirIMU/releases/download/subt_dataset/SubT-MRS-Dataset.zip).   🎈You can find more information about SubT-MRS datasets from the links [here](SubT-MRS-DATASET.md).

## Pretrained Model

> **Note**: You can download our trained ckpt here.

[KITTI](https://github.com/sleepycan/AirIMU/releases/download/pretrained_model/KITTI_odom_model.zip)

[EuRoC](https://github.com/sleepycan/AirIMU/releases/download/pretrained_model_euroc/EuRoCWholeaug.zip)

## Train

Easy way to start the training using the exisiting configuration.

> **Note**: You can also create your own configuration file for different datasets and set the parameters accordingly.

> **Training Mode**: Set `propcov = false` in your training config if you wish to train **only for IMU corrections** without covariance learning. Set `propcov = true` to enable uncertainty propagation and train the model to predict both corrections and per-sample covariance estimates.

```
python train.py --config configs/exp/EuRoC/codenet.conf
```

More specific option:

```
usage: train.py [-h] [--config CONFIG] [--device DEVICE] [--load_ckpt] [--log]

optional arguments:
  -h, --help       show this help message and exit
  --config CONFIG  config file path
  --device DEVICE  cuda or cpu, Default is cuda:0
  --load_ckpt      If True, try to load the newest.ckpt in the exp_dir specificed in our config file.
  --log            if True, save the meta data with wandb, Default is True
```

## Evaluation

To evaluate the model and generate network inference file net_output.pickle, run the following command:

```
python inference.py --config configs/exp/EuRoC/codenet.conf
```

### Inference on a standalone IMU CSV

If only the raw IMU measurements are available (timestamps, gyroscope and accelerometer) you can run the network directly on the CSV files listed in the dataset configuration without any ground‑truth orientation:

```
python inference_offline.py --config configs/exp/EuRoC/codenet.conf \
                        --ckpt experiments/EuRoC/codenet/ckpt/best_model.ckpt
```

The script reads each sequence's `mav0/imu0/data.csv` based on `data_root` and `data_drive` entries in the config and writes `net_output.pickle` keyed by sequence name containing corrected IMU data (`corrected_acc`, `corrected_gyro` and `dt`).

### Exporting to ONNX

Convert a training checkpoint to both ONNX and a plain PyTorch state dictionary:

```
python export_onnx.py --config configs/exp/EuRoC/codenet.conf \
                      --ckpt experiments/EuRoC/codenet/ckpt/best_model.ckpt \
                      --onnx airimu_euroc.onnx --torch airimu_euroc.pt
```

By default the ONNX model is exported in **float32** precision to ensure
compatibility with the CPU execution provider of ONNX Runtime. If you need a
float64 graph, pass `--fp64` (note that some runtimes lack double-precision
`Conv` support).

### Offline inference with ONNX

Run an exported ONNX model directly on the IMU CSV files listed in the config:

```
python inference_onnx.py --config configs/exp/EuRoC/codenet.conf \
                         --onnx airimu_euroc.onnx --seqlen 1000
```
Pass `--whole` to process an entire sequence in a single pass; otherwise the IMU
stream is chunked into windows of length `--seqlen`, mirroring the behaviour of
`inference.py`.

The resulting `net_output.pickle` stores all arrays as PyTorch tensors, matching the
format produced by `inference_offline.py` so that downstream evaluation scripts that call `.cpu()` work without modification.

<br>

You can use the evaluation tool to assess your model performance with net_output.pickle， run the following command.

> **Note**: Make sure to replace path/to/net_output_directory with the directory path where your network output pickle file is stored.

```
python evaluation/evaluate_state.py --dataconf configs/datasets/${DATASET}/${DATASET}.conf --exp path/to/net_output_directory
```

<br>
More specific option for the evaluation tool:

```
usage: evaluation/evaluate_state.py [-h] [--dataconf] [--device] [--exp] [--seqlen] [--savedir] [--usegtrot] [--mask]

optional arguments:
  -h, --help       show this help message and exit
  --config         config file path
  --device         cuda or cpu, Default is cuda:0
  --exp            the directory path where your network output pickle file is stored
  --seqlen         the length of the integration sequence
  --savedir        the save diretory for the evaluation results, default path is  "./result/loss_result"
  --usegtrot       use ground truth rotation for gravity compensation, default is true
  --mask           mask the segments if needed. 
```

For covariance-only AirIMU training through a differentiable 15-state EKF with
Vicon cutoff, see [the ESKF integration guide](doc/differentiable_eskf.md).
PBS launchers, local shell launchers, setup scripts and scheduler logs are
grouped under [`jobs/`](jobs/README.md). For example,
`./jobs/submit.sh airimu_ekf_benchmark.pbs` submits a two-epoch benchmark and
places its scheduler output in `jobs/logs/ekf/`. EKF-specific tests, preparation
tools and benchmarks are grouped under [`ekf/`](ekf/README.md).
The new entry points are `train_eskf.py` and
`python -m evaluation.evaluate_eskf`; existing training commands are unchanged.
For the training bottleneck, performance changes and measured A40 benchmark,
see [why EKF training became faster](doc/ekf_training_performance.md).

### Cite Our Work

Thanks for using our work. You can cite it as:

```bib
@article{qiu2023airimu,
            title={AirIMU: Learning Uncertainty Propagation for Inertial Odometry}, 
            author={Yuheng Qiu and Chen Wang and Can Xu and Yutian Chen and Xunfei Zhou and Youjie Xia and Sebastian Scherer},
            year={2023},
            eprint={2310.04874},
            archivePrefix={arXiv},
            primaryClass={cs.RO}
      }
```
