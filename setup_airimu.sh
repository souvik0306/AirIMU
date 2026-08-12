#!/bin/bash

set -e

cd ~/AirIMU

echo "========================================"
echo "AirIMU setup"
echo "========================================"

echo
echo "1. Loading Apptainer"
module load apptainer/1.4.1

image=/app1/common/singularity-img/vanda/pytorch_2.5_cuda_12.4_unsloth.sif

echo
echo "2. Downloading TLab dataset"

if [ ! -d "T-Lab_31st_July_dataset" ]; then
    wget https://github.com/souvik0306/AirIMU/releases/download/31st_July_fast_agile/T-Lab_31st_July_dataset.zip
    unzip T-Lab_31st_July_dataset.zip
else
    echo "TLab dataset already exists"
fi

echo
echo "3. Downloading pretrained AirIMU EuRoC weights"

if [ ! -d "AirIMU_EuRoC" ]; then
    wget https://github.com/Air-IO/Air-IO/releases/download/AirIMU/AirIMU_EuRoC.zip
    unzip AirIMU_EuRoC.zip
else
    echo "AirIMU EuRoC weights already exist"
fi

echo
echo "4. Installing Python requirements"

apptainer exec -e $image \
    python3 -m pip install --user -r requirements.txt

echo
echo "5. Checking required files"

if [ -f "./AirIMU_EuRoC/best_model.ckpt" ]; then
    echo "Checkpoint found"
else
    echo "ERROR: pretrained checkpoint not found"
    exit 1
fi

if [ -d "./T-Lab_31st_July_dataset" ]; then
    echo "TLab dataset found"
else
    echo "ERROR: TLab dataset not found"
    exit 1
fi

echo
echo "6. Checking PyTorch"

apptainer exec -e $image python3 <<'PYTHON'
import torch
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
PYTHON

echo
echo "========================================"
echo "AirIMU setup complete"
echo "========================================"
