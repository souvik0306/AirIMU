#!/usr/bin/env bash

# Fresh TLab training run: 4th_Nov_Scratch.
set -e

cd "$(dirname "$0")"
VENV_DIR="${VENV_DIR:-}"
if [[ -z "$VENV_DIR" ]]; then
    if [[ -f .venv311/bin/activate ]]; then
        VENV_DIR=.venv311
    else
        VENV_DIR=.venv
    fi
fi
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: No virtual environment found at $VENV_DIR. Run setup_airimu_ens.sh first, or set VENV_DIR to your environment path." >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"

DEVICE="${DEVICE:-cuda:0}"
STAGE1_CONFIG=configs/exp/TLab/4th_Nov_Scratch.conf
STAGE2_CONFIG=configs/exp/TLab/4th_Nov_Scratch_stage2.conf
STAGE2_EXP=experiments/tlab_scratch_4th_Nov/4th_Nov_Scratch_stage2

# -----------------------------------------------------------------------------
# Stage 1: train IMU corrections from scratch
# No --load_ckpt: always start from random initialization.
# Add --log to either training command to disable Weights & Biases logging.
# Outputs: experiments/tlab_scratch_4th_Nov/4th_Nov_Scratch/
# -----------------------------------------------------------------------------

python3 -u train_motion_finetune.py \
    --config "$STAGE1_CONFIG" \
    --pretrained_ckpt none \
    --device "$DEVICE"

# -----------------------------------------------------------------------------
# Stage 2: train uncertainty and covariance using the best scratch Stage 1 model
# Outputs: experiments/tlab_scratch_4th_Nov/4th_Nov_Scratch_stage2/
# -----------------------------------------------------------------------------

python3 -u train_motion_finetune.py \
    --config "$STAGE2_CONFIG" \
    --device "$DEVICE"

# -----------------------------------------------------------------------------
# Run inference with the best Stage 2 checkpoint
# -----------------------------------------------------------------------------

python3 -u inference.py \
    --config "$STAGE2_CONFIG" \
    --device "$DEVICE"

# -----------------------------------------------------------------------------
# Evaluate inference output and save results alongside the scratch run
# -----------------------------------------------------------------------------

python3 -u evaluation/evaluate_state.py \
    --device cpu \
    --dataconf configs/datasets/TLab/tlab_body.conf \
    --exp "$STAGE2_EXP" \
    --savedir "$STAGE2_EXP/evaluation_results"

# -----------------------------------------------------------------------------
# Export the best Stage 2 model to FP32 ONNX and PyTorch
# -----------------------------------------------------------------------------

python3 -u export_onnx_fp32.py \
    --config "$STAGE2_CONFIG" \
    --ckpt "$STAGE2_EXP/ckpt/best_model.ckpt" \
    --onnx "$STAGE2_EXP/airimu_cpu_fp32_scratch.onnx" \
    --torch "$STAGE2_EXP/airimu_cpu_fp32.pt" \
    --num-imu-frames 200

echo "4th_Nov_Scratch training, covariance training, inference, evaluation, and export complete."
