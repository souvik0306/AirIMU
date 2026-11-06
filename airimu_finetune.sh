#!/usr/bin/env bash

# Run from the AirIMU repository without PBS or Apptainer.
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
STAGE1_CONFIG=configs/exp/TLab/finetune_motion_body.conf
STAGE2_CONFIG=configs/exp/TLab/finetune_motion_body_stage2.conf

# A caller can explicitly provide RUN_ROOT. New runs get an SGT timestamp plus
# the process ID, so even simultaneous launches are unique.
if [[ -z "${RUN_ROOT:-}" ]]; then
    RUN_ID="$(TZ=Asia/Singapore date +%Y%m%d_%H%M%S_SGT)_pid$$"
    RUN_ROOT="experiments/$RUN_ID/finetune"
fi
STAGE1_EXP="$RUN_ROOT/stage1"
STAGE2_EXP="$RUN_ROOT/stage2"

echo "Run root: $RUN_ROOT"


# -----------------------------------------------------------------------------
# Stage 1: fine-tune IMU corrections
# Comment out this command if Stage 1 is already complete.
# Add --load_ckpt to resume from the newest Stage 1 checkpoint.
# Add --log to disable Weights & Biases logging.
# -----------------------------------------------------------------------------

python3 -u train_motion_finetune.py \
    --config "$STAGE1_CONFIG" \
    --exp_dir "$STAGE1_EXP" \
    --device "$DEVICE"

if [[ ! -f "$STAGE1_EXP/ckpt/best_model.ckpt" ]]; then
    echo "ERROR: Stage 1 completed without best_model.ckpt" >&2
    exit 3
fi


# -----------------------------------------------------------------------------
# Stage 2: fine-tune uncertainty and covariance
# Comment out this command if Stage 2 is already complete.
# Add --load_ckpt to resume from the newest Stage 2 checkpoint.
# Add --log to disable Weights & Biases logging.
# -----------------------------------------------------------------------------

python3 -u train_motion_finetune.py \
    --config "$STAGE2_CONFIG" \
    --exp_dir "$STAGE2_EXP" \
    --pretrained_ckpt "$STAGE1_EXP/ckpt/best_model.ckpt" \
    --device "$DEVICE"

if [[ ! -f "$STAGE2_EXP/ckpt/best_model.ckpt" ]]; then
    echo "ERROR: Stage 2 completed without an eligible best_model.ckpt" >&2
    exit 4
fi


# -----------------------------------------------------------------------------
# Run inference with the best Stage 2 checkpoint
# -----------------------------------------------------------------------------

python3 -u inference.py \
    --config "$STAGE2_CONFIG" \
    --exp_dir "$STAGE2_EXP" \
    --device "$DEVICE"


# -----------------------------------------------------------------------------
# Evaluate the inference output
# -----------------------------------------------------------------------------

python3 -u evaluation/evaluate_state.py \
    --device cpu \
    --dataconf configs/datasets/TLab/tlab_body.conf \
    --exp "$STAGE2_EXP" \
    --savedir "$STAGE2_EXP/evaluation_results"


# -----------------------------------------------------------------------------
# Export the best Stage 2 model to ONNX and PyTorch
# -----------------------------------------------------------------------------

python3 -u export_onnx_fp32.py \
    --config "$STAGE2_CONFIG" \
    --ckpt "$STAGE2_EXP/ckpt/best_model.ckpt" \
    --onnx "$STAGE2_EXP/airimu_cpu_fp32_finetuned.onnx" \
    --torch "$STAGE2_EXP/airimu_cpu_fp32.pt" \
    --num-imu-frames 200


echo "AirIMU fine-tuning, evaluation, and export complete."
