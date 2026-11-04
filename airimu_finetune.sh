#!/usr/bin/env bash

# Run from the AirIMU repository without PBS or Apptainer.
set -e

cd "$(dirname "$0")"
source .venv/bin/activate

DEVICE=cuda:0


# -----------------------------------------------------------------------------
# Stage 1: fine-tune IMU corrections
# Comment out this command if Stage 1 is already complete.
# Add --load_ckpt to resume from the newest Stage 1 checkpoint.
# Add --log to disable Weights & Biases logging.
# -----------------------------------------------------------------------------

python3 -u train_motion_finetune.py --config configs/exp/TLab/finetune_motion_body.conf --device "$DEVICE"


# -----------------------------------------------------------------------------
# Stage 2: fine-tune uncertainty and covariance
# Comment out this command if Stage 2 is already complete.
# Add --load_ckpt to resume from the newest Stage 2 checkpoint.
# Add --log to disable Weights & Biases logging.
# -----------------------------------------------------------------------------

python3 -u train_motion_finetune.py --config configs/exp/TLab/finetune_motion_body_stage2.conf --device "$DEVICE"


# -----------------------------------------------------------------------------
# Run inference with the best Stage 2 checkpoint
# -----------------------------------------------------------------------------

python3 -u inference.py --config configs/exp/TLab/finetune_motion_body_stage2.conf --device cpu


# -----------------------------------------------------------------------------
# Evaluate the inference output
# -----------------------------------------------------------------------------

python3 -u evaluation/evaluate_state.py --device cpu --dataconf configs/datasets/TLab/tlab_body.conf --exp experiments/tlab_stage2_3rd_Nov/finetune_motion_body_stage2/


# -----------------------------------------------------------------------------
# Export the best Stage 2 model to ONNX and PyTorch
# -----------------------------------------------------------------------------

python3 -u export_onnx_fp32.py --config configs/exp/TLab/finetune_motion_body_stage2.conf --ckpt experiments/tlab_stage2_3rd_Nov/finetune_motion_body_stage2/ckpt/best_model.ckpt --onnx experiments/tlab_stage2_3rd_Nov/finetune_motion_body_stage2/airimu_cpu_fp32_finetuned.onnx --torch experiments/tlab_stage2_3rd_Nov/finetune_motion_body_stage2/airimu_cpu_fp32.pt --num-imu-frames 200


echo "AirIMU fine-tuning, evaluation, and export complete."
