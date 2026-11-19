#!/usr/bin/env bash

# Run the joint EKF PBS workflow directly on a Linux/ENS machine.
# Run setup_airimu_ens.sh first, then: ./airimu_joint_ekf.sh
# Set AIRIMU_CKPT to the Stage 2 checkpoint for covariance-only fine-tuning.
# Overrides match airimu_joint_ekf.pbs; also supports VENV_DIR and DEVICE.
# Example: EKF_EPOCHS=10 DEVICE=cuda:0 ./airimu_joint_ekf.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
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

device=${DEVICE:-cuda:0}
config=${EKF_CONFIG:-configs/exp/TLab/4th_Nov_Scratch_stage2.conf}
checkpoint=${AIRIMU_CKPT:?Set AIRIMU_CKPT to the Stage 2 checkpoint}
[[ -f "$checkpoint" ]] || { echo "ERROR: checkpoint not found: $checkpoint" >&2; exit 2; }
dataset_config=${EKF_DATASET_CONFIG:-configs/datasets/TLab/tlab_body.conf}
sensor_metadata_root=${SENSOR_METADATA_ROOT:-initial_sensor_biases}
epochs=${EKF_EPOCHS:-30}
learning_rate=${EKF_LR:-1e-4}
cpu_threads=${EKF_CPU_THREADS:-1}
replay_workers=${EKF_REPLAY_WORKERS:-4}
flight_batch_size=${EKF_FLIGHT_BATCH_SIZE:-$((replay_workers > 0 ? replay_workers : 1))}
feature_cache_mb=${EKF_FEATURE_CACHE_MB:-512}
replay_backend=${EKF_REPLAY_BACKEND:-compiled}
default_ekf_device=$device
if [[ "$replay_workers" != "0" ]]; then default_ekf_device=cpu; fi
ekf_device=${EKF_DEVICE:-$default_ekf_device}
run_fixed_baseline=${EKF_FIXED_BASELINE:-1}

for required_file in "$config" "$dataset_config"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file not found: $required_file" >&2
        exit 2
    fi
done
if [[ ! -d "$sensor_metadata_root" ]]; then
    echo "ERROR: sensor metadata directory not found: $sensor_metadata_root" >&2
    exit 2
fi

run_timestamp=$(TZ=Asia/Singapore date +%Y%m%d_%H%M%S_SGT)
run_id="${run_timestamp}_pid$$"
run_root=${EKF_RUN_ROOT:-experiments/${run_id}/joint_ekf}
training_output="$run_root/training"
learned_evaluation="$run_root/evaluation_learned"
fixed_evaluation="$run_root/evaluation_fixed"

echo "Joint differentiable ESKF training"
echo "Run root: $run_root"
echo "Initialization: $checkpoint"
echo "Dataset config: $dataset_config"
echo "Sensor metadata: $sensor_metadata_root"
echo "Epochs: $epochs; learning rate: $learning_rate"
echo "ESKF device: $ekf_device (AirIMU network: $device)"
echo "Replay workers: $replay_workers; threads/worker: 1; flights/Adam step: $flight_batch_size"
echo "Replay backend: $replay_backend"
echo "Working directory: $(pwd)"
echo "Compute node: $(hostname)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true
fi

python3 -u train_eskf.py \
--config "$config" \
--ckpt "$checkpoint" \
--dataset-config "$dataset_config" \
--sensor-metadata-root "$sensor_metadata_root" \
--output "$training_output" \
--device "$device" \
--ekf-device "$ekf_device" \
--epochs "$epochs" \
--lr "$learning_rate" \
--cpu-threads "$cpu_threads" \
--replay-workers "$replay_workers" \
--replay-backend "$replay_backend" \
--flight-batch-size "$flight_batch_size" \
--feature-cache-mb "$feature_cache_mb"

if [[ ! -f "$training_output/best_model.ckpt" ]]; then
    echo "ERROR: ESKF training completed without best_model.ckpt" >&2
    exit 3
fi

echo "Training finished. Evaluating learned covariance on the test split."
python3 -u -m evaluation.evaluate_eskf \
--config "$config" \
--ckpt "$training_output/best_model.ckpt" \
--dataset-config "$dataset_config" \
--sensor-metadata-root "$sensor_metadata_root" \
--output "$learned_evaluation" \
--split test \
--device "$device" \
--ekf-device "$ekf_device" \
--cpu-threads "$cpu_threads"

if [[ "$run_fixed_baseline" == "1" ]]; then
    echo "Evaluating the fixed-noise EKF baseline on the same test split."
        python3 -u -m evaluation.evaluate_eskf \
    --dataset-config "$dataset_config" \
    --sensor-metadata-root "$sensor_metadata_root" \
    --output "$fixed_evaluation" \
    --split test \
    --device "$device" \
    --ekf-device "$ekf_device" \
    --cpu-threads "$cpu_threads"
fi

echo "Joint ESKF training and evaluation finished: $run_root"
