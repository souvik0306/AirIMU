#!/usr/bin/env bash

# Set up AirIMU directly on a regular Linux/ENS machine (no PBS or Apptainer).
# Environment overrides:
#   PYTHON_BIN=python3.11  VENV_DIR=/path/to/venv  SKIP_DOWNLOADS=1

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
SKIP_DOWNLOADS="${SKIP_DOWNLOADS:-0}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "$PYTHON_BIN was not found."
if [[ "$SKIP_DOWNLOADS" != "1" ]]; then
    command -v unzip >/dev/null 2>&1 || fail "unzip is required to extract the datasets."
    if command -v curl >/dev/null 2>&1; then
        download() { curl --fail --location --retry 3 --output "$2" "$1"; }
    elif command -v wget >/dev/null 2>&1; then
        download() { wget --tries=3 --output-document="$2" "$1"; }
    else
        fail "Install curl or wget before running this setup."
    fi
fi

echo "AirIMU repository: $SCRIPT_DIR"
echo "Virtual environment: $VENV_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating Python virtual environment..."
    "$PYTHON_BIN" -m venv "$VENV_DIR" || fail \
        "Could not create a venv. Install the Python venv package (for example, python3-venv)."
else
    echo "Using the existing virtual environment."
fi

VENV_PYTHON="$VENV_DIR/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install -r requirements.txt

download_and_extract() {
    local target_dir="$1"
    local url="$2"

    if [[ -d "$SCRIPT_DIR/$target_dir" ]]; then
        echo "Already present: $target_dir"
        return
    fi

    local archive
    archive="$(mktemp --tmpdir airimu-download-XXXXXX.zip)"
    echo "Downloading $target_dir..."
    if ! download "$url" "$archive"; then
        rm -f -- "$archive"
        fail "Download failed: $url"
    fi
    if ! unzip -q "$archive" -d "$SCRIPT_DIR"; then
        rm -f -- "$archive"
        fail "Could not extract $target_dir"
    fi
    rm -f -- "$archive"
    [[ -d "$SCRIPT_DIR/$target_dir" ]] || fail \
        "The archive did not create the expected directory: $target_dir"
}

if [[ "$SKIP_DOWNLOADS" == "1" ]]; then
    echo "Skipping datasets and pretrained weights (SKIP_DOWNLOADS=1)."
else
    download_and_extract \
        "T-Lab_31st_July_dataset" \
        "https://github.com/souvik0306/AirIMU/releases/download/31st_July_fast_agile/T-Lab_31st_July_dataset.zip"
    download_and_extract \
        "T-Lab_28th_July_dataset" \
        "https://github.com/souvik0306/AirIMU/releases/download/28th_july_hover/T-Lab_28th_July_dataset.zip"
    download_and_extract \
        "T-Lab_10th_September_dataset" \
        "https://github.com/souvik0306/AirIMU/releases/download/10th_Sept/T-Lab_10th_September_dataset.zip"
    download_and_extract \
        "T-Lab_14th_October_dataset" \
        "https://github.com/souvik0306/AirIMU/releases/download/15th_oct/T-Lab_14th_October_dataset.zip"
    download_and_extract \
        "T-Lab_15th_October_dataset" \
        "https://github.com/souvik0306/AirIMU/releases/download/15th_oct/T-Lab_15th_October_dataset.zip"
    download_and_extract \
        "AirIMU_EuRoC" \
        "https://github.com/Air-IO/Air-IO/releases/download/AirIMU/AirIMU_EuRoC.zip"
fi

echo
"$VENV_PYTHON" - <<'PYTHON'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA is unavailable; fine-tuning will run on CPU unless PyTorch/CUDA is fixed.")
PYTHON

if [[ -f "$SCRIPT_DIR/AirIMU_EuRoC/best_model.ckpt" ]]; then
    echo "Pretrained checkpoint: OK"
elif [[ "$SKIP_DOWNLOADS" != "1" ]]; then
    fail "AirIMU_EuRoC/best_model.ckpt was not found after extraction."
else
    echo "WARNING: AirIMU_EuRoC/best_model.ckpt is absent."
fi

echo
echo "AirIMU setup is complete."
echo "Activate it with: source \"$VENV_DIR/bin/activate\""
echo "Run fine-tuning with: ./airimu_finetune.sh all"
