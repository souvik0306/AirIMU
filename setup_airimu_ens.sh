#!/usr/bin/env bash

# Set up AirIMU directly on a regular Linux/ENS machine (no PBS or Apptainer).
# Automatically finds Python 3.10–3.12 or installs local Python 3.11 via uv.
# Extracts ZIP files using Python; no system unzip package is needed.
# Environment overrides:
#   PYTHON_BIN=python3.11  VENV_DIR=/path/to/venv  SKIP_DOWNLOADS=1

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_EXPLICIT="${PYTHON_BIN:+1}"
PYTHON_EXPLICIT="${PYTHON_EXPLICIT:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
SKIP_DOWNLOADS="${SKIP_DOWNLOADS:-0}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

TEMP_DIR=""
cleanup() {
    if [[ -n "$TEMP_DIR" ]]; then rm -rf -- "$TEMP_DIR"; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'echo "ERROR: Setup failed at line $LINENO; see the error above." >&2' ERR

supported_python() {
    "$1" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' >/dev/null 2>&1
}
check_python_version() {
    supported_python "$1" || fail "$1 requires Python 3.10–3.12. Choose a fresh VENV_DIR=.venv311 if the existing environment is incompatible."
}
download() {
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --connect-timeout 30 --output "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --timeout=30 --output-document="$2" "$1"
    else
        fail "Install curl or wget before running setup."
    fi
}
ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
    elif [[ -x "$SCRIPT_DIR/.airimu-tools/uv" ]]; then
        UV_BIN="$SCRIPT_DIR/.airimu-tools/uv"
    else
        echo "Installing uv locally (no root privileges needed)..."
        TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/airimu-setup-XXXXXX")"
        download "https://astral.sh/uv/install.sh" "$TEMP_DIR/install.sh"
        UV_INSTALL_DIR="$SCRIPT_DIR/.airimu-tools" UV_NO_MODIFY_PATH=1 sh "$TEMP_DIR/install.sh"
        cleanup
        TEMP_DIR=""
        UV_BIN="$SCRIPT_DIR/.airimu-tools/uv"
        [[ -x "$UV_BIN" ]] || fail "uv installation failed."
    fi
}

if [[ -e "$VENV_DIR" ]]; then
    [[ -x "$VENV_DIR/bin/python" ]] || fail "$VENV_DIR is not a working venv. Choose a fresh VENV_DIR=.venv311."
    check_python_version "$VENV_DIR/bin/python"
fi
if [[ "$PYTHON_EXPLICIT" == 1 ]]; then
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "$PYTHON_BIN was not found."
    check_python_version "$PYTHON_BIN"
elif [[ -x "$VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
else
    PYTHON_BIN=""
    for candidate in python3.11 python3.12 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && supported_python "$candidate"; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
    if [[ -z "$PYTHON_BIN" ]]; then
        ensure_uv
        echo "Installing managed Python 3.11; system Python is unchanged..."
        "$UV_BIN" python install 3.11
        PYTHON_BIN="$("$UV_BIN" python find --managed-python 3.11)"
    fi
fi

echo "AirIMU repository: $SCRIPT_DIR"
echo "Virtual environment: $VENV_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating Python virtual environment..."
    if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
        echo "Creating the environment with uv (Python venv support is missing)..."
        ensure_uv
        "$UV_BIN" venv --seed --python "$PYTHON_BIN" "$VENV_DIR"
    fi
else
    echo "Using the existing virtual environment."
fi

VENV_PYTHON="$VENV_DIR/bin/python"
check_python_version "$VENV_PYTHON"
if ! "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
    echo "Bootstrapping pip in the virtual environment..."
    "$VENV_PYTHON" -m ensurepip --upgrade || fail \
        "Could not bootstrap pip. Install the venv package matching $PYTHON_BIN, then rerun this setup."
fi
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install -r requirements.txt

download_and_extract() {
    local target_dir="$1"
    local url="$2"

    if [[ -d "$SCRIPT_DIR/$target_dir" ]]; then
        echo "Already present: $target_dir"
        return
    fi

    TEMP_DIR="$(mktemp -d "$SCRIPT_DIR/.airimu-download-XXXXXX")"
    local archive="$TEMP_DIR/archive.zip"
    echo "Downloading $target_dir..."
    download "$url" "$archive" || fail "Download failed: $url"
    echo "Extracting $target_dir..."
    "$VENV_PYTHON" - "$archive" "$TEMP_DIR/extracted" "$target_dir" <<'PYTHON'
import pathlib
import stat
import sys
import zipfile

archive, destination, target = sys.argv[1:]
root = pathlib.Path(destination).resolve()
with zipfile.ZipFile(archive) as source:
    for entry in source.infolist():
        path = (root / entry.filename).resolve()
        if not path.is_relative_to(root) or stat.S_ISLNK(entry.external_attr >> 16):
            raise SystemExit(f"Unsafe archive entry: {entry.filename}")
    source.extractall(root)
if not (root / target).is_dir():
    raise SystemExit(f"Archive is missing expected directory: {target}")
if target == "AirIMU_EuRoC" and not (root / target / "best_model.ckpt").is_file():
    raise SystemExit("Archive is missing best_model.ckpt")
PYTHON
    mv -- "$TEMP_DIR/extracted/$target_dir" "$SCRIPT_DIR/$target_dir"
    cleanup
    TEMP_DIR=""
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
