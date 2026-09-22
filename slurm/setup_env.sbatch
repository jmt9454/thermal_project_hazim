#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x_%j.out
# Builds the thermal_project conda env and downloads all model weights.
# Submit from the repo root:  sbatch slurm/setup_env.sbatch
# Safe to resubmit: an existing env is reused and pip skips what's installed.
# no -u: conda's activate scripts reference unset variables
set -eo pipefail
[ -d "$SLURM_SUBMIT_DIR/scripts" ] || { echo "submit from the repo root" >&2; exit 1; }
# activate by path: activating by name fails on Palmetto
ENV_PATH=$HOME/.conda/envs/thermal_project

module load anaconda3
export LD_PRELOAD=""  # Palmetto FAQ fix for conda errors after loading anaconda3
export PIP_NO_CACHE_DIR=1 HF_HOME=/scratch/$USER/hf_cache

# conda-forge only: the defaults channel needs an interactive ToS accept
if [ -x "$ENV_PATH/bin/python" ]; then
    echo "reusing existing env $ENV_PATH"
else
    conda create -y -p "$ENV_PATH" -c conda-forge --override-channels python=3.11 pip
fi
source activate "$ENV_PATH"
[[ "$(which python)" == "$ENV_PATH"/* ]] || { echo "env not active: $(which python)" >&2; exit 1; }

# plain pip (not conda env create) so the log shows each download
pip install --prefer-binary vllm ultralytics==8.4.148 bitsandbytes accelerate \
    qwen-vl-utils opencv-python-headless pillow polars tqdm

# ultralytics pulls in GUI opencv, which needs libGL; fall back to headless if absent
if ! python -c "import cv2" 2>/dev/null; then
    echo "cv2 import failed, switching to headless opencv"
    pip uninstall -y opencv-python
    pip install --force-reinstall --no-deps opencv-python-headless
fi

# Download weights once so parallel detection jobs don't race to fetch them
cd "$SLURM_SUBMIT_DIR/scripts"
python -c "from ultralytics import YOLO; [YOLO(w) for w in ['yolov8x-pose.pt','yolo26x-pose.pt','yolov8x.pt','yolo26x.pt']]"
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))"

python -c "import torch, vllm, ultralytics, cv2; print('env OK | torch', torch.__version__, '| vllm', vllm.__version__, '| ultralytics', ultralytics.__version__)"
ls -lh *.pt
