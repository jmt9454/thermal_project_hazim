#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x_%j.out
# Submit from the repo root: sbatch slurm/setup_env.sbatch
set -eo pipefail
module load anaconda3
export PIP_NO_CACHE_DIR=1 HF_HOME=/scratch/$USER/hf_cache

conda create -y -n thermal_project python=3.11
source activate thermal_project
# plain pip (not conda env create) so the log shows each download
pip install --prefer-binary vllm ultralytics==8.4.148 bitsandbytes accelerate \
    qwen-vl-utils opencv-python-headless pillow polars tqdm

cd "$SLURM_SUBMIT_DIR/scripts"
python -c "from ultralytics import YOLO; [YOLO(w) for w in ['yolov8x-pose.pt','yolo26x-pose.pt','yolov8x.pt','yolo26x.pt']]"
hf download Qwen/Qwen3.5-9B
python -c "import torch, vllm, ultralytics, cv2; print('env OK', torch.__version__, vllm.__version__)"