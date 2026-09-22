#!/bin/bash
#SBATCH --job-name=yolo_detect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/%x_%j.out
# Submit from the repo root:
#   sbatch slurm/yolo.sbatch /scratch/$USER/data/EP_3 [out.json]
# no -u: conda's activate scripts reference unset variables
set -eo pipefail
DATA=$(realpath "${1:?pass the data dir (parent of the episode folders)}")
OUT=$(realpath -m "${2:-$DATA/yolo_8_26s_detections.json}")

module load anaconda3
source activate thermal_project

# Weights (yolov8x*.pt, yolo26x*.pt) are resolved relative to the working dir
cd "$SLURM_SUBMIT_DIR/scripts"
python yolo_classifier_8_26s.py "$DATA" --out "$OUT"
