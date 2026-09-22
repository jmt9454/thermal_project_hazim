#!/bin/bash
#SBATCH --job-name=yolo_detect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# Usage: sbatch slurm/yolo.sbatch /scratch/$USER/data [/scratch/$USER/results/yolo.json]
set -euo pipefail
DATA=${1:?pass the data dir (parent of ep1, ep2, ...)}
OUT=${2:-$DATA/yolo_8_26s_detections.json}

module load anaconda3
source activate thermal_project

# Weights (yolov8x*.pt, yolo26x*.pt) are resolved relative to the working dir
cd "$SLURM_SUBMIT_DIR/scripts"
python yolo_classifier_8_26s.py "$DATA" --out "$OUT"
