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
# Submit from the repo root, one job per episode:
#   sbatch slurm/yolo.sbatch /scratch/$USER/data/EP_3
# Writes /scratch/$USER/results/<episode>_yolo_classifier_8_26s.json
# no -u: conda's activate scripts reference unset variables
set -eo pipefail
SCRIPT=yolo_classifier_8_26s
DATA=$(realpath "${1:?pass one episode folder, e.g. /scratch/\$USER/data/EP_3}")
[ -d "$DATA" ] || { echo "no such folder: $DATA" >&2; exit 1; }
[ -d "$SLURM_SUBMIT_DIR/scripts" ] || { echo "submit from the repo root" >&2; exit 1; }
RESULTS=/scratch/$USER/results
mkdir -p "$RESULTS"
OUT=$RESULTS/$(basename "$DATA")_$SCRIPT.json

module load anaconda3
source activate thermal_project

# Weights (yolov8x*.pt, yolo26x*.pt) are resolved relative to the working dir
cd "$SLURM_SUBMIT_DIR/scripts"
echo "$(hostname): $DATA -> $OUT"
python $SCRIPT.py "$DATA" --out "$OUT"
