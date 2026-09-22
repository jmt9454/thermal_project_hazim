#!/bin/bash
#SBATCH --job-name=qwen_detect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gpus=a100:1
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/%x_%j.out
# Submit from the repo root, one job per episode:
#   sbatch slurm/qwen.sbatch /scratch/$USER/data/EP_3 [extra qwen_batch_classifier.py args]
# Writes /scratch/$USER/results/<episode>_qwen_batch_classifier.json
# no -u: conda's activate scripts reference unset variables
set -eo pipefail
SCRIPT=qwen_batch_classifier
DATA=$(realpath "${1:?pass one episode folder, e.g. /scratch/\$USER/data/EP_3}")
shift
[ -d "$DATA" ] || { echo "no such folder: $DATA" >&2; exit 1; }
[ -d "$SLURM_SUBMIT_DIR/scripts" ] || { echo "submit from the repo root" >&2; exit 1; }
RESULTS=/scratch/$USER/results
mkdir -p "$RESULTS"
OUT=$RESULTS/$(basename "$DATA")_$SCRIPT.json

module load anaconda3
source activate thermal_project

# Keep the HF model cache off the home-dir quota
export HF_HOME=/scratch/$USER/hf_cache
# Per-job compile caches so episodes running at once don't write the same files
export VLLM_CACHE_ROOT=${TMPDIR:-/tmp}/vllm_cache
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton_cache

cd "$SLURM_SUBMIT_DIR/scripts"
echo "$(hostname): $DATA -> $OUT"
# bf16 is fastest on an A100; later args (e.g. --bits 4) override these
python $SCRIPT.py "$DATA" --bits 16 --out "$OUT" "$@"
