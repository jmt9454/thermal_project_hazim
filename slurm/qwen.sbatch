#!/bin/bash
#SBATCH --job-name=qwen_detect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
# Usage: sbatch slurm/qwen.sbatch /scratch/$USER/data [extra qwen_classifier.py args]
set -euo pipefail
DATA=${1:?pass the data dir (parent of ep1, ep2, ...)}
shift

module load anaconda3
source activate thermal_project

# Keep the HF model cache off the home-dir quota
export HF_HOME=/scratch/$USER/hf_cache

cd "$SLURM_SUBMIT_DIR/scripts"
python qwen_classifier.py "$DATA" "$@"
