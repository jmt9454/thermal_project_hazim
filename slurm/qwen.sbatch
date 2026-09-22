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
# Submit from the repo root:
#   sbatch slurm/qwen.sbatch /scratch/$USER/data/EP_3 [extra qwen_batch_classifier.py args]
# no -u: conda's activate scripts reference unset variables
set -eo pipefail
DATA=$(realpath "${1:?pass the data dir (parent of the episode folders)}")
shift

module load anaconda3
source activate thermal_project

# Keep the HF model cache off the home-dir quota
export HF_HOME=/scratch/$USER/hf_cache

cd "$SLURM_SUBMIT_DIR/scripts"
# bf16 is fastest on an A100; later args (e.g. --bits 4) override it
python qwen_batch_classifier.py "$DATA" --bits 16 "$@"
