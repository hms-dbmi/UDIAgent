#!/bin/bash
#SBATCH --job-name=vllm-a100
#SBATCH --partition=kempner
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --output=logs/vllm-a100-%j.out
#SBATCH --error=logs/vllm-a100-%j.err

HOURS="${1:-3}"
if [ "$HOURS" -lt 1 ] || [ "$HOURS" -gt 23 ]; then
    echo "Error: hours must be between 1 and 23, got ${HOURS}"
    exit 1
fi
PORT="${2:-8080}"
MODEL="${3:-/n/netscratch/mzitnik_lab/Lab/dlange/data/vis-v2/data/hidive/UDI-VIS-Beta-v2-Llama-3.1-8B}"

scontrol update JobId="$SLURM_JOB_ID" TimeLimit="0-${HOURS}:00:00"

module load ncf/1.0.0-fasrc01
module load miniconda3/py39_4.11.0-linux_x64-ncf
conda activate udienv

echo "Running on host: $(hostname)"
echo "Time limit: ${HOURS} hours"
echo "Port: ${PORT}"
echo "Model: ${MODEL}"

vllm serve "${MODEL}" --gpu-memory-utilization 0.85 --port "${PORT}" --host 127.0.0.1
