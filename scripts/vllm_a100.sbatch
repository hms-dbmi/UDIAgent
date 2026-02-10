#!/bin/bash
#SBATCH --job-name=vllm-a100
#SBATCH --partition=kempner
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --time=0-14:00
#SBATCH --output=logs/vllm-a100-%j.out
#SBATCH --error=logs/vllm-a100-%j.err

MODEL="${1:-/n/netscratch/mzitnik_lab/Lab/dlange/data/vis-v2/data/hidive/UDI-VIS-Beta-v2-Llama-3.1-8B}"
PORT="${2:-8080}"

module load ncf/1.0.0-fasrc01
module load miniconda3/py39_4.11.0-linux_x64-ncf
conda activate udienv

echo "Running on host: $(hostname)"
echo "Model: ${MODEL}"
echo "Port: ${PORT}"

vllm serve "${MODEL}" --gpu-memory-utilization 0.85 --port "${PORT}" --host 127.0.0.1
