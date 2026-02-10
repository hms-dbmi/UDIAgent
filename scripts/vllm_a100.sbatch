#!/bin/bash
#
# Serve a vLLM model on an A100 GPU.
# A100s are slightly slower than H100s but typically have a shorter queue.
# If speed is a priority and the queue is short, consider using vllm_h100.sbatch instead.
#
# Usage:
#   sbatch scripts/vllm_a100.sbatch [PORT] [MODEL]
#
# Arguments:
#   PORT   - Port for the vLLM server (default: 8080)
#   MODEL  - Model path or HuggingFace ID (default: UDI-VIS-Beta-v2-Llama-3.1-8B)
#
# Examples:
#   sbatch scripts/vllm_a100.sbatch
#   sbatch scripts/vllm_a100.sbatch 8081
#   sbatch scripts/vllm_a100.sbatch 8081 HIDIVE/UDI-VIS-Beta-v2-Llama-3.1-8B
#
#SBATCH --job-name=vllm-a100
#SBATCH --partition=kempner
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --time=0-03:00
#SBATCH --output=logs/vllm-a100-%j.out
#SBATCH --error=logs/vllm-a100-%j.err

PORT="${1:-8080}"
MODEL="${2:-/n/netscratch/mzitnik_lab/Lab/dlange/data/vis-v2/data/hidive/UDI-VIS-Beta-v2-Llama-3.1-8B}"

module load ncf/1.0.0-fasrc01
module load miniconda3/py39_4.11.0-linux_x64-ncf
source $(conda info --base)/etc/profile.d/conda.sh
conda activate udienv

echo "Running on host: $(hostname)"
echo "Port: ${PORT}"
echo "Model: ${MODEL}"

vllm serve "${MODEL}" --gpu-memory-utilization 0.85 --port "${PORT}" --host 127.0.0.1
