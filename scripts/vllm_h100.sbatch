#!/bin/bash
#
# Serve a vLLM model on an H100 GPU.
# H100s are faster than A100s but sometimes have a longer queue.
# If the queue is long, consider using vllm_a100.sbatch instead.
#
# Usage:
#   sbatch scripts/vllm_h100.sbatch [PORT] [MODEL]
#
# Arguments:
#   PORT   - Port for the vLLM server (default: 8080)
#   MODEL  - Model path or HuggingFace ID (default: UDI-VIS-Beta-v2-Llama-3.1-8B)
#
# Examples:
#   sbatch scripts/vllm_h100.sbatch
#   sbatch scripts/vllm_h100.sbatch 8081
#   sbatch scripts/vllm_h100.sbatch 8081 HIDIVE/UDI-VIS-Beta-v2-Llama-3.1-8B
#
#SBATCH --job-name=vllm-h100
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --time=0-03:00
#SBATCH --output=logs/vllm-h100-%j.out
#SBATCH --error=logs/vllm-h100-%j.err

PORT="${1:-8080}"
MODEL="${2:-/n/netscratch/mzitnik_lab/Lab/dlange/data/vis-v2/data/hidive/UDI-VIS-Beta-v2-Llama-3.1-8B}"

module load ncf/1.0.0-fasrc01
module load miniconda3/py39_4.11.0-linux_x64-ncf
conda activate udienv

echo "Running on host: $(hostname)"
echo "Port: ${PORT}"
echo "Model: ${MODEL}"

vllm serve "${MODEL}" --gpu-memory-utilization 0.85 --port "${PORT}" --host 127.0.0.1
