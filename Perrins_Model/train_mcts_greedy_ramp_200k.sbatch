#!/bin/bash
# Slurm batch job for Capstone RL training:
# ramp enemy mix from mcts@100:1.0,greedy@25:0.0 -> mcts@100:0.8,greedy@25:0.2

#SBATCH --job-name=catan_mcts_greedy_200k
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=compute

set -euo pipefail

source ~/.bashrc
micromamba activate catan311

cd "$HOME/catan_ai"
mkdir -p logs

python -u -m capstone_agent.run_simulation \
  --games 200000 \
  --train \
  --enemy-smooth-mix \
  --enemy-mix-start "mcts@100:1.0,greedy@25:0.0" \
  --enemy-mix-end "mcts@100:0.8,greedy@25:0.2" \
  --enemy-mix-games 200000 \
  --save-every-games 1000 \
  --save capstone_agent/models/capstone_model.pt \
  --save-placement-model capstone_agent/models/placement_model.pt \
  --map-mode fixed
