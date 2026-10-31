#!/bin/bash
# Example: downscale LUCIE CO2 experiment 0 (first 100 timesteps).
# Submit:  sbatch scripts/examples/lucie_co2.sh
#SBATCH --job-name=lucie_co2
#SBATCH --partition=<PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --mem=100000
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1

OUT=<OUTPUT_DIR>/lucie_co2_0
mkdir -p "$OUT"
DATASET=lucie \
INPUT=<LUCIE_DATA>/LUCIE_co2_sensitivity_experiments_combined.nc \
CO2_EXP=0 \
MAX_SAMPLES=100 \
OUTPUT="$OUT" \
bash <DEEPRISK_ROOT>/scripts/run_pipeline.sh
