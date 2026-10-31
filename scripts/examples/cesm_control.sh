#!/bin/bash
# Example: downscale the CESM SSTcontrol run (all timesteps).
# Submit:  sbatch scripts/examples/cesm_control.sh
#SBATCH --job-name=cesm_ctrl
#SBATCH --partition=<PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --mem=100000
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1

OUT=<CESM_DATA>/outputs/cesm_control
mkdir -p "$OUT"
DATASET=cesm \
INPUT=<CESM_DATA>/SSTcontrol_sigma_levels.nc \
OUTPUT="$OUT" \
bash <DEEPRISK_ROOT>/scripts/run_pipeline.sh
