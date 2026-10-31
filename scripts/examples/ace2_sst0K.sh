#!/bin/bash
# Example: downscale the ACE2 era5_sst+0K experiment (all timesteps).
# Submit:  sbatch scripts/examples/ace2_sst0K.sh
#SBATCH --job-name=ace_sst0K
#SBATCH --partition=<PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --mem=100000
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1

OUT=<OUTPUT_DIR>/ace_sst_0K
mkdir -p "$OUT"
DATASET=ace \
INPUT=era5_sst+0K \
ACE_ROOT=<ACE_ROOT> \
OUTPUT="$OUT" \
bash <DEEPRISK_ROOT>/scripts/run_pipeline.sh
