#!/bin/bash
# =============================================================================
# DeepRisk downscaling — single parameterized runner (SLURM-submittable)
# =============================================================================
# Replaces the ~90 per-experiment scripts that used to live here. It dispatches
# to the right pipeline based on DATASET and forwards the common options.
#
# Submit with sbatch (redirect logs to an EXISTING dir):
#   mkdir -p <CESM_DATA>/outputs/cesm_control
#   sbatch --output=<CESM_DATA>/outputs/cesm_control/slurm_%j.out \
#          --error=<CESM_DATA>/outputs/cesm_control/slurm_%j.err \
#          --export=ALL,DATASET=cesm,INPUT=<CESM_DATA>/SSTcontrol_sigma_levels.nc,OUTPUT=<CESM_DATA>/outputs/cesm_control \
#          scripts/run_pipeline.sh
#
# Or run directly on a node (no SLURM):
#   DATASET=ace INPUT=era5_sst+0K OUTPUT=<OUTPUT_DIR>/ace_demo MAX_SAMPLES=10 bash scripts/run_pipeline.sh
#
# Environment variables:
#   DATASET       ace | cesm | lucie                                   (required)
#   INPUT         ace : experiment dir name (e.g. era5_sst+0K)
#                 cesm: path to *_sigma_levels.nc
#                 lucie: path to LUCIE_*.nc                             (required)
#   OUTPUT        output directory                                     (required)
#   MAX_SAMPLES   limit number of timesteps        (optional)
#   START         starting sample index            (optional, default 0)
#   CO2_EXP       LUCIE CO2 experiment index 0-9   (optional, lucie only)
#   ACE_ROOT      ACE outputs root (optional, default <ACE_ROOT>)
#   GPU           CUDA_VISIBLE_DEVICES             (optional, default 0)
# =============================================================================
#SBATCH --job-name=deeprisk_ds
#SBATCH --partition=<PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --mem=100000
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1

set -euo pipefail

: "${DATASET:?Set DATASET=ace|cesm|lucie}"
: "${INPUT:?Set INPUT (see header)}"
: "${OUTPUT:?Set OUTPUT directory}"

source ~/.bashrc
conda activate jax
cd <DEEPRISK_ROOT>

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
mkdir -p "$OUTPUT"

# Optional flags shared by all pipelines
EXTRA=()
[ -n "${MAX_SAMPLES:-}" ] && EXTRA+=(--max_samples "$MAX_SAMPLES")
[ -n "${START:-}" ]       && EXTRA+=(--start_index "$START")

echo "=== DeepRisk $DATASET downscaling | start $(date) ==="
case "$DATASET" in
  ace)
    python run_ace_pipeline.py \
        --experiment_dir "$INPUT" \
        --ace_root "${ACE_ROOT:-<ACE_ROOT>}" \
        --output_dir "$OUTPUT" \
        --save_intermediate "${EXTRA[@]}"
    ;;
  cesm)
    python run_cesm_pipeline.py \
        --cesm_file "$INPUT" \
        --output_dir "$OUTPUT" \
        --save_intermediate "${EXTRA[@]}"
    ;;
  lucie)
    LUCIE_EXTRA=()
    [ -n "${CO2_EXP:-}" ] && LUCIE_EXTRA+=(--co2_experiment "$CO2_EXP")
    python run_integrated_pipeline.py \
        --dataset LUCIE \
        --data_path "$INPUT" \
        --output_dir "$OUTPUT" \
        --save_intermediate "${EXTRA[@]}" "${LUCIE_EXTRA[@]}"
    ;;
  *)
    echo "Unknown DATASET='$DATASET' (use ace|cesm|lucie)"; exit 1 ;;
esac
echo "=== completed $(date) | output: $OUTPUT ==="
