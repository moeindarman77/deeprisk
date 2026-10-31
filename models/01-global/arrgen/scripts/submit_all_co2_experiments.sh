#!/bin/bash

# Submit ACE downscaling jobs for all CO2 experiments
# This script submits one SLURM job per CO2 experiment

# List of CO2 experiments
CO2_EXPERIMENTS=(
    "era5_co2x1p00"
    "era5_co2x1p25"
    "era5_co2x1p50"
    "era5_co2x1p75"
    "era5_co2x2p00"
    "era5_co2x2p25"
)

echo "================================================================"
echo "Submitting ACE Downscaling Jobs for All CO2 Experiments"
echo "================================================================"
echo "Total experiments: ${#CO2_EXPERIMENTS[@]}"
echo "Start time: $(date)"
echo "================================================================"

# Create logs directory
mkdir -p <SCRATCH>/ace2/logs

# Navigate to scripts directory
cd <DEEPRISK_ROOT>/models/edm/arrgen/scripts

# Array to store job IDs
declare -a JOB_IDS

# Submit jobs for each CO2 experiment
for co2_exp in "${CO2_EXPERIMENTS[@]}"; do
    echo ""
    echo "Submitting job for: $co2_exp"

    # Submit job and capture job ID
    job_output=$(sbatch --job-name="ace_${co2_exp}" run_ace_downscaling_single.sh $co2_exp)
    job_id=$(echo $job_output | awk '{print $NF}')

    JOB_IDS+=($job_id)

    echo "  Job ID: $job_id"
    echo "  Output: <SCRATCH>/ace2/logs/downscale_${job_id}_ace_${co2_exp}.out"
    echo "  Error:  <SCRATCH>/ace2/logs/downscale_${job_id}_ace_${co2_exp}.err"

    # Small delay to avoid overwhelming the scheduler
    sleep 2
done

echo ""
echo "================================================================"
echo "All jobs submitted!"
echo "================================================================"
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u $USER"
echo ""
echo "Check logs in:"
echo "  <SCRATCH>/ace2/logs/"
echo ""
echo "Output will be saved in:"
echo "  <SCRATCH>/ace2/downscaled/[co2_experiment]/"
echo "================================================================"
