#!/bin/bash
#SBATCH --job-name=ace_downscale
#SBATCH --partition=<PARTITION>
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=<SCRATCH>/ace2/logs/downscale_%j_%x.out
#SBATCH --error=<SCRATCH>/ace2/logs/downscale_%j_%x.err

# Single ACE downscaling job for one CO2 experiment
# Usage: sbatch run_ace_downscaling_single.sh era5_co2x1p00

CO2_EXPERIMENT=${1:-era5_co2x1p00}

echo "================================================================"
echo "ACE Downscaling Job"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "CO2 Experiment: $CO2_EXPERIMENT"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "================================================================"

# Create log directory
mkdir -p <SCRATCH>/ace2/logs

# Activate environment
source ~/.bashrc
conda activate jax

# Navigate to working directory
cd <DEEPRISK_ROOT>/models/edm/arrgen

# Check GPU
nvidia-smi

# Run ACE downscaling
python sample_ace.py \
    +co2_experiment_dir=$CO2_EXPERIMENT \
    +max_samples=null \
    +merge_gpu_files=false \
    +remove_gpu_files=false

echo "================================================================"
echo "Job completed at: $(date)"
echo "================================================================"
