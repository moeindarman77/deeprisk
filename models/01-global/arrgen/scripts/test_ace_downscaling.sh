#!/bin/bash
#SBATCH --job-name=ace_test
#SBATCH --partition=<DEBUG_PARTITION>
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=<SCRATCH>/ace2/logs/test_downscale_%j.out
#SBATCH --error=<SCRATCH>/ace2/logs/test_downscale_%j.err

# Test ACE downscaling with 10 samples
# Usage: sbatch test_ace_downscaling.sh

echo "================================================================"
echo "ACE Downscaling TEST Job (10 samples)"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "================================================================"

# Create log and output directories
mkdir -p <SCRATCH>/ace2/logs
mkdir -p <SCRATCH>/ace2/ds

# Activate environment
source ~/.bashrc
conda activate jax

# Navigate to working directory
cd <DEEPRISK_ROOT>/models/edm/arrgen

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Run ACE downscaling with ONLY 10 samples for testing
echo "Starting ACE downscaling test (10 samples from era5_co2x1p00)..."
echo ""

python sample_ace.py \
    +co2_experiment_dir=era5_co2x1p00 \
    +max_samples=10 \
    +merge_gpu_files=false \
    +remove_gpu_files=false

EXIT_CODE=$?

echo ""
echo "================================================================"
echo "Test job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Test completed without errors"
    echo ""
    echo "Output location:"
    echo "  <SCRATCH>/ace2/ds/era5_co2x1p00/"
    echo ""
    echo "Verify the output files:"
    ls -lh <SCRATCH>/ace2/ds/era5_co2x1p00/
else
    echo "❌ FAILED: Test encountered errors (exit code: $EXIT_CODE)"
    echo ""
    echo "Check error log:"
    echo "  <SCRATCH>/ace2/logs/test_downscale_${SLURM_JOB_ID}.err"
fi

echo "================================================================"

exit $EXIT_CODE
