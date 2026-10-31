#!/bin/bash
# Test processing on 2 files from SSTplus4K
# Run interactively: bash test_process.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_DIR="<SCRATCH>/sst_data/postprocessed/test"
RAW_DIR="<SCRATCH>/archive/run.SSTplus4K.01/atm/hist"

# Get first raw file for spot-check
FIRST_RAW=$(ls ${RAW_DIR}/run.SSTplus4K.01.cam.h1.*.nc | head -1)

module load conda
conda activate deeprisk

mkdir -p ${TEST_DIR}

echo "============================================"
echo "Testing on 2 files from SSTplus4K"
echo "============================================"

python ${SCRIPT_DIR}/process_cam6.py \
    --experiment SSTplus4K \
    --test-nfiles 2 \
    --output-dir ${TEST_DIR}

echo ""
echo "============================================"
echo "Verifying output"
echo "============================================"

python ${SCRIPT_DIR}/verify_output.py \
    --file ${TEST_DIR}/SSTplus4K_sigma_levels.nc \
    --raw-file ${FIRST_RAW}

echo ""
echo "============================================"
echo "Quick ncdump -h of output"
echo "============================================"
ncdump -h ${TEST_DIR}/SSTplus4K_sigma_levels.nc

echo ""
echo "Done. Review the output above before submitting the full job."
