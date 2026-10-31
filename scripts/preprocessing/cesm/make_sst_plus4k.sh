#!/bin/bash
# Build the +4 K SST boundary file for the CESM perturbation run (see docs/CESM.md, Section 5).
# Copies the standard HadOIBl monthly SST/sea-ice climatology and adds a spatially and temporally
# uniform +4.0 K to SST_cpl. ice_cov is left unchanged. Requires NCO (ncap2).
set -euo pipefail
CTRL=${1:-<CESM_INPUT>/inputdata/atm/cam/sst/sst_HadOIBl_bc_0.9x1.25_clim_c040926.nc}
OUT=${2:-sst_perturbed_plus4K.nc}
cp "$CTRL" "$OUT"
ncap2 -O -s 'SST_cpl=SST_cpl+4.0' "$OUT" "$OUT"
echo "wrote $OUT"
