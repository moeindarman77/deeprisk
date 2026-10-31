# CESM input preparation

Scripts that turn the raw CESM 2.1.5 output into the pipeline's coarse input. The full
experimental design (compset, grid, SST forcing, run configuration) is in
[`docs/CESM.md`](../../../docs/CESM.md).

| File | Purpose |
|------|---------|
| `make_sst_plus4k.sh` | Builds the +4 K SST boundary file from the HadOIBl climatology with `ncap2` (sea ice unchanged). |
| `process_cam6.py` | Interpolates the 6-hourly CAM6 `h1` history files (log-pressure) to σ = 0.95 (T, q) and σ = 0.34 (u, v), converts PRECT to mm/day, and writes one `*_sigma_levels.nc` per experiment. |
| `verify_output.py` | Sanity checks on the post-processed file (ranges, dimensions, time axis). |
| `submit_process.pbs`, `test_process.sh` | PBS submission and a short test run. |

Paths are `<PLACEHOLDER>` tokens; set them for your system before running.
