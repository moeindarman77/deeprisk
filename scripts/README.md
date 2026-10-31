# scripts/

Batch (SLURM) entry points for running the downscaling pipeline at scale.

| File | Purpose |
|------|---------|
| `run_pipeline.sh` | Single parameterized runner. Set `DATASET` (`ace`/`cesm`/`lucie`), `INPUT`, `OUTPUT` (plus optional `MAX_SAMPLES`, `START`, `CO2_EXP`, `ACE_ROOT`, `GPU`). Dispatches to the matching pipeline. |
| `examples/ace2_sst0K.sh` | ACE2 `era5_sst+0K`, full run. |
| `examples/cesm_control.sh` | CESM `SSTcontrol`, full run. |
| `examples/lucie_co2.sh` | LUCIE CO2 experiment 0, first 100 timesteps. |

Notes:
- Activate the conda env first (`conda activate deeprisk`); `run_pipeline.sh` does this for you.
- The SLURM `--output`/`--error` directory must exist **before** `sbatch`; the examples `mkdir -p` it.
- Outputs are written to `/space` (ample free space); avoid `/data` (near full).
- The ~90 historical per-experiment scripts were moved to `../_archive/scripts_experiments/`.
