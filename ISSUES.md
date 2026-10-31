# Known Issues & Future Work

Status as of the shipping pass. Resolved items from older docs are folded into
[`docs/`](docs/); the originals are kept in [`_archive/docs_originals/`](_archive/docs_originals/).

## Open

1. **Emulator SST-forcing perturbation script not included.** The +4 K forcing files for
   LUCIE-3D and ACE2 were produced by a script (ocean-only +4 K with Gaussian coastal blending,
   σ = 1.7 grid cells, up to 20 cells from the coast; control processed with 0 K) that is not in
   this repository. The procedure is specified in README "Code and Data Availability" and in the
   paper's Methods; re-implementing it is straightforward but has not been done here.

2. **No bit-exact reproducibility (by design).** The Stage-1 diffusion sampler is stochastic;
   the churn step uses the unseeded global RNG and CUDA kernels are non-deterministic, so runs
   differ by ~1 K (temperature). See [docs/VERIFICATION.md](docs/VERIFICATION.md). If exact
   reproducibility is needed, add `torch.use_deterministic_algorithms(True)`, a global
   `torch.manual_seed(17)`, and pass a seeded generator into the churn `randn_like`. This would
   *change* numerics vs. all existing outputs, so it was intentionally **not** done.

3. **Hard-coded absolute paths.** Drivers, configs, and notebooks use absolute
   `<DEEPRISK_ROOT>/...` and `<SPACE_ROOT>/...` paths. Fine on this cluster; would need
   parameterizing for another environment. Notebook `INPUT_DIR`/`OUTPUT_DIR` are surfaced at the
   top of each notebook for easy editing.

4. **LUCIE-only integrated path.** `run_integrated_pipeline.py` (via `core/`) supports **LUCIE**
   only — `core/models/edm_model.py` raises `NotImplementedError` for the ACE/CESM
   super-resolution shapes. ACE2 and CESM use the standalone drivers (`run_ace_pipeline.py`,
   `run_cesm_pipeline.py`), which is the supported path. (The superseded
   `run_integrated_pipeline_ace.py` was archived.)

5. **Stale paths inside `_archive/`.** Archived scripts/docs still reference the old
   `models/edm` / `models/l2` names and external checkpoint paths. They are kept for history
   only and are not part of the shipped pipeline; do not run them as-is.

6. **Regional precipitation units label.** Regional `tp6hr` is de-normalized with HRRR
   statistics (mean 0.606, std 3.589) and so legitimately reaches values of tens, yet the
   NetCDF attribute says `units = 'm'` (inherited from the source pipeline). The values are on
   the HRRR accumulation scale, not metres. Additionally, the diffusion (intermediate) stage can
   produce small **negative** precip — a known generative artifact. Both are existing pipeline
   behavior, left unchanged (zero-deviation). Consider clamping precip at ≥0 and correcting the
   units attribute in a future, deliberate change.

## Resolved (folded into current docs)

- External `arrgen` dependency: the package is now vendored at `models/01-global/arrgen/` (the
  working copy used for the paper runs, scrubbed of local paths) and imported through a
  repository-relative `sys.path` entry in the drivers, `core/models/edm_model.py`, and the
  notebooks. Two classes (`IGRAOperator`, `SR_IGRA_Operator`) loaded an HRRR grid file in their
  default arguments, i.e. at import time; those defaults are now lazy so the module imports anywhere.
  Verified: imports succeed and `checkpoint-007862.pt` loads into the vendored network with strict
  key matching.

- CESM input data located and validated (`<CESM_DATA>/*_sigma_levels.nc`); all 6
  notebooks run end-to-end and CESM is cross-validated against pre-existing outputs.
- Model dirs renamed `edm→01-global`, `l2→02-regional`; all kept code/config repointed.
- Small artifacts (`condition.npy`, orography) vendored into the repo and verified byte-identical.
- L2 output denormalization (HRRR stats) and ACE/CESM latitude orientation — fixed and verified.
