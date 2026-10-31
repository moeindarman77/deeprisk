# Known Issues & Future Work

Status as of the shipping pass. Resolved items from older docs are folded into
[`docs/`](docs/); the originals are kept in [`_archive/docs_originals/`](_archive/docs_originals/).

## Open

1. **External `arrgen` dependency (not vendored).** The EDM network class is imported from
   `<ARRGEN_SRC>` at runtime (added to `sys.path`). The repo is otherwise
   self-contained (checkpoints, stats, `condition.npy`, orography are vendored under `models/`).
   To make it fully portable, vendor or `pip install -e` the arrgen package and drop the
   hard-coded `sys.path.insert`. *Impact: the pipeline will not run on a machine without that
   path.* `models/01-global/arrgen/` is a stale partial copy (not imported) — do not rely on it.

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

- CESM input data located and validated (`<CESM_DATA>/*_sigma_levels.nc`); all 6
  notebooks run end-to-end and CESM is cross-validated against pre-existing outputs.
- Model dirs renamed `edm→01-global`, `l2→02-regional`; all kept code/config repointed.
- Small artifacts (`condition.npy`, orography) vendored into the repo and verified byte-identical.
- L2 output denormalization (HRRR stats) and ACE/CESM latitude orientation — fixed and verified.
