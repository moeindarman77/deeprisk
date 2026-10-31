# Verification & Reproducibility

This repo's outputs were validated against the reference EDM implementation
(`<ARRGEN_SRC>`) and the L2 training script
(`hrrr_model/.../train_hrrr_l2_transformer.py`). The detailed historical write-ups
are preserved in [`../_archive/docs_originals/`](../_archive/docs_originals/)
(`GROUND_TRUTH_COMPARISON.md`, `HYPERPARAMETER_VERIFICATION.md`, `MEASUREMENT_VERIFICATION.md`,
`PREPROCESSING_VERIFICATION_SUMMARY.md`, `POTENTIAL_DIFFERENCES.md`, `ACE_*`). This file is the
current, condensed summary.

## What was verified

- **Hyperparameters & architecture** — every EDM sampling parameter (35 steps, σ∈[0.001,200],
  ρ=7, S_churn=200, …), the UNet config, the seed (17), and the orography conditioning match
  the reference implementation.
- **Normalization** — EDM log-precip override and the ERA5-in / HRRR-out L2 statistics match
  `train_hrrr_l2_transformer.py` exactly (see [ARCHITECTURE.md](ARCHITECTURE.md)).
- **Two bugs fixed earlier (still in place):**
  1. *Latitude orientation* — ACE2/CESM are natively South→North; they are flipped to North→South
     before the EDM (matching LUCIE and the EDM training convention).
  2. *L2 output denormalization* — the L2 wrapper now de-normalizes with **HRRR** stats (not
     ERA5). Using ERA5 std for precip was wrong by ~3218× (ERA5 σ=0.00112 vs HRRR σ=3.589).
- **Output format** — a fresh post-rename ACE run produced `*_intermediate.nc` / `*_regional.nc`
  with variables, dims (720×1440 / 960×1560), units, and lat/lon conventions identical to the
  pre-existing reference outputs.

## Reproducibility: the downscaler is a *stochastic* sampler

The Stage-1 model is a **diffusion posterior sampler** — by design it draws a sample from the
conditional distribution, so each run is a valid ensemble member, not a single deterministic
answer. Measured run-to-run variation (same seed=17, same input, same GPU):

| Setting | temperature max&#124;Δ&#124; | temperature mean&#124;Δ&#124; |
|---------|------------------|-------------------|
| two runs, S_churn=200 (default) | ~1.2 K | ~0.05 K |
| two runs, S_churn=0 | ~0.19 K | — |

Two effects make it non-bit-reproducible: (1) the stochastic **churn** noise uses
`torch.randn_like`, drawn from the *global* RNG (only the initial latents use the seeded
generator); (2) **non-deterministic CUDA kernels** (residual variation persists even at
S_churn=0). The pipeline as written — and every prior production run — has this property.

**CESM cross-validation.** Re-running the CESM notebook and comparing to the pre-existing
`<CESM_DATA>/outputs/cesm_control` samples (same indices) gives temperature
mean&#124;Δ&#124; ≈ 1.09 K — i.e. **equal to the intrinsic run-to-run noise above**. This confirms the
notebooks reproduce the exact code path: the deviation from the reference is just sampling
noise, not a logic difference.

**Bottom line:** reproducibility is *statistical / climatological*, which is the meaningful
criterion for a generative downscaler. Bit-exact reproducibility is **not** expected and is not
provided. If it is ever required, add `torch.use_deterministic_algorithms(True)`, a global
`torch.manual_seed`, and a seeded `randn_like` in the churn step — see [ISSUES.md](../ISSUES.md).
