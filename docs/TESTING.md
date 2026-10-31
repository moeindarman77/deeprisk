# Testing, Running & Training

## Environment

```bash
source ~/.bashrc && conda activate jax     # PyTorch+CUDA, xarray, cartopy, jupyter, ...
# GPU required for inference; pick a free device, e.g. export CUDA_VISIBLE_DEVICES=0
```

## Smoke tests

Minimal checks kept in [`../test_scripts/`](../test_scripts/):

```bash
python test_scripts/test_config.py        # config files load & validate
python test_scripts/test_components.py     # core wrappers import / construct
python test_scripts/test_integration.py    # end-to-end wiring on a tiny sample
```

(The larger GPU / multi-GPU / batch / CO2 performance tests were moved to
`../_archive/test_scripts/`.)

## Run the notebooks

```bash
cd notebooks
# Stage 1 first (produces the intermediate inputs Stage 2 consumes):
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 \
  "CESM - Intermediate Downscaling Inference.ipynb"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 \
  "CESM - Regional Downscaling Inference.ipynb"
```

Each Intermediate notebook downscales 10 sample timesteps (~12 s/sample on an GPU); the
Regional stage is a single forward pass per sample (fast). All 6 notebooks were executed
end-to-end during shipping and produce 10 intermediate + 10 regional `.nc` files plus styled
plots.

## Batch inference

See [`../scripts/README.md`](../scripts/README.md): one `run_pipeline.sh` dispatches to the
ACE/CESM/LUCIE pipeline by `DATASET`.

## Reproducibility

The Stage-1 diffusion model is a **stochastic posterior sampler**: outputs are reproducible
*statistically*, not bit-for-bit (see [VERIFICATION.md](VERIFICATION.md) for measurements). The
seed (`17`) fixes the initial latents; the churn step and CUDA kernels add run-to-run variation
of ~1 K (temperature). This matches all prior production runs.

## Training (external, long-running)

Neither model is trained inside this repo — they are trained in their source repos and the
checkpoints are vendored here. Commands, for reference:

**Stage 1 — EDM (arrgen, Hydra, multi-GPU):**
```bash
cd <ARRGEN_SRC>
torchrun --nproc_per_node=<N> -m arrgen.train experiment=era5_0.25   # see arrgen/configs/
```

**Stage 2 — L2 Transformer:**
```bash
cd <HRRR_MODEL_SRC>
python src/tools/train_hrrr_l2_transformer.py \
    --config_path <ERA5_config_final_transformer_based_L2.yaml> --use_conditioning
# data: HRRR <SCRATCH>/hrrr/hrrr_data_2015_2024_t2m_acpc.h5 + ERA5 <ERA5_DATA>/
# normalization: ERA5 in, HRRR out (see docs/ARCHITECTURE.md)
```

Both are multi-hour GPU jobs (submit via SLURM `<PARTITION>`); run them to convergence in
their home repos, then copy the resulting checkpoint into `models/01-global/checkpoints/` or
`models/02-regional/checkpoints/`.
