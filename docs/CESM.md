# CESM2 Simulation Setup — Full Methods Context

*Reconstructed from the case directories on an HPC cluster (`<WORK>/cases/` and
`<SCRATCH>/`). Intended as source material for a paper Methods section.*

---

## 1. Overview / experimental design

A pair of **atmosphere-only (AMIP / prescribed-SST) CESM2 simulations** was run to
measure the atmospheric response to uniform ocean warming:

| Case | SST forcing | Purpose |
|------|-------------|---------|
| `run.SSTcontrol.01` | HadOIBl monthly SST/sea-ice **climatology** | Control |
| `run.SSTplus4K.01`  | Same climatology **+ uniform 4.0 K** | Warming perturbation |

The two cases are **identical in every respect except the prescribed SST boundary
condition** (same model version, grid, compset, physics, PE layout, output configuration,
and namelists — verified that `user_nl_*` files match and `SourceMods` is empty). This is
the classic single-forcing design: differences between the two integrations isolate the
atmospheric response to a +4 K sea-surface warming.

---

## 2. Model, version, and platform

- **Model:** CESM 2.1.5 (source at `<WORK>/code/cesm2.1.5`)
- **Machine:** an HPC center **an HPC cluster** (HPE Cray EX, Intel compiler)
- **Project/allocation:** `<PROJECT_CODE>`
- **Case creation command (both cases identical except `--case`):**

  ```
  ./create_newcase \
      --case run.SST{control|plus4K}.01 \
      --res f09_f09_mg17 \
      --compset SSP370_CAM60_CLM50%BGC-CROP_CICE%PRES_DOCN%DOM_MOSART_SGLC_SWAV \
      --machine <MACHINE> \
      --project <PROJECT_CODE> \
      --run-unsupported
  ```

- **No source-code modifications** (`SourceMods/` empty) — out-of-the-box CESM2.1.5.

## 3. Component set (compset)

Longname: `SSP370_CAM60_CLM50%BGC-CROP_CICE%PRES_DOCN%DOM_MOSART_SGLC_SWAV`

| Component | Configuration |
|-----------|---------------|
| **Forcing scenario** | CMIP6 **SSP3-7.0** (greenhouse gases, aerosols, etc. follow SSP3-7.0) |
| **ATM** | **CAM6** (cam6 physics) — active, prognostic atmosphere |
| **LND** | **CLM5.0** with BGC (biogeochemistry, CN + methane) and prognostic **CROP** |
| **ICE** | **CICE5** in **prescribed mode** (`%PRES`) — sea-ice concentration read from the SST/ice boundary file, not computed |
| **OCN** | **Data ocean (DOCN) in "DOM" prescribed-SST mode** (`%DOM`) — SST read from boundary file; this is what makes it an AMIP-type run |
| **ROF** | **MOSART** river-transport model |
| **GLC** | Stub (`SGLC`) — no ice sheet |
| **WAV** | Stub (`SWAV`) — no waves |

## 4. Resolution / grid

- **Grid alias:** `f09_f09_mg17`
- **Full grid name:** `a%0.9x1.25_l%0.9x1.25_oi%0.9x1.25_r%r05_g%null_w%null_m%gx1v7`
- Atmosphere & land: **finite-volume 0.9° × 1.25°** (nominal 1°), **288 × 192** grid points
- Ocean/ice data grid: 0.9° × 1.25°
- River (MOSART): `r05` (0.5°)
- Ocean mask: `gx1v7`

## 5. SST / sea-ice boundary forcing  ← key difference between cases

Both cases use the standard CESM data-ocean prescribed-SST machinery (`DOCN%DOM`, prescribed
CICE), reading SST (`SST_cpl`) and sea-ice concentration (`ice_cov`) from a boundary file.
Domain/grid file for the stream: `domain.ocn.fv0.9x1.25_gx1v7.151020.nc`.

**Control (`run.SSTcontrol.01`):**
- `SSTICE_DATA_FILENAME = .../atm/cam/sst/sst_HadOIBl_bc_0.9x1.25_clim_c040926.nc`
- This is the standard CESM **HadOIBl monthly SST/sea-ice climatology** on the 0.9×1.25 grid
  (12 monthly values, mid-month). Hurrell et al. (2008)-style merged HadISST/Reynolds product.

**+4 K (`run.SSTplus4K.01`):**
- `SSTICE_DATA_FILENAME = .../run.SSTplus4K.01/sst_perturbed_plus4K.nc`
- Created from a copy of the **same climatology file** by adding a **spatially and temporally
  uniform +4.0 K to `SST_cpl`** using NCO:
  ```
  ncap2 -O -s 'SST_cpl=SST_cpl+4.0' sst_perturbed_plus4K.nc sst_perturbed_plus4K.nc
  ```
- Verified: file has 12 months, 288×192; SST minimum shifts from −4.49 °C (control) to
  −0.49 °C (+4 K); **`ice_cov` is left unchanged** (sea-ice field identical to control).

**SST cycling settings (both cases):** `SSTICE_YEAR_ALIGN=1`, `SSTICE_YEAR_START=0`,
`SSTICE_YEAR_END=0` → the 12-month climatology is applied **cyclically (repeating annual
cycle)** for every simulated year. There is **no interannual SST variability**; each model
year sees the identical SST annual cycle (control) or that cycle + 4 K (perturbation).

> **Note on provenance:** A later Python script (`create_sst_plus4k.py`, dated 26 Mar 2026 —
> *after* these runs finished on 17 Mar) describes an alternative +4 K file built from the
> transient 1850–2021 1° HadOIBl file with sea ice removed where warmed SST > 0 °C. **That
> alternative was not used here and should be ignored for this paper.** The simulations analyzed
> in this study used the simple uniform-+4 K **climatology** file (`sst_perturbed_plus4K.nc`,
> from `ncap2`) with sea ice unchanged, as described above.

## 6. Run configuration (both cases identical)

- **Run type:** startup
- **Start date:** 2015-01-01
- **Calendar:** `NO_LEAP` (365-day)
- **Atmosphere coupling frequency:** `ATM_NCPL = 48` per day → **30-minute** coupling/physics timestep
- **Job submission:** `STOP_N = 2`, `STOP_OPTION = nyears` (2-year segments), `RESUBMIT = 0`;
  the 8-year integration was completed by repeated manual `case.submit`/continue-run resubmissions
  (visible in `CaseStatus`).
- **Total simulated period:** **2015-01 through 2022-12 (8 model years)**, confirmed from the
  archived history files (control h1 spans 2015-01-01 → 2022-12-22; restart/initial files reach
  2023-01-01). Note: since SST is a repeating climatology, the calendar years 2015–2022 are
  effectively **8 realizations of the same SST annual cycle**, differing only in atmospheric
  internal variability. **All 8 years are used in the analysis** (no spin-up year discarded).

## 7. Output / history configuration (`user_nl_cam`)

```
empty_htapes = .true.
nhtfrq       = 0, -6          ! h0 monthly, h1 6-hourly
mfilt        = 1, 120         ! h1: 120 time samples per file (~30 days)
fincl2       = 'T:I','U:I','V:I','Q:I','PRECT','PS:I'
```

- **Primary output is the `h1` stream: 6-hourly fields**, instantaneous (`:I`) temperature (T),
  zonal/meridional wind (U, V), specific humidity (Q), surface pressure (PS), plus total
  precipitation rate PRECT (time-averaged, as it is a rate).
- `empty_htapes=.true.` with no `fincl1` ⇒ the **monthly `h0` stream carries no fields**;
  in the archive there are **0 `h0` files and 98 `h1` files** per case — i.e. only the 6-hourly
  stream was saved.
- Each `h1` file = 120 samples × 6-hourly ≈ 30 days; 98 files ≈ 8 years.

All other components used default namelists (`user_nl_clm`, `user_nl_cice`, `user_nl_docn`,
`user_nl_cpl`, `user_nl_mosart` contain no user changes).

## 8. Computational setup / performance

- **PE layout:** 512 MPI tasks, 1 thread/task, all components on the same 512 PEs
  (`TOTALPES = 512`); 128 tasks/node ⇒ **4 an HPC cluster nodes**.
- **Throughput:** ≈ **8.4–8.6 simulated years/day**; cost ≈ **1,430–1,470 pe-hrs/simulated year**.
- Wallclock request 12:00:00 per 2-year segment.

## 9. Decisions locked in (author-confirmed)

1. **+4 K file:** Describe the experiment **as actually run** — a spatially/temporally **uniform
   +4.0 K added to a repeating monthly HadOIBl SST climatology, with sea ice left unchanged**
   (`sst_perturbed_plus4K.nc`, built with `ncap2`). The alternative transient/ice-adjusted file
   (`create_sst_plus4k.py`) is **not** part of this study and should not be mentioned.
2. **Terminology — do NOT call this "GOGA."** GOGA implies observed, interannually-varying global
   SST; this experiment uses a **climatological SST annual cycle plus a uniform +4 K offset**.
   Describe it as a **"prescribed-SST (AMIP-style) uniform-warming (+4 K) sensitivity experiment."**
3. **Analysis window:** **All 8 simulated years (2015–2022) are used**; no spin-up year is discarded.

## 10. Post-processing of CAM6 output

The raw 6-hourly `h1` files are post-processed before analysis
(`<WORK>/cases/postprocess_sst/process_cam6.py`, run via `submit_process.pbs`
on an HPC cluster, project <PROJECT_CODE>).

- **Vertical interpolation to sigma surfaces.** For each time step the model's hybrid-sigma
  fields are interpolated to fixed sigma levels, where the target pressure is
  `P_target = σ · PS(t,lat,lon)` and the pressure at each hybrid level is
  `P(k) = hyam(k)·P0 + hybm(k)·PS`. Interpolation is **logarithmic in pressure** between the
  two bracketing hybrid levels.
- **Extracted fields:**
  - `T_sigma095` — temperature at **σ = 0.95** (K)
  - `Q_sigma095` — specific humidity at **σ = 0.95** (converted to g/kg)
  - `U_sigma034` — zonal wind at **σ = 0.34** (m/s)
  - `V_sigma034` — meridional wind at **σ = 0.34** (m/s)
  - `PRECT` — total precipitation, converted to **mm/day**
- **Output:** one consolidated file per experiment —
  `<SCRATCH>/sst_data/postprocessed/SSTcontrol_sigma_levels.nc` and
  `SSTplus4K_sigma_levels.nc` (~9.5 GB each, full 8-year 6-hourly record), plus a `verify_output.py`
  validation pass.

**Observational reference (ERA5).** ERA5 fields were downloaded from the Copernicus CDS
(`reanalysis-era5-single-levels`; scripts in
`<SCRATCH>/sst_data/era5_data_download/`, e.g. `download_sst.py`,
`download_newvars.py`, `download_tisr.py`) to provide an observational comparison for the same
variables. (Note: this directory also contains assets for a separate ML-emulator effort — LUCIE
initial-condition generation, `goga_vs_lucie*` notebooks — which is outside the scope of the
CESM run description here.)

## 11. Key file locations (provenance)

- Control case: `<WORK>/cases/run.SSTcontrol.01`
- +4 K case: `<WORK>/cases/run.SSTplus4K.01`
- +4 K SST file (as used): `<WORK>/cases/run.SSTplus4K.01/sst_perturbed_plus4K.nc`
- Control SST file: `<CESM_INPUT>/inputdata/atm/cam/sst/sst_HadOIBl_bc_0.9x1.25_clim_c040926.nc`
- Archived output: `<SCRATCH>/archive/run.SST{control,plus4K}.01/atm/hist/*.cam.h1.*`
- Later (unused) generator script: `<SCRATCH>/sst_data/era5_data_download/create_sst_plus4k.py`
