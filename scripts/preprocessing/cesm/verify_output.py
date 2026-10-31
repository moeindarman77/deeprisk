#!/usr/bin/env python
"""
Verify post-processed CAM6 sigma-level output files.

Checks:
  1. File structure (dimensions, variables, attributes)
  2. Physical range of each variable
  3. NaN / missing value check
  4. Spot-check interpolation against manual calculation from raw data
"""

import argparse
import sys
import numpy as np
import xarray as xr

# Expected physical ranges (generous bounds)
RANGES = {
    "T_sigma095": (180.0, 340.0, "K"),
    "Q_sigma095": (-0.1, 30.0, "g/kg"),
    "U_sigma034": (-120.0, 120.0, "m/s"),
    "V_sigma034": (-100.0, 100.0, "m/s"),
    "PRECT": (-0.01, 1000.0, "mm/day"),
}

REQUIRED_VARS = ["T_sigma095", "Q_sigma095", "U_sigma034", "V_sigma034", "PRECT"]
REQUIRED_COORDS = ["time", "lat", "lon"]


def parse_args():
    parser = argparse.ArgumentParser(description="Verify post-processed sigma-level NetCDF file")
    parser.add_argument("--file", required=True, help="Path to the output .nc file")
    parser.add_argument("--raw-file", default=None,
                        help="Path to a raw h1 file for spot-check comparison (optional)")
    return parser.parse_args()


def check_structure(ds):
    """Check that all required variables and coordinates are present."""
    print("\n=== Structure Check ===")
    ok = True

    for coord in REQUIRED_COORDS:
        if coord in ds.coords:
            print(f"  [OK] Coordinate '{coord}': {len(ds[coord])} values")
        else:
            print(f"  [FAIL] Missing coordinate: {coord}")
            ok = False

    for var in REQUIRED_VARS:
        if var in ds.data_vars:
            shape = ds[var].shape
            print(f"  [OK] Variable '{var}': shape={shape}, dtype={ds[var].dtype}")
        else:
            print(f"  [FAIL] Missing variable: {var}")
            ok = False

    # Check CF attributes
    if hasattr(ds, "Conventions"):
        print(f"  [OK] Conventions: {ds.Conventions}")
    else:
        print("  [WARN] Missing 'Conventions' global attribute")

    if hasattr(ds, "sigma_mode"):
        print(f"  [OK] Sigma mode: {ds.sigma_mode}")

    return ok


def check_ranges(ds):
    """Check that variable values fall within expected physical ranges."""
    print("\n=== Physical Range Check ===")
    ok = True

    for var, (vmin, vmax, units) in RANGES.items():
        if var not in ds.data_vars:
            continue

        data = ds[var].values
        dmin = float(np.nanmin(data))
        dmax = float(np.nanmax(data))
        dmean = float(np.nanmean(data))
        dstd = float(np.nanstd(data))

        in_range = dmin >= vmin and dmax <= vmax
        status = "[OK]" if in_range else "[WARN]"
        if not in_range:
            ok = False

        print(f"  {status} {var} ({units}):")
        print(f"         min={dmin:.4f}, max={dmax:.4f}, mean={dmean:.4f}, std={dstd:.4f}")
        print(f"         expected range: [{vmin}, {vmax}]")

    return ok


def check_nans(ds):
    """Check for NaN or infinite values."""
    print("\n=== NaN/Inf Check ===")
    ok = True

    for var in REQUIRED_VARS:
        if var not in ds.data_vars:
            continue

        data = ds[var].values
        n_nan = int(np.isnan(data).sum())
        n_inf = int(np.isinf(data).sum())
        n_total = data.size
        pct_nan = 100.0 * n_nan / n_total if n_total > 0 else 0

        if n_nan > 0 or n_inf > 0:
            print(f"  [WARN] {var}: {n_nan} NaN ({pct_nan:.4f}%), {n_inf} Inf out of {n_total}")
            ok = False
        else:
            print(f"  [OK] {var}: no NaN or Inf ({n_total} values)")

    return ok


def check_time(ds):
    """Check time coordinate for gaps or duplicates."""
    print("\n=== Time Coordinate Check ===")
    times = ds["time"].values
    n = len(times)
    print(f"  Total timesteps: {n}")
    print(f"  First: {times[0]}")
    print(f"  Last:  {times[-1]}")

    # Check for duplicates
    n_unique = len(np.unique(times))
    if n_unique < n:
        print(f"  [WARN] {n - n_unique} duplicate time values found!")
        return False
    else:
        print(f"  [OK] All {n} time values are unique")

    # Check time spacing (expect ~0.25 days = 6 hours)
    if n > 1:
        dt = np.diff(times.astype("datetime64[h]")).astype(float)
        dt_mean = np.mean(dt)
        dt_min = np.min(dt)
        dt_max = np.max(dt)
        print(f"  Time spacing (hours): mean={dt_mean:.1f}, min={dt_min:.1f}, max={dt_max:.1f}")
        if dt_min < 0:
            print("  [WARN] Negative time step detected — possible ordering issue!")
            return False

    return True


def spot_check(ds, raw_file):
    """
    Spot-check a few grid points by manually computing the interpolation
    from a raw h1 file and comparing to the output.
    """
    print(f"\n=== Spot Check vs Raw File ===")
    print(f"  Raw file: {raw_file}")

    raw = xr.open_dataset(raw_file)

    hyam = raw["hyam"].values  # (lev,)
    hybm = raw["hybm"].values  # (lev,)
    P0 = float(raw["P0"].values)  # Pa

    # Use first timestep, check 3 grid points
    t_idx = 0
    check_points = [
        (96, 144, "Equatorial Pacific"),
        (10, 100, "Near South Pole"),
        (150, 200, "Northern midlatitudes"),
    ]

    # Use first timestep of both files (output file starts from same raw file)
    out_t_idx = 0
    print(f"  Checking timestep {t_idx} (raw) = index {out_t_idx} (output)")

    PS = raw["PS"].isel(time=t_idx).values  # (lat, lon) Pa
    T_raw = raw["T"].isel(time=t_idx).values  # (lev, lat, lon)

    all_ok = True
    for ilat, ilon, label in check_points:
        ps = PS[ilat, ilon]

        # Compute pressure column
        p_col = (hyam * P0 + hybm * ps) / 100.0  # hPa
        p_target = 0.95 * ps / 100.0  # hPa

        # Find bracketing levels
        above = p_col <= p_target
        n_above = above.sum()
        k_upper = max(0, min(n_above - 1, len(hyam) - 2))
        k_lower = k_upper + 1

        p1, p2 = p_col[k_upper], p_col[k_lower]
        v1, v2 = T_raw[k_upper, ilat, ilon], T_raw[k_lower, ilat, ilon]

        weight = (np.log(p_target) - np.log(p1)) / (np.log(p2) - np.log(p1))
        weight = np.clip(weight, 0, 1)
        expected_T = v1 + weight * (v2 - v1)

        actual_T = float(ds["T_sigma095"].isel(time=out_t_idx).values[ilat, ilon])
        diff = abs(expected_T - actual_T)

        status = "[OK]" if diff < 0.01 else "[WARN]"
        if diff >= 0.01:
            all_ok = False
        print(f"  {status} {label} (lat={ilat}, lon={ilon}):")
        print(f"         PS={ps:.1f} Pa, P_target={p_target:.2f} hPa")
        print(f"         Expected T={expected_T:.4f} K, Got T={actual_T:.4f} K, Diff={diff:.6f} K")

    raw.close()
    return all_ok


def main():
    args = parse_args()

    print(f"Verifying: {args.file}")
    ds = xr.open_dataset(args.file)

    results = {}
    results["structure"] = check_structure(ds)
    results["ranges"] = check_ranges(ds)
    results["nans"] = check_nans(ds)
    results["time"] = check_time(ds)

    if args.raw_file:
        results["spot_check"] = spot_check(ds, args.raw_file)

    ds.close()

    # Summary
    print(f"\n{'='*50}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*50}")
    all_ok = True
    for check, passed in results.items():
        status = "PASS" if passed else "WARN"
        if not passed:
            all_ok = False
        print(f"  {check:15s}: {status}")

    if all_ok:
        print("\nAll checks passed.")
    else:
        print("\nSome checks had warnings — review above for details.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
