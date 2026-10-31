#!/usr/bin/env python
"""
Post-process CAM6 h1 files: interpolate T, Q, U, V to sigma levels and convert units.

Interpolation method:
  - True sigma: P_target(t,lat,lon) = sigma * PS(t,lat,lon)
  - Logarithmic interpolation between bracketing hybrid levels
  - Pressure at each hybrid level: P(k) = hyam(k)*P0 + hybm(k)*PS(t,lat,lon)

Variables produced:
  T_sigma095 : Temperature at sigma=0.95 (K)
  Q_sigma095 : Specific humidity at sigma=0.95 (g/kg)
  U_sigma034 : Zonal wind at sigma=0.34 (m/s)
  V_sigma034 : Meridional wind at sigma=0.34 (m/s)
  PRECT      : Total precipitation (mm/day)
"""

import argparse
import glob
import os
import sys
import time as timer
from datetime import datetime

import numpy as np
import xarray as xr
from netCDF4 import Dataset

SIGMA_TQ = 0.95
SIGMA_UV = 0.34


def parse_args():
    parser = argparse.ArgumentParser(description="Post-process CAM6 h1 files to sigma levels")
    parser.add_argument("--experiment", required=True, choices=["SSTplus4K", "SSTcontrol"],
                        help="Experiment name")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--test-nfiles", type=int, default=None,
                        help="Process only first N files (for testing)")
    return parser.parse_args()


def get_file_list(experiment):
    """Return sorted list of h1 files for the experiment."""
    base = f"<SCRATCH>/archive/run.{experiment}.01/atm/hist"
    pattern = os.path.join(base, f"run.{experiment}.01.cam.h1.*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"ERROR: No files found matching {pattern}")
    return files


def interpolate_to_sigma(var_3d, pressure_3d, p_target_2d):
    """
    Vectorized logarithmic interpolation of a 3D field to a target pressure surface.

    Parameters
    ----------
    var_3d : ndarray (lev, lat, lon)
        Variable on hybrid levels.
    pressure_3d : ndarray (lev, lat, lon)
        Pressure at each hybrid level in hPa. Must be monotonically increasing
        with level index (top of atmosphere to surface).
    p_target_2d : ndarray (lat, lon)
        Target pressure in hPa at each grid point.

    Returns
    -------
    result : ndarray (lat, lon)
        Interpolated values at the target pressure surface.
    """
    nlev, nlat, nlon = var_3d.shape

    # Boolean mask: True where level pressure is <= target (level is above or at target)
    above = pressure_3d <= p_target_2d[np.newaxis, :, :]  # (lev, lat, lon)

    # Count how many levels are above the target at each grid point
    n_above = above.sum(axis=0)  # (lat, lon)

    # k_upper = index of the last level that is above (or at) the target
    # k_lower = k_upper + 1 (the first level below the target)
    # Clamp to valid range
    k_upper = np.clip(n_above - 1, 0, nlev - 2).astype(int)  # (lat, lon)
    k_lower = k_upper + 1  # (lat, lon)

    # Advanced indexing to gather values at bracketing levels
    lat_idx = np.arange(nlat)[:, np.newaxis]
    lon_idx = np.arange(nlon)[np.newaxis, :]

    p1 = pressure_3d[k_upper, lat_idx, lon_idx]
    p2 = pressure_3d[k_lower, lat_idx, lon_idx]
    v1 = var_3d[k_upper, lat_idx, lon_idx]
    v2 = var_3d[k_lower, lat_idx, lon_idx]

    # Logarithmic interpolation weight
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = (np.log(p_target_2d) - np.log(p1)) / (np.log(p2) - np.log(p1))

    # Clamp weight to [0, 1] for safety (handles edge cases at top/bottom)
    weight = np.clip(weight, 0.0, 1.0)

    # Handle case where p1 == p2 (shouldn't happen but be safe)
    same_level = np.abs(p2 - p1) < 1e-10
    weight = np.where(same_level, 0.0, weight)

    result = v1 + weight * (v2 - v1)
    return result.astype(np.float32)


def create_output_file(filepath, lat, lon, experiment):
    """
    Create the output NetCDF4 file with proper CF attributes and unlimited time dimension.
    Returns the netCDF4 Dataset handle (open for writing).
    """
    ds = Dataset(filepath, "w", format="NETCDF4")

    # Dimensions
    ds.createDimension("time", None)  # unlimited
    ds.createDimension("lat", len(lat))
    ds.createDimension("lon", len(lon))

    # Coordinate variables
    t_var = ds.createVariable("time", "f8", ("time",), zlib=True)
    t_var.units = "days since 2015-01-01 00:00:00"
    t_var.calendar = "noleap"
    t_var.long_name = "time"
    t_var.axis = "T"

    lat_var = ds.createVariable("lat", "f8", ("lat",), zlib=True)
    lat_var.units = "degrees_north"
    lat_var.long_name = "latitude"
    lat_var.axis = "Y"
    lat_var[:] = lat

    lon_var = ds.createVariable("lon", "f8", ("lon",), zlib=True)
    lon_var.units = "degrees_east"
    lon_var.long_name = "longitude"
    lon_var.axis = "X"
    lon_var[:] = lon

    # Data variables with compression
    comp_kwargs = {"zlib": True, "complevel": 4, "chunksizes": (1, len(lat), len(lon))}

    v = ds.createVariable("T_sigma095", "f4", ("time", "lat", "lon"), **comp_kwargs)
    v.long_name = "Temperature at sigma=0.95"
    v.units = "K"
    v.sigma_level = SIGMA_TQ
    v.interpolation_method = "logarithmic in pressure"

    v = ds.createVariable("Q_sigma095", "f4", ("time", "lat", "lon"), **comp_kwargs)
    v.long_name = "Specific humidity at sigma=0.95"
    v.units = "g/kg"
    v.sigma_level = SIGMA_TQ
    v.interpolation_method = "logarithmic in pressure"
    v.note = "Converted from kg/kg to g/kg (multiplied by 1000)"

    v = ds.createVariable("U_sigma034", "f4", ("time", "lat", "lon"), **comp_kwargs)
    v.long_name = "Zonal wind at sigma=0.34"
    v.units = "m/s"
    v.sigma_level = SIGMA_UV
    v.interpolation_method = "logarithmic in pressure"

    v = ds.createVariable("V_sigma034", "f4", ("time", "lat", "lon"), **comp_kwargs)
    v.long_name = "Meridional wind at sigma=0.34"
    v.units = "m/s"
    v.sigma_level = SIGMA_UV
    v.interpolation_method = "logarithmic in pressure"

    v = ds.createVariable("PRECT", "f4", ("time", "lat", "lon"), **comp_kwargs)
    v.long_name = "Total precipitation rate"
    v.units = "mm/day"
    v.note = "Converted from m/s to mm/day (multiplied by 86400000)"

    # Global attributes
    ds.Conventions = "CF-1.0"
    ds.source = f"Post-processed from run.{experiment}.01 CAM6 h1 6-hourly files"
    ds.experiment = f"run.{experiment}.01"
    ds.sigma_mode = "true_sigma: P_target = sigma * PS(t,lat,lon)"
    ds.sigma_TQ = SIGMA_TQ
    ds.sigma_UV = SIGMA_UV
    ds.interpolation = "Logarithmic interpolation in pressure between bracketing hybrid levels"
    ds.pressure_formula = "P(k,lat,lon) = hyam(k)*P0 + hybm(k)*PS(t,lat,lon)"
    ds.history = f"Created {datetime.now().isoformat()} by process_cam6.py"

    return ds


def process_one_file(filepath, hyam, hybm, P0):
    """
    Read one h1 file, interpolate all timesteps to sigma levels.

    Returns
    -------
    T_out, Q_out, U_out, V_out, PRECT_out : ndarray (ntimes, lat, lon)
    time_values : ndarray of float64 (time in days since reference)
    """
    ds = xr.open_dataset(filepath)
    ntimes = len(ds.time)
    nlat = len(ds.lat)
    nlon = len(ds.lon)

    # Pre-allocate
    T_out = np.empty((ntimes, nlat, nlon), dtype=np.float32)
    Q_out = np.empty((ntimes, nlat, nlon), dtype=np.float32)
    U_out = np.empty((ntimes, nlat, nlon), dtype=np.float32)
    V_out = np.empty((ntimes, nlat, nlon), dtype=np.float32)
    PRECT_out = np.empty((ntimes, nlat, nlon), dtype=np.float32)

    # Read raw numeric time values (days since 2015-01-01, noleap calendar)
    # Use the underlying encoded values to avoid cftime object issues
    from netCDF4 import Dataset as NC4Dataset
    nc = NC4Dataset(filepath, "r")
    time_values = nc.variables["time"][:].data  # float64 array
    nc.close()

    for t in range(ntimes):
        # Surface pressure in Pa
        PS = ds["PS"].isel(time=t).values  # (lat, lon), Pa

        # Build 3D pressure field in hPa: (lev, lat, lon)
        # hyam, hybm are (lev,); P0 is scalar in Pa; PS is (lat, lon) in Pa
        pressure_3d = (hyam[:, np.newaxis, np.newaxis] * P0 +
                       hybm[:, np.newaxis, np.newaxis] * PS[np.newaxis, :, :]) / 100.0

        # Target pressures (true sigma): hPa
        p_target_TQ = SIGMA_TQ * PS / 100.0  # (lat, lon)
        p_target_UV = SIGMA_UV * PS / 100.0  # (lat, lon)

        # Interpolate
        T_out[t] = interpolate_to_sigma(
            ds["T"].isel(time=t).values, pressure_3d, p_target_TQ)
        Q_out[t] = interpolate_to_sigma(
            ds["Q"].isel(time=t).values, pressure_3d, p_target_TQ) * 1000.0  # g/kg
        U_out[t] = interpolate_to_sigma(
            ds["U"].isel(time=t).values, pressure_3d, p_target_UV)
        V_out[t] = interpolate_to_sigma(
            ds["V"].isel(time=t).values, pressure_3d, p_target_UV)

        # Precipitation: m/s -> mm/day
        PRECT_out[t] = ds["PRECT"].isel(time=t).values.astype(np.float32) * 86400.0 * 1000.0

    ds.close()
    return T_out, Q_out, U_out, V_out, PRECT_out, time_values


def main():
    args = parse_args()
    experiment = args.experiment
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Get file list
    files = get_file_list(experiment)
    if args.test_nfiles:
        files = files[: args.test_nfiles]
    print(f"Experiment: {experiment}")
    print(f"Files to process: {len(files)}")
    print(f"Output directory: {output_dir}")

    # Read coordinate info from first file
    print("\nReading coordinate metadata from first file...")
    ds0 = xr.open_dataset(files[0])
    lat = ds0["lat"].values
    lon = ds0["lon"].values
    hyam = ds0["hyam"].values  # (lev,) — dimensionless
    hybm = ds0["hybm"].values  # (lev,) — dimensionless
    P0 = float(ds0["P0"].values)  # Pa (should be 100000.0)
    print(f"  Grid: {len(lat)} lat x {len(lon)} lon, {len(hyam)} levels")
    print(f"  P0 = {P0:.1f} Pa ({P0/100:.1f} hPa)")
    print(f"  Sigma targets: T,Q at sigma={SIGMA_TQ}, U,V at sigma={SIGMA_UV}")
    ds0.close()

    # Create output file
    out_filename = f"{experiment}_sigma_levels.nc"
    out_path = os.path.join(output_dir, out_filename)
    print(f"\nCreating output file: {out_path}")
    out_ds = create_output_file(out_path, lat, lon, experiment)

    # Process files incrementally
    time_offset = 0
    total_start = timer.time()

    for i, filepath in enumerate(files):
        fname = os.path.basename(filepath)
        file_start = timer.time()
        print(f"\n[{i+1}/{len(files)}] Processing {fname}...")

        T, Q, U, V, PRECT, time_vals = process_one_file(filepath, hyam, hybm, P0)
        ntimes = T.shape[0]

        # Write time values (already numeric: days since 2015-01-01)
        out_ds["time"][time_offset: time_offset + ntimes] = time_vals

        # Write data
        out_ds["T_sigma095"][time_offset: time_offset + ntimes, :, :] = T
        out_ds["Q_sigma095"][time_offset: time_offset + ntimes, :, :] = Q
        out_ds["U_sigma034"][time_offset: time_offset + ntimes, :, :] = U
        out_ds["V_sigma034"][time_offset: time_offset + ntimes, :, :] = V
        out_ds["PRECT"][time_offset: time_offset + ntimes, :, :] = PRECT

        time_offset += ntimes
        elapsed = timer.time() - file_start
        print(f"  {ntimes} timesteps written (total: {time_offset}), took {elapsed:.1f}s")

    out_ds.close()

    total_elapsed = timer.time() - total_start
    file_size_gb = os.path.getsize(out_path) / (1024**3)
    print(f"\n{'='*60}")
    print(f"DONE: {out_path}")
    print(f"  Total timesteps: {time_offset}")
    print(f"  File size: {file_size_gb:.2f} GB")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
