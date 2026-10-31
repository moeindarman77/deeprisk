import math
import dataclasses
from typing import Union

import torch
import xarray as xr

Tensor = torch.Tensor

@dataclasses.dataclass(frozen=True)
class Grid:
    """Representation of a rectilinear grid with radian coordinates."""
    lon: Tensor  # shape (n_lon,)
    lat: Tensor  # shape (n_lat,)

    @classmethod
    def from_degrees(cls, lon: Union[Tensor, list, tuple], lat: Union[Tensor, list, tuple]) -> "Grid":
        lon_t = lon if isinstance(lon, torch.Tensor) else torch.tensor(lon, dtype=torch.float32)
        lat_t = lat if isinstance(lat, torch.Tensor) else torch.tensor(lat, dtype=torch.float32)
        deg2rad = math.pi / 180.0
        return cls(lon_t * deg2rad, lat_t * deg2rad)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.lon.size(0), self.lat.size(0))

    def __eq__(self, other):
        return isinstance(other, Grid) and torch.equal(self.lon, other.lon) and torch.equal(self.lat, other.lat)

    def __hash__(self):
        return hash((tuple(self.lon.tolist()), tuple(self.lat.tolist())))

@dataclasses.dataclass(frozen=True)
class Regridder:
    source: Grid
    target: Grid

    def regrid_array(self, field: Tensor) -> Tensor:
        """Regrid a tensor with dims (..., lon, lat) to (..., new_lon, new_lat)."""
        raise NotImplementedError

    def regrid_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        """Regrid an xarray.Dataset from source to target using torch ops."""
        if not (dataset['latitude'].diff('latitude') > 0).all():
            dataset = dataset.isel(latitude=slice(None, None, -1))
        out = dataset.copy()
        for var in dataset.data_vars:
            arr = dataset[var].values
            x = torch.from_numpy(arr.astype('float32'))
            r = self.regrid_array(x)
            out_vals = r.detach().cpu().numpy()
            dims = dataset[var].dims[:-2] + ('longitude', 'latitude')
            coords = {d: dataset.coords[d] for d in dataset[var].dims[:-2]}
            coords['longitude'] = self.target.lon.cpu().numpy()
            coords['latitude'] = self.target.lat.cpu().numpy()
            out[var] = xr.DataArray(out_vals, dims=dims, coords=coords)
        return out

# Helpers for monotonicity and bounds

def _assert_increasing(x: Tensor) -> None:
    if not (x[1:] > x[:-1]).all():
        raise ValueError(f"Tensor not strictly increasing: {x}")


def _latitude_cell_bounds(x: Tensor) -> Tensor:
    pi2 = math.pi / 2
    mids = (x[:-1] + x[1:]) * 0.5
    return torch.cat([torch.tensor([-pi2], dtype=x.dtype, device=x.device), mids, torch.tensor([pi2], dtype=x.dtype, device=x.device)])


def _latitude_overlap(src: Tensor, tgt: Tensor) -> Tensor:
    sb = _latitude_cell_bounds(src)
    tb = _latitude_cell_bounds(tgt)
    up = torch.min(tb[1:].unsqueeze(1), sb[1:].unsqueeze(0))
    lo = torch.max(tb[:-1].unsqueeze(1), sb[:-1].unsqueeze(0))
    return (up > lo) * (torch.sin(up) - torch.sin(lo))


def _conservative_latitude_weights(src: Tensor, tgt: Tensor) -> Tensor:
    _assert_increasing(src)
    _assert_increasing(tgt)
    w = _latitude_overlap(src, tgt)
    return w / w.sum(dim=1, keepdim=True)

# Longitude-side

def _align_phase(x: Tensor, target: Tensor, period: float) -> Tensor:
    down = x > target + period * 0.5
    up = x < target - period * 0.5
    return x - down.to(x.dtype) * period + up.to(x.dtype) * period


def _periodic_bounds(x: Tensor, period: float) -> tuple[Tensor, Tensor]:
    x_mod = x % period
    x_next = torch.roll(x_mod, -1)
    up = (x_mod + _align_phase(x_next, x_mod, period)) * 0.5
    x_prev = torch.roll(x_mod, 1)
    lo = (x_mod + _align_phase(x_prev, x_mod, period)) * 0.5
    return lo, up


def _longitude_overlap(fp: Tensor, sp: Tensor, period: float = 2 * math.pi) -> Tensor:
    fl, fu = _periodic_bounds(fp, period)
    sl, su = _periodic_bounds(sp, period)
    i = fl.unsqueeze(1)
    j = fu.unsqueeze(1)
    k = sl.unsqueeze(0)
    l = su.unsqueeze(0)
    lo = torch.max(i, k)
    up = torch.min(j, l)
    return torch.clamp(up - lo, min=0)


def _conservative_longitude_weights(src: Tensor, tgt: Tensor) -> Tensor:
    _assert_increasing(src)
    _assert_increasing(tgt)
    w = _longitude_overlap(tgt, src)
    return w / w.sum(dim=1, keepdim=True)

class ConservativeRegridder(Regridder):
    """Fully differentiable conservative regridder using torch."""

    def __init__(self, source: Grid, target: Grid):
        object.__setattr__(self, 'source', source)
        object.__setattr__(self, 'target', target)
        lon_w = _conservative_longitude_weights(source.lon, target.lon)
        lat_w = _conservative_latitude_weights(source.lat, target.lat)
        object.__setattr__(self, 'lon_weights', lon_w)
        object.__setattr__(self, 'lat_weights', lat_w)

    def _mean(self, field: Tensor) -> Tensor:
        return torch.einsum('ij,kl,...jl->...ik', self.lon_weights, self.lat_weights, field)

    def _nanmean(self, field: Tensor) -> Tensor:
        mask = torch.isnan(field)
        total = self._mean(torch.where(mask, torch.zeros_like(field), field))
        # cast mask to float for proper dtype in einsum
        count = self._mean(mask.logical_not().to(field.dtype))
        return total / count

    def regrid_array(self, field: Tensor) -> Tensor:
        # expects field shape (..., lon, lat)
        return self._nanmean(field)

    def regrid_tensor(self, x: Tensor) -> Tensor:
        """Regrid tensor with dims (..., lat, lon)."""
        perm = list(range(x.ndim)); perm[-2], perm[-1] = perm[-1], perm[-2]
        y = x.permute(*perm)
        z = self._nanmean(y)
        return z.permute(*perm)



class Regrid:
    """
    A class to perform conservative regridding.

    This class initializes the source and target grids upon instantiation,
    allowing for efficient regridding of multiple data tensors without
    re-computing the regridding weights.
    """

    def __init__(
        self,
        src_lat: torch.Tensor,      # 1D, strictly increasing, in degrees
        src_lon: torch.Tensor,      # 1D, in degrees
        ddeg_out: float,            # Output grid spacing in degrees
    ):
        """
        Initializes the source grid, target grid, and the regridder object.

        Args:
            src_lat: A 1D tensor of source latitude values, which must be
                     strictly increasing.
            src_lon: A 1D tensor of source longitude values.
            ddeg_out: The grid spacing for the output regular grid in degrees.
        """
        # 1) Sanity-check that lat is sorted
        if not torch.all(src_lat[1:] > src_lat[:-1]):
            raise ValueError("`src_lat` must be strictly increasing")

        # 2) Compute new grid centers for the target grid
        lat_start = -90.0 + 0.5 * ddeg_out
        lat_stop  =  90.0 - 0.5 * ddeg_out

        device, dtype = src_lat.device, src_lat.dtype

        # Latitude: includes both endpoints, with exactly 180/ddeg_out points
        self.new_lat = torch.linspace(
            lat_start,
            lat_stop,
            steps=int(180.0 / ddeg_out),
            dtype=dtype,
            device=device
        )

        # Longitude: 0, ddeg_out, ..., up to (but not including) 360
        self.new_lon = torch.arange(
            0.0,
            360.0,
            step=ddeg_out,
            dtype=dtype,
            device=device
        )

        # 3) Build grid objects
        source_grid = Grid.from_degrees(lon=src_lon, lat=src_lat)
        target_grid = Grid.from_degrees(lon=self.new_lon, lat=self.new_lat)

        # 4) Initialize the regridder and store it as an instance attribute
        self.regridder = ConservativeRegridder(source_grid, target_grid)

    def regrid(
        self,
        data: torch.Tensor,      # Shape (..., lat, lon)
    ) -> torch.Tensor:
        """
        Regrids the input data from the source grid to the target grid.

        Args:
            data: The tensor to be regridded. The last two dimensions must
                  match the source grid's (lat, lon) dimensions.

        Returns:
            A new tensor with the data regridded to the target grid.
        """
        return self.regridder.regrid_tensor(data)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """A callable alias for the `regrid` method."""
        return self.regrid(data)


if __name__ == "__main__":

    inp = torch.tensor(data) # data with shape (var, lat, lon)

    lat_start = -90.0 + 1.875
    lat_stop  =  90.0 - 1.875
    src_lat = torch.linspace(lat_start, lat_stop, steps=48, dtype=torch.float32)
    src_lon = torch.arange(0.0, 360.0, step=3.75, dtype=torch.float32)

    # `inp` of shape (1, 48, 96), for instance
    out = regrid(src_lat, src_lon, inp, ddeg_out=5.6)