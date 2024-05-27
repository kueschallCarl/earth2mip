# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Routines to save domains to a netCDF file
"""
from typing import Iterable, List

import numpy as np
import torch
import sys
import xarray as xr

from typing import List
import sys
from earth2mip.weather_events import Diagnostic, Domain

import earth2mip.grid
import logger
from earth2mip import geometry
from earth2mip.diagnostics import Diagnostics, DiagnosticTypes
from earth2mip.weather_events import Domain

__all__ = ["initialize_netcdf", "update_netcdf"]


def _assign_lat_attributes(nc_variable):
    nc_variable.units = "degrees_north"
    nc_variable.standard_name = "latitude"
    nc_variable.long_name = "latitude"


def _assign_lon_attributes(nc_variable):
    nc_variable.units = "degrees_east"
    nc_variable.standard_name = "longitude"
    nc_variable.long_name = "longitude"


def init_dimensions(domain: Domain, group, grid: earth2mip.grid.LatLonGrid):

    lat = np.array(grid.lat)
    lon = np.array(grid.lon)

    if domain.type == "CWBDomain":
        cwb_path = "/lustre/fsw/sw_climate_fno/nbrenowitz/2023-01-24-cwb-4years.zarr"
        lat = xr.open_zarr(cwb_path)["XLAT"][:, 0]
        lon = xr.open_zarr(cwb_path)["XLONG"][0, :]
        nlat = lat.size
        nlon = lon.size
        group.createDimension("lat", nlat)
        group.createDimension("lon", nlon)
        v = group.createVariable("lat", np.float32, ("lat"))
        _assign_lat_attributes(v)
        v = group.createVariable("lon", np.float32, ("lon"))
        _assign_lon_attributes(v)

        group["lat"][:] = lat
        group["lon"][:] = lon
    elif domain.type == "Window":
        lat_sl, lon_sl = geometry.get_bounds_window(domain, lat, lon)
        group.createVariable("imin", int, ())
        group.createVariable("imax", int, ())
        group.createVariable("jmin", int, ())
        group.createVariable("jmax", int, ())

        group["imin"][:] = lat_sl.start
        group["imax"][:] = lat_sl.stop
        group["jmin"][:] = lon_sl.start
        group["jmax"][:] = lon_sl.stop

        nlat = np.r_[lat_sl].size
        nlon = np.r_[lon_sl].size
        group.createDimension("lat", nlat)
        group.createDimension("lon", nlon)
        v = group.createVariable("lat", np.float32, ("lat"))
        _assign_lat_attributes(v)
        v = group.createVariable("lon", np.float32, ("lon"))
        _assign_lon_attributes(v)

        group["lat"][:] = lat[lat_sl]
        group["lon"][:] = lon[lon_sl]

    elif domain.type == "MultiPoint":
        assert len(domain.lat) == len(  # noqa
            domain.lon
        ), "Lat and Lon arrays must be of same size!"
        group.createDimension("npoints", len(domain.lon))
        v = group.createVariable("lat_point", np.float32, ("npoints"))
        _assign_lat_attributes(v)

        v = group.createVariable("lon_point", np.float32, ("npoints"))
        _assign_lon_attributes(v)

        for diagnostic in domain.diagnostics:
            group.createDimension("n_channel", len(diagnostic.channels))
        group["lat_point"][:] = domain.lat
        group["lon_point"][:] = domain.lon
    else:
        raise NotImplementedError(f"domain type {domain.type} not supported")
    return


def initialize_netcdf(nc, domains, output_grid, n_ensemble, device):
    # Create the global group
    global_group = nc.createGroup('global')

    # Initialize dimensions
    nc.createDimension('time', None)
    nc.createDimension('ensemble', n_ensemble)
    global_group.createDimension('lat', len(output_grid.lat))
    global_group.createDimension('lon', len(output_grid.lon))

    # Create coordinate variables
    times = nc.createVariable('time', 'f8', ('time',))
    ensemble = nc.createVariable('ensemble', 'i4', ('ensemble',))
    latitudes = global_group.createVariable('lat', 'f4', ('lat',))
    longitudes = global_group.createVariable('lon', 'f4', ('lon',))

    # Assign attributes
    latitudes.units = 'degrees_north'
    longitudes.units = 'degrees_east'

    latitudes[:] = output_grid.lat
    longitudes[:] = output_grid.lon

    # Initialize data variables for each diagnostic
    diagnostics = []
    for domain in domains:
        for diagnostic in domain.diagnostics:
            for channel in diagnostic.channels:
                var_name = f"{channel}_{diagnostic.type}"
                if var_name not in global_group.variables:
                    var = global_group.createVariable(var_name, 'f4', ('time', 'ensemble', 'lat', 'lon',), zlib=True)
                    diagnostics.append(var)

    return diagnostics

def update_netcdf(
    data: torch.Tensor,
    total_diagnostics: List[List[Diagnostic]],
    domains: List[Domain],
    batch_id,
    time_count,
    grid: earth2mip.grid.LatLonGrid,
    channel_names_of_data: List[str],
):
    logger.debug("Entering update_netcdf")
    assert len(total_diagnostics) == len(domains), (total_diagnostics, domains)
    lat, lon = grid.lat, grid.lon
    logger.debug(f"Lat size: {lat.size}, Lon size: {lon.size}")

    for d_index, domain in enumerate(domains):
        domain_diagnostics = total_diagnostics[d_index]
        for diagnostic in domain_diagnostics:
            for i, channel in enumerate(channel_names_of_data):
                var_name = f"{channel}_{diagnostic.type}"
                if var_name in data.variables:
                    var_data = data.variables[var_name][:]
                    logger.debug(f"Writing to variable: {var_name}, Data shape: {var_data.shape}")
                    data.variables[var_name][time_count, batch_id, :, :] = var_data[:, i, :, :].cpu().numpy()
                    logger.debug(f"Written to variable: {var_name}, Data shape: {data.variables[var_name][time_count, batch_id, :, :].shape}")
    logger.debug("Exiting update_netcdf")