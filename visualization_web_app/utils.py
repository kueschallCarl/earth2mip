import dash
from dash import dcc, html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import xarray as xr
import numpy as np
import json
import config




def update_plot_placeholders(ds, config_input, selected_time, initial_condition):
    # Use the configuration to adjust the simulation (placeholder)
    # For simplicity, we use the existing dataset (ds) for the example

    lons = ds.lon.values
    lats = ds.lat.values
    data = ds.t2m[0, selected_time, :, :].values.flatten()
    
    print(f"Dataset in 'update_plot': {ds}")
    fig = go.Figure(go.Choropleth(
        z=data,
        locations=ds.lon.values,
        locationmode='geojson-id',
        geojson='https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json',
        colorscale='RdBu',
        marker_line_color='darkgray',
        marker_line_width=0.5
    ))

    fig.update_geos(projection_type="orthographic", showcountries=True, showcoastlines=True, showland=True)
    fig.update_layout(title=f"Weather Simulation at Time {selected_time}")

    return fig



def update_plot(ds, config_input, selected_time, initial_condition):
    # Extract a subset of the data for performance reasons
    lon_step = 10  # Adjust this step size for performance vs. resolution
    lat_step = 10  # Adjust this step size for performance vs. resolution

    lons = ds.lon.values[::lon_step]
    lats = ds.lat.values[::lat_step]
    data = ds.t2m[0, selected_time, ::lat_step, ::lon_step].values.flatten()
    print(f"Dataset in 'update_plot': {ds}")

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_grid_flat = lon_grid.flatten()
    lat_grid_flat = lat_grid.flatten()

    locations = [f"point{i}" for i in range(len(lon_grid_flat))]

    fig = go.Figure(go.Choropleth(
        z=data,
        locations=locations,
        locationmode='geojson-id',
        geojson='https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json',
        colorscale='RdBu',
        marker_line_color='darkgray',
        marker_line_width=0.5
    ))

    fig.update_geos(projection_type="orthographic", showcountries=True, showcoastlines=True, showland=True)
    fig.update_layout(title=f"Weather Simulation at Time {selected_time}")
    print(f"Finished updating the plot")
    return fig