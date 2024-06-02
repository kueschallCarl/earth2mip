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
    lon_step = config.LON_STEP  # Adjust this step size for performance vs. resolution
    lat_step = config.LAT_STEP  # Adjust this step size for performance vs. resolution

    lons = ds.lon.values[::lon_step]
    lats = ds.lat.values[::lat_step]
    data = ds.t2m[0, selected_time, ::lat_step, ::lon_step].values

    # Convert temperature from Kelvin to Celsius
    data_celsius = data - 273.15

    print(f"Dataset in 'update_plot': {ds}")

    # Create a meshgrid for the coordinates
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_grid_flat = lon_grid.flatten()
    lat_grid_flat = lat_grid.flatten()
    data_flat = data_celsius.flatten()

    # Create scattergeo trace for 3D globe visualization
    fig = go.Figure(go.Scattergeo(
        lon=lon_grid_flat,
        lat=lat_grid_flat,
        text=data_flat,
        marker=dict(
            color=data_flat,
            colorscale='RdBu_r',
            colorbar=dict(title="Temperature (°C)"),
            size=2,
            opacity=0.7
        ),
        mode='markers'
    ))

    fig.update_geos(
        projection_type="orthographic",
        showcountries=True,
        showcoastlines=True,
        showland=True,
        landcolor='rgb(243, 243, 243)',
        oceancolor='rgb(204, 204, 255)',
        showocean=True,
    )

    fig.update_layout(
        title=f"Weather Simulation at Time {selected_time}",
        geo=dict(
            lataxis=dict(range=[-90, 90]),
            lonaxis=dict(range=[-180, 180])
        )
    )

    print(f"Finished updating the plot")
    return fig