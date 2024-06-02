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
    lon_step = 4  # Adjust this step size for performance vs. resolution
    lat_step = 4  # Adjust this step size for performance vs. resolution

    lons = ds.lon.values[::lon_step]
    lats = ds.lat.values[::lat_step]
    data = ds.t2m[0, selected_time, ::lat_step, ::lon_step].values
    print(f"Dataset in 'update_plot': {ds}")

    fig = go.Figure(go.Heatmap(
        z=data,
        x=lons,
        y=lats,
        colorscale='RdBu'
    ))

    fig.update_geos(
        projection_type="orthographic",
        showcountries=True,
        showcoastlines=True,
        showland=True,
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