import dash
from dash import dcc, html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import xarray as xr
import numpy as np
import json
import config




def update_plot(ds, config_input, selected_time, initial_condition):
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