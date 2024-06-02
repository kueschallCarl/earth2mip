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
    lon_step = config_input['LON_STEP']  # Adjust this step size for performance vs. resolution
    lat_step = config_input['LAT_STEP']  # Adjust this step size for performance vs. resolution

    # Extract the subset of data based on the step sizes
    lons = ds.lon.values[::lon_step].tolist()
    lats = ds.lat.values[::lat_step].tolist()
    data = ds.t2m[0, selected_time, ::lat_step, ::lon_step].values

    # Convert temperature from Kelvin to Celsius
    data_celsius = data - 273.15

    # Debugging: Print the data
    print(f"Selected time index: {selected_time}")
    print(f"Selected time value: {ds.time.values[selected_time]}")
    print(f"Sample data at selected time (first 5 values): {data[:5, :5]}")
    print(f"Data in Celsius (2D sample):\n{data_celsius[:5, :5]}")
    
    # Check the shape and range of data_celsius
    print(f"Data shape: {data_celsius.shape}")
    print(f"Data range: min={data_celsius.min()}, max={data_celsius.max()}")

    # Create a meshgrid for the coordinates
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_grid_flat = lon_grid.flatten()
    lat_grid_flat = lat_grid.flatten()
    data_flat = data_celsius.flatten()

    # Verify the flattened data
    print(f"Flattened lon grid (first 5 values): {lon_grid_flat[:5]}")
    print(f"Flattened lat grid (first 5 values): {lat_grid_flat[:5]}")
    print(f"Flattened data (first 5 values): {data_flat[:5]}")

    # Create the scattergeo trace for 3D globe visualization
    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lon=lon_grid_flat,
        lat=lat_grid_flat,
        text=data_flat,
        marker=dict(
            size=2,
            color=data_flat,
            colorscale='RdBu_r',
            cmin=data_celsius.min(),
            cmax=data_celsius.max(),
            colorbar=dict(title="Temperature (°C)"),
        )
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