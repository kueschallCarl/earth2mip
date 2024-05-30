import dash
from dash import dcc, html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import xarray as xr
import numpy as np

# Sample data - replace with your dataset
ds = xr.Dataset({
    't2m': (('ensemble', 'time', 'lat', 'lon'), np.random.rand(4, 10, 721, 1440)),
    'lon': ('lon', np.linspace(0, 359.75, 1440)),
    'lat': ('lat', np.linspace(90, -90, 721)),
    'time': ('time', np.arange(10))
})

# Initialize Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div(style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'justify-content': 'center', 'height': '100vh', 'width': '100vw'}, children=[
    dcc.Graph(id='globe-plot', style={'width': '80vw', 'height': '80vh'}),
    html.Div(style={'width': '80vw'}, children=[
        dcc.Slider(
            id='time-slider',
            min=0,
            max=len(ds.time) - 1,
            step=1,
            value=0,
            marks={i: str(i) for i in range(len(ds.time))},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.Div(style={'display': 'flex', 'justify-content': 'space-between', 'margin-top': '10px'}, children=[
            html.Button('Run Simulation', id='run-button', n_clicks=0),
            dcc.Dropdown(
                id='initial-condition-dropdown',
                options=[{'label': f'Condition {i}', 'value': i} for i in range(1, 6)],
                value=1,
                style={'width': '200px'}
            )
        ])
    ])
])

# Callback to update the plot based on user inputs
@app.callback(
    Output('globe-plot', 'figure'),
    [Input('time-slider', 'value'),
     Input('run-button', 'n_clicks'),
     Input('initial-condition-dropdown', 'value')]
)
def update_plot(selected_time, n_clicks, initial_condition):
    # Example data update based on inputs
    # In practice, this function would run the simulation with the provided initial conditions
    lons = ds.lon.values
    lats = ds.lat.values
    data = ds.t2m[0, selected_time, :, :].values.flatten()

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

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
