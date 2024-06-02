# app.py

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import xarray as xr
import numpy as np
import config
import inference
from utils import update_plot, update_plot_placeholders


ds = config.DS

# Initialize Dash app
app = dash.Dash(__name__)

# Sample configuration as a string
sample_config = config.CONFIG_SAMPLE_TEXT

# Layout of the app
app.layout = html.Div(style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'justifyContent': 'center', 'height': '100vh', 'width': '100vw'}, children=[
    dcc.Textarea(
        id='config-input',
        value=sample_config,
        style={'width': '80%', 'height': '200px', 'margin-bottom': '20px'}
    ),
    html.Button('Apply Config', id='apply-config-button', n_clicks=0),
    dcc.Graph(id='globe-plot', style={'width': '80vw', 'height': '60vh', 'margin-top': '20px'}),
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
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'margin-top': '10px'}, children=[
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

# Callback to handle the Apply Config button click
@app.callback(
    Output('globe-plot', 'figure'),
    [Input('apply-config-button', 'n_clicks')],
    [State('config-input', 'value'),
     State('time-slider', 'value'),
     State('initial-condition-dropdown', 'value')]
)
def apply_config(n_clicks, config_input, selected_time, initial_condition):
    config_dict = inference.parse_config(config_input)
    if config_dict and n_clicks!=0:
        # Update global dataset or settings if needed based on config
        inference.run_inference(config_dict)
        print(f"Inference finished, loading ds from inference output")
        ds = inference.load_dataset_from_inference_output(config_dict=config_dict)
        return update_plot(ds, config_input, selected_time, initial_condition)
    else:
        print(f"Didn't run inference, loading ds from config")
        ds = config.DS
        return update_plot_placeholders(ds, config_input, selected_time, initial_condition)


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
