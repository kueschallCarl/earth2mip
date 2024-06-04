from flask import Flask, jsonify, send_from_directory, request
import numpy as np
import config
import inference

app = Flask(__name__)

def preprocess_xarray_data(ds, max_points=100000):
    lons = ds.lon.values
    lats = ds.lat.values
    data = ds.t2m[0,0].values
    data_celsius = data - 273.15  # Convert to Celsius

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_grid_flat = lon_grid.flatten()
    lat_grid_flat = lat_grid.flatten()
    data_flat = data_celsius.flatten()

    total_points = lon_grid_flat.size
    step = max(1, int(np.ceil(total_points / max_points)))

    downsampled_indices = np.arange(0, total_points, step)

    data_json = {
        'lons': lon_grid_flat[downsampled_indices].tolist(),
        'lats': lat_grid_flat[downsampled_indices].tolist(),
        'values': data_flat[downsampled_indices].tolist()
    }
    return data_json

@app.route('/data')
def data():
    ds = inference.load_dataset_from_inference_output(config_dict=inference.parse_config(config.CONFIG_SAMPLE_TEXT))
    ds_json_ready = preprocess_xarray_data(ds)
    return jsonify(ds_json_ready)

@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    data = request.get_json()
    config_text = data['configText']
    region_select = data['regionSelect']
    skip_inference = data['skipInference']

    # You can use the config_text and region_select variables to set up your simulation
    # For example, you might modify the configuration here before starting the simulation
    # For demonstration, just printing the values
    print("Config Text:", config_text)
    print("Region Selected:", region_select)
    print("Skip Inference:", skip_inference)

    # Save the config or process it as needed
    # This example just re-parses the config and starts the dataset loading
    config_dict = inference.parse_config(config_text)

    # Conditionally run inference
    if not skip_inference:
        inference.run_inference(config_dict)

    ds = inference.load_dataset_from_inference_output(config_dict=config_dict)

    # Optionally, you might want to save this to a global variable or session
    # For simplicity, we'll assume the dataset is always accessible

    return '', 200

@app.route('/cesium')
def cesium():
    return send_from_directory('', 'cesium.html')

@app.route('/')
def index():
    return send_from_directory('', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
