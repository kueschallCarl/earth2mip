from flask import Flask, jsonify, send_from_directory, request
import numpy as np
import config
import inference

app = Flask(__name__)

def preprocess_xarray_data(ds, region_select, max_points=250000):
    lons = ds.lon.values
    lats = ds.lat.values
    data = ds.t2m[0,0].values
    data_celsius = data - 273.15  # Convert to Celsius

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_grid_flat = lon_grid.flatten()
    lat_grid_flat = lat_grid.flatten()
    data_flat = data_celsius.flatten()

    if region_select == "germany-only":
        mask = (lat_grid_flat >= 47.2) & (lat_grid_flat <= 55.0) & \
               (lon_grid_flat >= 5.8) & (lon_grid_flat <= 15.0)
        lon_grid_flat = lon_grid_flat[mask]
        lat_grid_flat = lat_grid_flat[mask]
        data_flat = data_flat[mask]

    total_points = lon_grid_flat.size
    step = max(1, int(np.ceil(total_points / max_points)))

    downsampled_indices = np.arange(0, total_points, step)

    data_json = {
        'lons': lon_grid_flat[downsampled_indices].tolist(),
        'lats': lat_grid_flat[downsampled_indices].tolist(),
        'values': data_flat[downsampled_indices].tolist()
    }
    return data_json

@app.route('/data/<region_select>')
def data(region_select):
    ds = inference.load_dataset_from_inference_output(config_dict=inference.parse_config(config.CONFIG_SAMPLE_TEXT))
    print(f"ds shape: {ds.dims}")
    ds_json_ready = preprocess_xarray_data(ds, region_select)
    return jsonify(ds_json_ready)

@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    data = request.get_json()
    config_text = data['configText']
    region_select = data['regionSelect']
    skip_inference = data['skipInference']

    print("Config Text:", config_text)
    print("Region Selected:", region_select)
    print("Skip Inference:", skip_inference)

    config_dict = inference.parse_config(config_text)

    if not skip_inference:
        inference.run_inference(config_dict)

    ds = inference.load_dataset_from_inference_output(config_dict=config_dict)

    return '', 200

@app.route('/cesium')
def cesium():
    return send_from_directory('', 'cesium.html')

@app.route('/geojson/<path:filename>')
def geojson(filename):
    return send_from_directory('geojson', filename)

@app.route('/')
def index():
    return send_from_directory('', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
