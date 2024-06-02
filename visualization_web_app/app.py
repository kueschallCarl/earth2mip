from flask import Flask, jsonify, send_from_directory
import numpy as np
import config
import inference

app = Flask(__name__)

def preprocess_xarray_data(ds):
    lons = ds.lon.values
    lats = ds.lat.values
    data = ds.t2m[0,0].values
    data_celsius = data - 273.15  # Convert to Celsius

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_grid_flat = lon_grid.flatten()
    lat_grid_flat = lat_grid.flatten()
    data_flat = data_celsius.flatten()

    data_json = {
        'lons': lon_grid_flat.tolist(),
        'lats': lat_grid_flat.tolist(),
        'values': data_flat.tolist()
    }
    return data_json

@app.route('/data')
def data():
    ds = inference.load_dataset_from_inference_output(config_dict=inference.parse_config(config.CONFIG_SAMPLE_TEXT))
    ds_json_ready = preprocess_xarray_data(ds)
    return jsonify(ds_json_ready)

@app.route('/')
def index():
    return send_from_directory('', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
