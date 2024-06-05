from flask import Flask, jsonify, send_from_directory, request, session
import numpy as np
import config
import inference

app = Flask(__name__)
app.secret_key = '032849783209458u092509234850809'

def preprocess_xarray_data(ds, region_select, longitude=None, latitude=None, region_size=0.5, time_index=0, max_points=250000):
    lons = ds.lon.values
    lats = ds.lat.values
    data = ds.t2m[0, time_index].values  # Select the appropriate time slice
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
    elif region_select == "custom":
        if longitude is None or latitude is None:
            raise ValueError("Longitude and latitude must be provided for custom region")
        mask = (lat_grid_flat >= latitude - region_size / 2) & (lat_grid_flat <= latitude + region_size / 2) & \
               (lon_grid_flat >= longitude - region_size / 2) & (lon_grid_flat <= longitude + region_size / 2)
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
    # Check for custom region data in session
    custom_region_data = session.get('custom_region_data')

    longitude = None
    latitude = None
    region_size = 0.5
    time_index = int(request.args.get('time', 0))

    if region_select == "custom":
        longitude = custom_region_data['longitude']
        latitude = custom_region_data['latitude']
        region_size = custom_region_data['region_size']
    
    ds = inference.load_dataset_from_inference_output(config_dict=inference.parse_config(config.CONFIG_SAMPLE_TEXT))
    ds_json_ready = preprocess_xarray_data(ds, region_select, longitude, latitude, region_size, time_index)
    return jsonify(ds_json_ready)

@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    data = request.get_json()
    config_text = data['configText']
    region_select = data['regionSelect']
    skip_inference = data['skipInference']
    longitude = None
    latitude = None
    region_size = None

    if region_select == "custom":
        longitude = data.get('longitude')
        latitude = data.get('latitude')
        region_size = data.get('regionSize')
        session['custom_region_data'] = {
            'longitude': longitude,
            'latitude': latitude,
            'region_size': region_size
        }

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
