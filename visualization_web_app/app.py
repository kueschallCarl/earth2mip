from flask import Flask, jsonify, send_from_directory, request, session
import numpy as np
import config
import inference
import time
import pandas as pd

app = Flask(__name__)
app.secret_key = '032849783209458u092509234850809'
inference_status = {'status': 'idle'}

# Load country coordinates
country_coords = pd.read_csv('static/country-coord.csv')
country_coords.set_index('country', inplace=True)

def preprocess_xarray_data(ds, channel, ensemble_member_index=0, region_select="global", longitude=None, latitude=None, region_size=0.5, time_index=0, max_points=250000):
    lons = ds.lon.values
    lats = ds.lat.values
    time_steps = ds[channel].shape[1]  # Get the number of time steps available
    if time_index >= time_steps:
        raise IndexError(f"Time index {time_index} is out of bounds for available time steps {time_steps}")

    data = ds[channel][ensemble_member_index, time_index].values  # Select the appropriate time slice and ensemble member
    if channel == "t2m":
        data = data - 273.15  # Convert to Celsius if needed (this assumes all channels need this conversion)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_grid_flat = lon_grid.flatten()
    lat_grid_flat = lat_grid.flatten()
    data_flat = data.flatten()

    if region_select == "country":
        mask = (lat_grid_flat >= latitude - region_size / 2) & (lat_grid_flat <= latitude + region_size / 2) & \
               (lon_grid_flat >= longitude - region_size / 2) & (lon_grid_flat <= longitude + region_size / 2)
        lon_grid_flat = lon_grid_flat[mask]
        lat_grid_flat = lat_grid_flat[mask]
        data_flat = data_flat[mask]
    elif region_select == "custom":
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

@app.route('/reset_status', methods=['POST'])
def reset_status():
    global inference_status
    inference_status['status'] = 'idle'
    return '', 200

@app.route('/get_status')
def get_status():
    return jsonify(inference_status)

@app.route('/set_status', methods=['POST'])
def set_status():
    global inference_status
    data = request.get_json()
    inference_status['status'] = data['status']
    return '', 200

@app.route('/data/<region_select>')
def data(region_select):
    # Check for custom region data in session
    custom_region_data = session.get('custom_region_data')

    longitude = None
    latitude = None
    region_size = 0.5
    time_index = int(request.args.get('time', 0))
    ensemble_member_index = int(request.args.get('ensemble', 0))
    channel = request.args.get('channel', 't2m')

    if region_select == "custom" or region_select == "country":
        longitude = custom_region_data['longitude']
        latitude = custom_region_data['latitude']
        region_size = custom_region_data['region_size']
    
    config_dict = session.get('config_dict', {})
    ds = inference.load_dataset_from_inference_output(config_dict=config_dict)
    ds_json_ready = preprocess_xarray_data(ds, channel, ensemble_member_index, region_select, longitude, latitude, region_size, time_index)
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

    if region_select == 'custom':
        longitude = data.get('longitude')
        latitude = data.get('latitude')
        region_size = data.get('regionSize')
        session['custom_region_data'] = {
            'longitude': longitude,
            'latitude': latitude,
            'region_size': region_size
        }
    elif region_select == 'country':
        country = data.get('country')
        if country in country_coords.index:
            longitude = country_coords.at[country, 'lon']
            latitude = country_coords.at[country, 'lat']
            region_size = 10  # Example region size for a country, adjust as needed
            session['custom_region_data'] = {
                'longitude': longitude,
                'latitude': latitude,
                'region_size': region_size
            }
        else:
            return jsonify({'error': 'Country not found in the CSV file'}), 400

    config_dict = inference.parse_config(config_text)
    session['config_dict'] = config_dict

    if not skip_inference:
        # Set status to started
        inference_status['status'] = 'Inference started, this can take a minute...'
        print("Inference started")
        inference.run_inference(config_dict)
        print("Inference completed")
        # Set status to completed
        inference_status['status'] = 'Inference completed'

    ds = inference.load_dataset_from_inference_output(config_dict=config_dict)

    return '', 200

@app.route('/get_config')
def get_config():
    config_dict = session.get('config_dict', {})
    return jsonify(config_dict)

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
