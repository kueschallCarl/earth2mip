import numpy as np
from flask import Flask, jsonify, send_from_directory, request, session
import config
import inference
import pandas as pd

app = Flask(__name__)
app.secret_key = '032849783209458u092509234850809'
inference_status = {'status': 'idle'}

country_coords = pd.read_csv('static/country-coord.csv')
country_coords.set_index('country', inplace=True)

def sigmoid(x):
    z = np.zeros_like(x)
    positive_mask = (x >= 0)
    negative_mask = ~positive_mask
    z[positive_mask] = np.exp(-x[positive_mask])
    z[negative_mask] = np.exp(x[negative_mask])
    result = np.zeros_like(x)
    result[positive_mask] = 1 / (1 + z[positive_mask])
    result[negative_mask] = z[negative_mask] / (1 + z[negative_mask])
    return result

def calculate_wildfire_risk(t2m_data, u10m_data, v10m_data, r50_data):
    # Normalize temperature data from -20 to 50 degrees
    normalized_temp = (t2m_data + 20) / 70
    
    # Calculate risks based on normalized and transformed values
    temp_risk = sigmoid((normalized_temp - 0.7) * 15)  # Emphasize high temperatures
    wind_speed = np.sqrt(u10m_data**2 + v10m_data**2)
    normalized_wind = wind_speed / 20  # Normalize wind speed to [0, 1]
    wind_risk = sigmoid((normalized_wind - 0.5) * 10)  # Emphasize high wind speeds
    dry_risk = sigmoid((1 - r50_data) * 20)  # Emphasize low r50 values
    
    # Weightings for each risk component
    temp_weight = 0.7
    wind_weight = 0.15
    dry_weight = 0.15
    
    # Calculate wildfire risk for each point, ensuring it's between 0 and 1
    wildfire_risk = temp_risk * temp_weight + wind_risk * wind_weight + dry_risk * dry_weight * 100
    return wildfire_risk

def preprocess_xarray_data(ds, channel, ensemble_member_index=0, region_select="global", longitude=None, latitude=None, region_size=0.5, time_index=0, max_points=250000, n_days=7):
    lons = ds.lon.values
    lats = ds.lat.values
    time_steps = ds[channel].shape[1]
    if time_index >= time_steps:
        raise IndexError(f"Time index {time_index} is out of bounds for available time steps {time_steps}")

    data = ds[channel][ensemble_member_index, time_index].values
    if channel == "t2m":
        data = data - 273.15

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_grid_flat = lon_grid.flatten()
    lat_grid_flat = lat_grid.flatten()
    data_flat = data.flatten()

    if region_select == "country" or region_select == "custom":
        mask = (lat_grid_flat >= latitude - region_size / 2) & (lat_grid_flat <= latitude + region_size / 2) & \
               (lon_grid_flat >= longitude - region_size / 2) & (lon_grid_flat <= longitude + region_size / 2)
        lon_grid_flat = lon_grid_flat[mask]
        lat_grid_flat = lat_grid_flat[mask]
        data_flat = data_flat[mask]

    total_points = lon_grid_flat.size
    step = max(1, int(np.ceil(total_points / max_points)))

    downsampled_indices = np.arange(0, total_points, step)

    # Prepare downsampled data
    downsampled_lons = lon_grid_flat[downsampled_indices]
    downsampled_lats = lat_grid_flat[downsampled_indices]
    downsampled_values = data_flat[downsampled_indices]

    # Calculate wildfire risk for all datapoints
    t2m_data = ds.t2m[ensemble_member_index, -n_days:, :, :].values
    u10m_data = ds.u10m[ensemble_member_index, -n_days:, :, :].values
    v10m_data = ds.v10m[ensemble_member_index, -n_days:, :, :].values
    r50_data = ds.r50[ensemble_member_index, -n_days:, :, :].values

    wildfire_risk = calculate_wildfire_risk(t2m_data, u10m_data, v10m_data, r50_data)
    wildfire_risk_flat = wildfire_risk.flatten()

    # Downsample wildfire risk
    downsampled_wildfire_risk = wildfire_risk_flat[downsampled_indices]

    data_json = {
        'lons': downsampled_lons.tolist(),
        'lats': downsampled_lats.tolist(),
        'values': downsampled_values.tolist(),
        'wildfire_risk': downsampled_wildfire_risk.tolist()
    }

    #print(data_json['wildfire_risk'])

    return data_json

@app.route('/data/<region_select>')
def data(region_select):
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
            region_size = 10
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
        inference_status['status'] = 'Inference started, this can take a minute...'
        print("Inference started")
        inference.run_inference(config_dict)
        print("Inference completed")
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
