#!/usr/bin/env python3.10
import tensorflow as tf
from lib import viz
from math import cos, radians
import numpy as np
from skimage import measure
import os
import random
import argparse
import csv
from lib import rawdata
from lib import dataset
from lib import model
from lib import preprocess
import sys
import json   
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation, distance_transform_edt
from lib.perimeter_filter import analyze_perimeter_changes
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from scipy.spatial import distance
import cv2
import requests
from datetime import datetime, timedelta
from shapely.ops import unary_union
import pandas as pd
import glob

#ensure output directories exist
os.makedirs('output/csv', exist_ok=True)

#redirect stdout and stderr to output files
from datetime import datetime

#create timestamp for unique log files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"output/getCSVPredictions_{timestamp}.log"
error_file = f"output/getCSVPredictions_{timestamp}_errors.log"

#redirect stdout to log file
sys.stdout = open(log_file, 'w')
sys.stderr = open(error_file, 'w')

print(f"Starting getCSVPredictions.py at {datetime.now()}")
print(f"Log file: {log_file}")
print(f"Error file: {error_file}")

def loadJSON(api_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(api_url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()

def normalize_fire_name(fire_name):
    return fire_name.strip().replace(' ', '_')

def get_polygon_bounds(coordinates, geom_type):
    all_coords = []
    if geom_type == "Polygon":
        all_coords = coordinates[0]
    elif geom_type == "MultiPolygon":
        for polygon in coordinates:
            all_coords.extend(polygon[0])
    if not all_coords:
        return None
    longs, lats = [coord[0] for coord in all_coords],[coord[1] for coord in all_coords]
    return {'min_lon': min(longs), 'max_lon': max(longs), 'min_lat': min(lats), 'max_lat': max(lats)}

def parse_date(date_str):
    """Parse date string in MMDD format and return datetime object."""
    try:
        #add current year to MMDD format
        current_year = datetime.now().year
        full_date_str = f"{current_year}{date_str}"
        return datetime.strptime(full_date_str, '%Y%m%d')
    except ValueError:
        print(f"Error: Date format should be MMDD, got: {date_str}")
        return None

def find_latest_perimeter_files(fire_path, target_date):
    """Find the two most recent perimeter files before or on the target date."""
    #get all .npy files in the perimeter directory
    pattern = os.path.join(fire_path, "*.npy")
    all_files = glob.glob(pattern)
    
    if not all_files:
        print(f"No .npy files found in {fire_path}")
        return None, None
    
    valid_files = []
    target_year = target_date.year
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        if filename.endswith('.npy') and len(filename) == 8:  # MMDD.npy format
            date_str = filename[:4]  # Extract MMDD
            try:
                #parse as current year first
                file_date = datetime.strptime(f"{target_year}{date_str}", '%Y%m%d')
                
                #if the file date is after target date, try previous year
                if file_date > target_date:
                    file_date = datetime.strptime(f"{target_year-1}{date_str}", '%Y%m%d')
                
                #only include files on or before target date
                if file_date <= target_date:
                    valid_files.append((file_date, file_path))
                    
            except ValueError:
                continue
    
    if not valid_files:
        print(f"No valid perimeter files found before {target_date.strftime('%m%d')}")
        return None, None
    
    #sort by date (most recent first)
    valid_files.sort(key=lambda x: x[0], reverse=True)
    
    print(f"Found {len(valid_files)} perimeter files before or on {target_date.strftime('%m%d')}:")
    for date, path in valid_files[:5]:  # Show first 5
        print(f"  {date.strftime('%m%d')} ({date.strftime('%Y-%m-%d')})")
    
    #return the two most recent
    if len(valid_files) >= 2:
        return valid_files[0], valid_files[1]  # (date, path) tuples
    elif len(valid_files) == 1:
        return valid_files[0], None
    else:
        return None, None

def load_perimeter_file(filepath):
    """Load a binary numpy perimeter file."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    
    try:
        data = np.load(filepath)
        print(f"Loaded {filepath}: shape {data.shape}, dtype {data.dtype}")
        return data.astype(bool)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def resize_if_needed(mask):
    """Resize mask to 1024x1024 if necessary."""
    if mask is None:
        return None
    if mask.shape != (1024, 1024):
        print(f"Warning: Resizing mask from {mask.shape} to (1024, 1024)")
        try:
            from scipy.ndimage import zoom
            zoom_factors = (1024 / mask.shape[0], 1024 / mask.shape[1])
            return zoom(mask.astype(float), zoom_factors, order=0) > 0.5
        except ImportError:
            print("Warning: scipy not available, using basic numpy resize")
            h, w = mask.shape
            new_mask = np.zeros((1024, 1024), dtype=bool)
            scale_h, scale_w = 1024 / h, 1024 / w
            for i in range(1024):
                for j in range(1024):
                    orig_i = min(int(i / scale_h), h - 1)
                    orig_j = min(int(j / scale_w), w - 1)
                    new_mask[i, j] = mask[orig_i, orig_j]
            return new_mask
    return mask.astype(bool)

def run_production_inference(fire_name, date, target_points=10000, model_file="20260101-173820mod"):
    #normalize the fire name consistently
    normalized_fire_name = normalize_fire_name(fire_name)
    print(f"[Production] Loading fire data for {fire_name} (normalized: {normalized_fire_name}) on date {date}...")
    
    try:
        #use normalized name for data loading
        data = rawdata.RawData.load(burnNames=[normalized_fire_name], dates={normalized_fire_name: [date]}, inference=True)
        
        #validate that we have valid data
        if not data or not data.burns or normalized_fire_name not in data.burns:
            print(f"Warning: No valid data found for fire {normalized_fire_name}")
            return None
            
        burn = data.burns[normalized_fire_name]
        if not burn.layers or len(burn.layers) == 0:
            print(f"Warning: No layers found for fire {normalized_fire_name}")
            return None
            
        #debug: Check layer shapes and content
        print(f"Debug: Fire {normalized_fire_name} has {len(burn.layers)} layers")
        for layer_name, layer_data in burn.layers.items():
            valid_count = np.sum(~np.isnan(layer_data))
            print(f"  {layer_name}: shape {layer_data.shape}, valid values: {valid_count}")
            if valid_count == 0:
                print(f"    WARNING: Layer {layer_name} has no valid values!")
                return None
            
        #create dataset with vulnerable pixels around fire perimeter
        test_dataset = dataset.Dataset(data, dataset.Dataset.vulnerablePixels)
        print(f"Created dataset with {len(test_dataset)} total points")
        
        #check if we have any valid points
        if len(test_dataset) == 0:
            print(f"Warning: No vulnerable pixels found for fire {normalized_fire_name} on date {date}")
            return None
            
        #sample points to reduce density while maintaining coverage
        all_points = test_dataset.toList(test_dataset.points)
        if len(all_points) > target_points:
            print(f"Sampling {target_points} points from {len(all_points)} total points (balanced sampling)")
            sampled_points = random.sample(all_points, target_points)
            test_dataset = dataset.Dataset(data, sampled_points)
            
    except Exception as e:
        print(f"Error loading data for {normalized_fire_name} {date}: {e}")
        return None
    
    #load model and preprocessor
    try:
        mod, pp = getModel(model_file, date)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
    #preprocess in inference mode (returns only inputs and ptList)
    try:
        inputs, ptList = pp.process(test_dataset, inference=True)
        
        #validate inputs
        if not inputs or len(inputs) != 2:
            print(f"Error: Invalid inputs generated for {normalized_fire_name}")
            return None
            
        weather_inputs, img_inputs = inputs
        if len(weather_inputs) == 0 or len(img_inputs) == 0:
            print(f"Error: Empty input arrays for {normalized_fire_name}")
            return None
            
    except Exception as e:
        print(f"Error processing data for {normalized_fire_name} {date}: {e}")
        return None
    
    #predict
    predictions = None
    try:
        predictions = mod.predict(inputs).flatten()
        print(f"Generated {len(predictions)} predictions")
            
    except Exception as e:
        print(f"Error making predictions for {normalized_fire_name} {date}: {e}")
        return None
    
    #ensure predictions were generated successfully
    if predictions is None:
        print(f"Error: No predictions generated for {normalized_fire_name} {date}")
        return None
    
    #calculate bounding box for this fire/date
    img_size = (1024, 1024)
    area_m = 1024 * 30
    #find center_lat, center_lon for this fire - use normalized name for file path
    meta_path = os.path.join('training_data', normalized_fire_name, 'center.json')
    print(f"Looking for center.json at: {meta_path}")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            center_lat = meta['center_lat']
            center_lon = meta['center_lon']
    else:
        print(f"Warning: Missing center.json for fire {normalized_fire_name}, skipping")
        return None
        
    lat_meter = 111320
    lon_meter = 111320 * cos(radians(center_lat))
    half_width_deg = (area_m / 2) / lon_meter
    half_height_deg = (area_m / 2) / lat_meter
    min_lat = center_lat - half_height_deg
    max_lat = center_lat + half_height_deg
    min_lon = center_lon - half_width_deg
    max_lon = center_lon + half_width_deg
    
    def pixel_to_latlon(x, y):
        lon = min_lon + (x / (img_size[1] - 1)) * (max_lon - min_lon)
        lat = max_lat - (y / (img_size[0] - 1)) * (max_lat - min_lat)
        return lat, lon
    
    #filter predictions to only include p >= 0.5
    initial_filtered_data = []
    for pt, pred in zip(ptList, predictions):
        if pred >= 0.5:
            burnName, date, (y, x) = pt
            lat, lon = pixel_to_latlon(x, y)
            initial_filtered_data.append({
                'burnName': burnName,
                'date': date,
                'y': y,
                'x': x,
                'lat': lat,
                'lon': lon,
                'predicted_prob': pred
            })
    
    print(f"Initial high-probability predictions (p >= 0.5): {len(initial_filtered_data)} / {len(predictions)}")
    
    if not initial_filtered_data:
        print(f"No high-probability predictions (p >= 0.5) found for {normalized_fire_name} {date}")
        return None
    
    # ==== APPLY PERIMETER FILTERING (SKIP FOR NEW FIRES) ====
    print("\n=== Checking for perimeter-based filtering ===")
    
    #parse date for perimeter file search
    target_date = parse_date(date)
    if target_date is None:
        print(f"Error parsing date {date}, skipping filtering and using all predictions")
        final_filtered_data = initial_filtered_data
        current_mask = None
    else:
        #find perimeter files
        base_path = os.path.join('training_data', normalized_fire_name, 'perims')
        latest_files = find_latest_perimeter_files(base_path, target_date)
        
        if latest_files[0] is None:
            print("Warning: No valid perimeter files found, skipping filtering and using all predictions")
            final_filtered_data = initial_filtered_data
            current_mask = None
        else:
            current_date_info, previous_date_info = latest_files
            current_date_actual, current_file = current_date_info
            
            if previous_date_info is not None:
                #we have both current and previous perimeters - apply filtering
                previous_date_actual, previous_file = previous_date_info
                print(f"Using perimeter files:")
                print(f"  Current: {current_date_actual.strftime('%m%d')} ({current_file})")
                print(f"  Previous: {previous_date_actual.strftime('%m%d')} ({previous_file})")
                
                #load perimeter files
                current_perim = load_perimeter_file(current_file)
                previous_perim = load_perimeter_file(previous_file)
                
                if current_perim is None or previous_perim is None:
                    print("Error: Failed to load perimeter files, skipping filtering and using all predictions")
                    final_filtered_data = initial_filtered_data
                    current_mask = None
                else:
                    #resize if needed
                    current_mask = resize_if_needed(current_perim)
                    previous_mask = resize_if_needed(previous_perim)
                    
                    #analyze perimeter changes to get filter zones (adaptive keep-buffer, same as compareModelAccuracy)
                    true_growth, stable_core, filter_zones, alignment_overlap, regression = analyze_perimeter_changes(
                        current_mask, previous_mask,
                        alignment_buffer_pixels=7,
                        vulnerable_radius=50,
                        keep_buffer_pixels=15,
                        adaptive_keep_buffer=True,
                        buffer_scale=3,
                        min_buffer_radius=5,
                        max_buffer_radius=200,
                    )
                    
                    if filter_zones is None:
                        print("Warning: Failed to generate filter zones, keeping all predictions")
                        final_filtered_data = initial_filtered_data
                    else:
                        #apply filtering - remove predictions that fall in filter zones
                        final_filtered_data = []
                        removed_count = 0
                        
                        for prediction in initial_filtered_data:
                            y, x = int(prediction['y']), int(prediction['x'])
                            
                            #check if point is within image bounds and not in filter zone
                            if 0 <= y < 1024 and 0 <= x < 1024:
                                if not filter_zones[y, x]:  # Keep if NOT in filter zone
                                    final_filtered_data.append(prediction)
                                else:
                                    removed_count += 1
                            else:
                                removed_count += 1  # Remove out-of-bounds points
                        
                        print(f"\nFiltering results:")
                        print(f"  Initial predictions (p >= 0.5): {len(initial_filtered_data)}")
                        print(f"  Removed by filter zones: {removed_count}")
                        print(f"  Final predictions: {len(final_filtered_data)}")
            else:
                #nEW FIRE: Only one perimeter file found - skip post-processing entirely
                print(f"NEW FIRE DETECTED: Only one perimeter file found ({current_date_actual.strftime('%m%d')})")
                print("Skipping perimeter-based filtering for new fire - using all high-probability predictions")
                final_filtered_data = initial_filtered_data
    
    if not final_filtered_data:
        print(f"No predictions remaining after processing for {normalized_fire_name} {date}")
        return None
    
    #output filtered predictions to CSV - use only fire name, no date
    output_csv = f"output/csv/{normalized_fire_name}.csv"
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['burnName', 'date', 'y', 'x', 'lat', 'lon', 'predicted_prob'])
        writer.writeheader()
        writer.writerows(final_filtered_data)
    
    print(f"Filtered predictions saved to {output_csv}")
    print(f"Final high-probability pixels: {len(final_filtered_data)}")
    
    return output_csv
def getModel(weightsFile=None, date=None):
    numWeatherInputs = 9
    if date:
        usedLayers = ['dem','ndvi', 'aspect', 'slope', 'band_2', 'band_3', 'band_4', 'band_5', f'hotspot_{date}']
    else:
        usedLayers = ['dem','ndvi', 'aspect', 'slope', 'band_2', 'band_3', 'band_4', 'band_5']
    AOIRadius = 30
    pp = preprocess.PreProcessor(numWeatherInputs, usedLayers, AOIRadius)
    if weightsFile:
        #fix: Look for model file in the models directory
        fname = os.path.join('models', weightsFile + '.h5')
        if not os.path.exists(fname):
            #fallback: try current directory for backward compatibility
            fname = weightsFile + '.h5'
        print(f"Loading model from: {fname}")
        mod = tf.keras.models.load_model(fname)
    else:
        mod = model.fireCastModel(pp)
    return mod, pp

def process_recent_fires():
    """Process all fires from the last 24 hours"""
    print("Fetching recent fires from NIFC API...")
    
    #nIFC API URL
    api_url = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson"
    try:
        data = loadJSON(api_url)
        print(f"✓ Successfully fetched data from NIFC API")
    except Exception as e:
        print(f"✗ Error fetching data from NIFC API: {e}")
        return
    
    if data['type'] != 'FeatureCollection':
        print("✗ Error: Expected FeatureCollection")
        return
    
    print(f"✓ Processing {len(data['features'])} features from NIFC...")
    
    #alaska bounds to filter out Alaskan fires
    ALASKA_BOUNDS = {
        'min_lat': 51.2,
        'max_lat': 71.4,
        'min_lon': -173.0,  # 173°E = -173°W
        'max_lon': -130.0
    }
    
    processed_count = 0
    recent_fires_count = 0
    alaskan_fires_count = 0
    small_fires_count = 0
    missing_data_count = 0
    current_time = datetime.now()
    
    print(f"Current time: {current_time}")
    print(f"Looking for fires updated within last 24 hours...")
    
    for i, feature in enumerate(data['features']):
        try:
            geometry = feature['geometry']
            properties = feature['properties']
            bounds = get_polygon_bounds(geometry['coordinates'], geometry['type'])
            if not bounds:
                continue
            
            #get basic fire info
            if 'poly_IncidentName' in properties:
                incident_name = properties['poly_IncidentName']
            else:
                print(f"  Feature {i+1}: No incident name, skipping")
                continue
            
            #check date
            if 'poly_DateCurrent' in properties:
                dt = datetime.fromtimestamp(properties['poly_DateCurrent'] / 1000)
                time_diff = current_time - dt
                hours_ago = time_diff.total_seconds() / 3600
                print(f"  {incident_name}: Last updated {hours_ago:.1f} hours ago ({dt})")
                if time_diff.total_seconds() > 24 * 3600:  # Skip if older than 24 hours
                    continue
                recent_fires_count += 1
                date_str = dt.strftime('%m%d')  # MMDD format for our system
            else:
                print(f"  {incident_name}: No date info, skipping")
                continue
            
            #filter out Alaskan fires
            center_lon = (bounds['min_lon'] + bounds['max_lon']) / 2
            center_lat = (bounds['min_lat'] + bounds['max_lat']) / 2
            
            #check if fire is within Alaska bounds
            if (ALASKA_BOUNDS['min_lat'] <= center_lat <= ALASKA_BOUNDS['max_lat'] and 
                ALASKA_BOUNDS['min_lon'] <= center_lon <= ALASKA_BOUNDS['max_lon']):
                print(f"  ✗ Skipping Alaskan fire: {incident_name} (lat: {center_lat:.3f}, lon: {center_lon:.3f})")
                alaskan_fires_count += 1
                continue
            
            #filter out fires smaller than 100 acres
            if 'poly_Acres_AutoCalc' in properties:
                acres = properties['poly_Acres_AutoCalc']
                if acres is None or acres < 100:
                    print(f"  ✗ Skipping {incident_name}: {acres} acres (below 100-acre minimum)")
                    small_fires_count += 1
                    continue
            else:
                print(f"  ✗ Skipping {incident_name}: no acreage data")
                missing_data_count += 1
                continue
            
            print(f"  ✓ Recent fire candidate: {incident_name} ({acres} acres, {hours_ago:.1f}h ago)")
            
            #check if this fire exists in training_data directory
            #use the normalize_fire_name function consistently
            normalized_fire_name = normalize_fire_name(incident_name)
            nifc_fire_path = os.path.join('training_data', normalized_fire_name)
            
            if not os.path.exists(nifc_fire_path):
                print(f"  ✗ {incident_name} (normalized: {normalized_fire_name}): not found in training_data directory ({nifc_fire_path})")
                #list available fires in training_data for debugging
                if os.path.exists('training_data'):
                    available_fires = [f for f in os.listdir('training_data') if os.path.isdir(os.path.join('training_data', f))]
                    print(f"    Available fires in training_data: {available_fires[:5]}{'...' if len(available_fires) > 5 else ''}")
                continue
            
            #check if perimeter file exists for this date
            perim_file = os.path.join(nifc_fire_path, 'perims', f'{date_str}.npy')
            if not os.path.exists(perim_file):
                print(f"  ✗ {normalized_fire_name}: no perimeter file for date {date_str} ({perim_file})")
                #list available dates for this fire
                perims_dir = os.path.join(nifc_fire_path, 'perims')
                if os.path.exists(perims_dir):
                    available_dates = [f.replace('.npy', '') for f in os.listdir(perims_dir) if f.endswith('.npy')]
                    print(f"    Available dates: {available_dates}")
                continue
            
            print(f"  ✓ All checks passed for {normalized_fire_name}, running prediction...")
            
            #run prediction for this fire - pass the original incident_name, function will normalize it
            try:
                result = run_production_inference(incident_name, date_str, target_points=10000)
                if result:
                    processed_count += 1
                    print(f"  ✓ Successfully processed {normalized_fire_name}")
                else:
                    print(f"  ✗ Failed to process {normalized_fire_name}")
            except Exception as e:
                print(f"  ✗ Error processing {normalized_fire_name}: {e}")
                continue
                
        except Exception as e:
            print(f"✗ Error processing feature {i+1}: {e}")
            continue
    
    print(f"\n=== SUMMARY ===")
    print(f"Total features processed: {len(data['features'])}")
    print(f"Recent fires (< 24h): {recent_fires_count}")
    print(f"Alaskan fires (excluded): {alaskan_fires_count}")
    print(f"Small fires (excluded): {small_fires_count}")
    print(f"Missing data (excluded): {missing_data_count}")
    print(f"Successfully processed: {processed_count}")
    print(f"Processing complete!")

if __name__ == "__main__":
    process_recent_fires()
