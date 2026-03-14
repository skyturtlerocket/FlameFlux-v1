#!/usr/bin/env python3.10
import tensorflow as tf
from lib import viz
from math import cos, radians
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
import sys
import os
import random
import argparse
import csv
from lib import rawdata
from lib import dataset
from lib import model
from lib import preprocess
import json
from scipy.ndimage import binary_fill_holes
import alphashape
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from scipy.spatial import distance
# from concave_hull import concave_hull
import cv2

#ensure output directory exists
os.makedirs('output', exist_ok=True)

#redirect stdout and stderr to output files
import sys
from datetime import datetime

#create timestamp for unique log files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"output/runProduction_{timestamp}.log"
error_file = f"output/runProduction_{timestamp}_errors.log"

#redirect stdout to log file
sys.stdout = open(log_file, 'w')
sys.stderr = open(error_file, 'w')

print(f"Starting runProduction.py at {datetime.now()}")
print(f"Log file: {log_file}")
print(f"Error file: {error_file}")


def run_production_inference(fire_name, date, target_points=10000, model_file="20200903-193223mod", all_predictions=None):
    print(f"[Production] Loading fire data for {fire_name} on date {date}...")
    
    try:
        #load data for the specified fire and date in inference mode
        data = rawdata.RawData.load(burnNames=[fire_name], dates={fire_name: [date]}, inference=True)
        
        #validate that we have valid data
        if not data or not data.burns or fire_name not in data.burns:
            print(f"Warning: No valid data found for fire {fire_name}")
            return None
            
        burn = data.burns[fire_name]
        if not burn.layers or len(burn.layers) == 0:
            print(f"Warning: No layers found for fire {fire_name}")
            return None
            
        #debug: Check layer shapes and content
        print(f"Debug: Fire {fire_name} has {len(burn.layers)} layers")
        for layer_name, layer_data in burn.layers.items():
            valid_count = np.sum(~np.isnan(layer_data))
            print(f"  {layer_name}: shape {layer_data.shape}, valid values: {valid_count}")
            if valid_count == 0:
                print(f"    WARNING: Layer {layer_name} has no valid values!")
                #debug the problematic layer
                print(f"    Debug {layer_name}: min={np.min(layer_data)}, max={np.max(layer_data)}")
                print(f"    Debug {layer_name}: unique values: {np.unique(layer_data)[:10]}")  # First 10 unique values
                print(f"    Debug {layer_name}: NaN count: {np.sum(np.isnan(layer_data))}")
                print(f"    Debug {layer_name}: Inf count: {np.sum(np.isinf(layer_data))}")
                return None
            
        #create dataset with vulnerable pixels around fire perimeter
        test_dataset = dataset.Dataset(data, dataset.Dataset.vulnerablePixels)
        print(f"Created dataset with {len(test_dataset)} total points")
        
        #debug: Check perimeter data
        if len(test_dataset) == 0:
            print(f"Debug: Checking why no vulnerable pixels found for {fire_name}")
            day = data.getDay(fire_name, date)
            print(f"  Starting perimeter shape: {day.startingPerim.shape}")
            print(f"  Starting perimeter sum: {np.sum(day.startingPerim)}")
            print(f"  Starting perimeter min/max: {np.min(day.startingPerim)}/{np.max(day.startingPerim)}")
            print(f"  Starting perimeter unique values: {np.unique(day.startingPerim)}")
            
            #check if perimeter file exists
            perim_path = f"training_data/{fire_name}/perims/{date}.npy"
            print(f"  Perimeter file exists: {os.path.exists(perim_path)}")
            
            #try to understand the vulnerable pixels calculation
            startingPerim = day.startingPerim
            kernel = np.ones((3,3))
            radius = dataset.Dataset.VULNERABLE_RADIUS
            its = int(round((2*(radius)**2)**.5))
            print(f"  Dilating with radius {radius}, iterations {its}")
            dilated = cv2.dilate(startingPerim, kernel, iterations=its)
            border = dilated - startingPerim
            print(f"  Dilated sum: {np.sum(dilated)}")
            print(f"  Border sum: {np.sum(border)}")
            ys, xs = np.where(border)
            print(f"  Border pixels found: {len(ys)}")
        
        #check if we have any valid points
        if len(test_dataset) == 0:
            print(f"Warning: No vulnerable pixels found for fire {fire_name} on date {date}")
            return None
            
        #sample points to reduce density while maintaining coverage
        all_points = test_dataset.toList(test_dataset.points)
        if len(all_points) > target_points:
            print(f"Sampling {target_points} points from {len(all_points)} total points (balanced sampling)")
            sampled_points = random.sample(all_points, target_points)
            test_dataset = dataset.Dataset(data, sampled_points)
            
    except Exception as e:
        print(f"Error loading data for {fire_name} {date}: {e}")
        return None
    
    #load model and preprocessor
    try:
        mod, pp = getModel(model_file, date=date)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
    #preprocess in inference mode (returns only inputs and ptList)
    try:
        inputs, ptList = pp.process(test_dataset, inference=True)
        
        if not inputs or len(inputs) < 2:
            print(f"Error: Invalid inputs generated for {fire_name}")
            return None

        weather_inputs, img_inputs = inputs[0], inputs[1]
        if len(weather_inputs) == 0 or len(img_inputs) == 0:
            print(f"Error: Empty input arrays for {fire_name}")
            return None
            
    except Exception as e:
        print(f"Error processing data for {fire_name} {date}: {e}")
        return None
    
    #predict
    predictions = None
    try:
        predictions = mod.predict(inputs).flatten()
        print(f"Generated {len(predictions)} predictions")
        
        #create perimeter visualization
        if all_predictions is None:  # Only for single fire mode
            #ensure the images directory exists
            os.makedirs("output/images", exist_ok=True)
            viz_path = f"output/images/perimeter_viz_{fire_name}_{date}.png"
            create_perimeter_visualization(predictions, ptList, fire_name, date, None, viz_path)
            
            #create perimeter overlay visualization
            overlay_path = f"output/images/perimeter_overlay_{fire_name}_{date}.png"
            create_perimeter_overlay_visualization(predictions, ptList, fire_name, date, None, overlay_path)
            
    except Exception as e:
        print(f"Error making predictions for {fire_name} {date}: {e}")
        return None
    
    #ensure predictions were generated successfully
    if predictions is None:
        print(f"Error: No predictions generated for {fire_name} {date}")
        return None
    
    #store predictions for GeoJSON output if all_predictions dict provided
    if all_predictions is not None:
        # --- Calculate bounding box for this fire/date (same as getData.py) ---
        img_size = (1024, 1024)
        area_m = 1024 * 30
        #find center_lat, center_lon for this fire
        fire_dir = fire_name.replace(' ', '_')
        meta_path = os.path.join('training_data', fire_dir, 'center.json')
        print(f"Looking for center.json at: {meta_path}")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                center_lat = meta['center_lat']
                center_lon = meta['center_lon']
        else:
            print(f"Warning: Missing center.json for fire {fire_name}, skipping GeoJSON output")
            print(f"Available files in {os.path.join('training_data', fire_dir)}: {os.listdir(os.path.join('training_data', fire_dir)) if os.path.exists(os.path.join('training_data', fire_dir)) else 'Directory not found'}")
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
        
        for pt, pred in zip(ptList, predictions):
            burnName, date, (y, x) = pt
            lat, lon = pixel_to_latlon(x, y)
            key = (fire_name, date, (lat, lon))  # (fire_name, date, (lat, lon))
            all_predictions[key] = pred
        
        #create perimeter visualization for all_predictions mode
        #ensure the images directory exists
        os.makedirs("output/images", exist_ok=True)
        viz_path = f"output/images/perimeter_viz_{fire_name}_{date}.png"
        create_perimeter_visualization(predictions, ptList, fire_name, date, pixel_to_latlon, viz_path)
    
    #output predictions to CSV with lat/lon (only if not in all_predictions mode)
    if all_predictions is None:
        output_csv = f"output/predictions_{fire_name}_{date}.csv"
        # --- Calculate bounding box for this fire/date (same as getData.py) ---
        img_size = (1024, 1024)
        area_m = 1024 * 30
        #find center_lat, center_lon for this fire
        fire_dir = fire_name.replace(' ', '_')
        meta_path = os.path.join('training_data', fire_dir, 'center.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                center_lat = meta['center_lat']
                center_lon = meta['center_lon']
        else:
            raise RuntimeError(f"Missing center.json for fire {fire_name} in training_data/{fire_dir}/. Please create this file with center_lat and center_lon.")
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
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['burnName', 'date', 'y', 'x', 'lat', 'lon', 'predicted_prob'])
            for pt, pred in zip(ptList, predictions):
                burnName, date, (y, x) = pt
                lat, lon = pixel_to_latlon(x, y)
                writer.writerow([burnName, date, y, x, lat, lon, pred])
        print(f"Predictions saved to {output_csv}")
        #print summary
        burned = sum(1 for p in predictions if p > 0.5)
        print(f"Predicted burned pixels (prob > 0.5): {burned} / {len(predictions)}")

        # --- Visualization: Save annotated image ---

        # --- GeoJSON perimeter export function ---
        def save_perimeter_geojson_and_overlay(prob_map, pixel_to_latlon, out_path, overlay_path, threshold=0.5):
            """
            Extracts all outer perimeters of the predicted burned area (holes filled), saves as a GeoJSON (MultiPolygon if needed),
            and overlays the perimeter(s) on the probability map image.
            """
            #grid-based outer boundary approach
            burned_pixels = np.argwhere((prob_map > threshold).astype(np.uint8))
            if burned_pixels.shape[0] == 0:
                print('No burned pixels found!')
                return
            pixel_coordinates = [(int(x), int(y)) for y, x in burned_pixels]
            pixel_set = set(pixel_coordinates)
            #step 1: Find boundary pixels
            boundary_pixels = []
            for x, y in pixel_coordinates:
                neighbors = [
                    (x-1, y-1), (x, y-1), (x+1, y-1),
                    (x-1, y),             (x+1, y),
                    (x-1, y+1), (x, y+1), (x+1, y+1)
                ]
                empty_neighbors = sum(1 for neighbor in neighbors if neighbor not in pixel_set)
                if empty_neighbors > 0:
                    boundary_pixels.append((x, y))
            if not boundary_pixels:
                print('No boundary pixels found!')
                return
            #step 2: For each angle from centroid, keep only the farthest pixel
            center_x = sum(p[0] for p in boundary_pixels) / len(boundary_pixels)
            center_y = sum(p[1] for p in boundary_pixels) / len(boundary_pixels)
            import math
            angle_groups = {}
            for x, y in boundary_pixels:
                angle = math.atan2(y - center_y, x - center_x)
                angle_key = round(angle, 1)  # Adjust precision as needed
                if angle_key not in angle_groups:
                    angle_groups[angle_key] = []
                angle_groups[angle_key].append((x, y))
            outermost = []
            for angle, pixels in angle_groups.items():
                if pixels:
                    farthest = max(pixels, key=lambda p: (p[0] - center_x)**2 + (p[1] - center_y)**2)
                    outermost.append(farthest)
            if len(outermost) < 3:
                print('Not enough outermost pixels for a polygon!')
                return
            #sort outermost points by angle from centroid
            center_x = sum(p[0] for p in outermost) / len(outermost)
            center_y = sum(p[1] for p in outermost) / len(outermost)
            import math
            outermost_sorted = sorted(
                outermost,
                key=lambda p: math.atan2(p[1] - center_y, p[0] - center_x)
            )
            #ensure the loop is closed
            if outermost_sorted[0] != outermost_sorted[-1]:
                outermost_sorted.append(outermost_sorted[0])
            #convert to lat/lon
            polygon = [pixel_to_latlon(x, y) for x, y in outermost_sorted]
            geojson_coords = [[lon, lat] for lat, lon in polygon]
            if geojson_coords[0] != geojson_coords[-1]:
                geojson_coords.append(geojson_coords[0])
            geometry = {"type": "Polygon", "coordinates": [geojson_coords]}
            geojson = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": {"threshold": threshold, "method": "grid_outer_boundary_sorted"}
                }]
            }
            with open(out_path, 'w') as f:
                json.dump(geojson, f, indent=2, separators=(',', ': '))
            print(f"Perimeter GeoJSON saved to {out_path}")
            #overlay: plot probability map and draw the sorted grid-based outer boundary
            plt.figure(figsize=(8, 6))
            plt.imshow(prob_map, cmap='hot', interpolation='nearest')
            arr = np.array(outermost_sorted)
            plt.plot(arr[:,0], arr[:,1], color='cyan', linewidth=2)
            plt.colorbar(label='Predicted Burn Probability')
            plt.title('Predicted Burn Probabilities with Sorted Grid-Based Outer Boundary')
            plt.savefig(overlay_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Perimeter overlay visualization saved to: {overlay_path}")
        #render probability map
        prob_maps = viz.renderPredictions(test_dataset, dict(zip(ptList, predictions)), predictions)
        for (burnName, date), prob_map in prob_maps.items():
            plt.figure(figsize=(8, 6))
            
            #load the perimeter mask for background
            perim_path = f"training_data/{burnName}/perims/{date}.npy"
            perimeter_mask = None
            if os.path.exists(perim_path):
                try:
                    perimeter_mask = np.load(perim_path)
                    print(f"Loaded perimeter mask from {perim_path}, shape: {perimeter_mask.shape}")
                except Exception as e:
                    print(f"Error loading perimeter mask: {e}")
            
            #create custom colormap: yellow (low probability) to red (high probability)
            from matplotlib.colors import LinearSegmentedColormap
            colors = ['yellow', 'orange', 'red']
            custom_cmap = LinearSegmentedColormap.from_list('yellow_to_red', colors, N=100)
            
            #print some statistics about the probability values
            print(f"Probability stats for {burnName} {date}:")
            print(f"  Min: {np.min(prob_map):.4f}")
            print(f"  Max: {np.max(prob_map):.4f}")
            print(f"  Mean: {np.mean(prob_map):.4f}")
            print(f"  Non-zero pixels: {np.sum(prob_map > 0)}")
            
            #create masked array to handle unpredicted areas (zeros) as transparent
            import numpy.ma as ma
            masked_prob_map = ma.masked_where(prob_map == 0, prob_map)
            
            #plot perimeter mask as background if available
            if perimeter_mask is not None:
                plt.imshow(perimeter_mask, cmap='gray', interpolation='nearest', alpha=0.3)
            
            #overlay predicted pixels with fire intensity color scale
            plt.imshow(masked_prob_map, cmap=custom_cmap, interpolation='nearest', vmin=0, vmax=1, alpha=0.8)
            plt.colorbar(label='Predicted Burn Probability (Red=High, Yellow=Low)')
            plt.title(f"Predicted Burn Probabilities: {burnName} {date}")
            img_path = f"output/predictions_{burnName}_{date}.png"
            plt.savefig(img_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Prediction visualization saved to: {img_path}")
            # --- Save GeoJSON perimeter and overlay image ---
            geojson_path = f"output/predicted_perimeter_{burnName}_{date}.geojson"
            overlay_path = f"output/predicted_perimeter_{burnName}_{date}_overlay.png"
            save_perimeter_geojson_and_overlay(prob_map, pixel_to_latlon, geojson_path, overlay_path, threshold=0.5)
        return output_csv
    
    return None

def getModel(weightsFile=None, date=None):
    numWeatherInputs = 9
    if date:
        usedLayers = ['dem','ndvi', 'aspect', 'slope', 'band_2', 'band_3', 'band_4', 'band_5', f'hotspot_{date}']
    else:
        usedLayers = ['dem','ndvi', 'aspect', 'slope', 'band_2', 'band_3', 'band_4', 'band_5']
    AOIRadius = 30
    pp = preprocess.PreProcessor(numWeatherInputs, usedLayers, AOIRadius)
    if weightsFile:
        base = weightsFile if weightsFile.endswith('.h5') else weightsFile + '.h5'
        fname = base if base.startswith('models/') else os.path.join('models', base)
        mod = tf.keras.models.load_model(fname)
    else:
        mod = model.fireCastModel(pp)
    return mod, pp


def run_inference_return_predictions(fire_name, date, model_file, target_points=10000, vulnerable_radius=None):
    """
    Run production-style inference for (fire_name, date) with the given model.
    Returns (predictions, ptList) or None on failure. No CSV or visualization output.
    vulnerable_radius: pixels from perimeter to consider (default: Dataset.VULNERABLE_RADIUS, typically 50).
    """
    radius = int(vulnerable_radius) if vulnerable_radius is not None else dataset.Dataset.VULNERABLE_RADIUS
    point_selector = lambda b, d: dataset.Dataset.vulnerablePixels(b, d, radius=radius)
    try:
        data = rawdata.RawData.load(burnNames=[fire_name], dates={fire_name: [date]}, inference=True)
        if not data or not data.burns or fire_name not in data.burns:
            return None
        burn = data.burns[fire_name]
        if not burn.layers or len(burn.layers) == 0:
            return None
        for layer_name, layer_data in burn.layers.items():
            if np.sum(~np.isnan(layer_data)) == 0:
                return None
        test_dataset = dataset.Dataset(data, point_selector)
        if len(test_dataset) == 0:
            return None
        all_points = test_dataset.toList(test_dataset.points)
        if len(all_points) > target_points:
            sampled_points = random.sample(all_points, target_points)
            test_dataset = dataset.Dataset(data, sampled_points)
    except Exception:
        return None
    try:
        mod, pp = getModel(model_file, date=date)
    except Exception:
        return None
    try:
        inputs, ptList = pp.process(test_dataset, inference=True)
        if not inputs or len(inputs) < 2:
            return None
        if len(inputs[0]) == 0 or len(inputs[1]) == 0:
            return None
        predictions = mod.predict(inputs).flatten()
        return (predictions, ptList)
    except Exception:
        return None


def find_outer_perimeter(pixel_coordinates):
    """
    Find the outermost pixels that form the perimeter
    """
    pixel_set = set(pixel_coordinates)
    outer_boundary = []
    
    for x, y in pixel_coordinates:
        #check if this pixel is on the outer edge
        #a pixel is on outer edge if it has empty space in outward directions
        
        #check 8 directions around the pixel
        neighbors = [
            (x-1, y-1), (x, y-1), (x+1, y-1),
            (x-1, y),             (x+1, y),
            (x-1, y+1), (x, y+1), (x+1, y+1)
        ]
        
        #count how many neighbors are NOT fire pixels
        empty_neighbors = sum(1 for neighbor in neighbors if neighbor not in pixel_set)
        
        #if it has empty neighbors, it's a boundary pixel
        if empty_neighbors > 0:
            #additional check: is it on the OUTER boundary?
            #calculate distance from centroid to determine if it's outer edge
            outer_boundary.append((x, y))
    
    return outer_boundary

def filter_outermost_pixels(boundary_pixels):
    """
    From boundary pixels, keep only the outermost ones and ensure proper polygon ordering
    """
    if not boundary_pixels:
        return []
    
    #find centroid
    center_x = sum(p[0] for p in boundary_pixels) / len(boundary_pixels)
    center_y = sum(p[1] for p in boundary_pixels) / len(boundary_pixels)
    
    #group pixels by angle from center
    import math
    
    angle_groups = {}
    for x, y in boundary_pixels:
        angle = math.atan2(y - center_y, x - center_x)
        #round angle to group nearby pixels
        angle_key = round(angle, 1)  # Adjust precision as needed
        
        if angle_key not in angle_groups:
            angle_groups[angle_key] = []
        angle_groups[angle_key].append((x, y))
    
    #for each angle group, keep only the pixel farthest from center
    outermost = []
    for angle, pixels in angle_groups.items():
        if pixels:
            #find pixel farthest from center
            farthest = max(pixels, key=lambda p: (p[0] - center_x)**2 + (p[1] - center_y)**2)
            outermost.append(farthest)
    
    #sort points in counterclockwise order around the centroid
    if len(outermost) >= 3:
        #calculate angles from centroid
        angles = []
        for x, y in outermost:
            angle = math.atan2(y - center_y, x - center_x)
            angles.append(angle)
        
        #sort by angle (counterclockwise)
        sorted_pairs = sorted(zip(outermost, angles), key=lambda pair: pair[1])
        outermost = [point for point, angle in sorted_pairs]
        
        #ensure polygon is closed (first and last points should be the same)
        if outermost[0] != outermost[-1]:
            outermost.append(outermost[0])
    
    return outermost

def save_all_predictions_to_geojson(all_predictions, output_file="output/all_predictions.geojson"):
    """Save all predictions as polygon perimeters to a single GeoJSON file"""
    
    #group predictions by fire and date
    fire_date_predictions = {}
    for (fire_name, date, (lat, lon)), prediction in all_predictions.items():
        key = (fire_name, date)
        if key not in fire_date_predictions:
            fire_date_predictions[key] = []
        fire_date_predictions[key].append((lat, lon, prediction))
    
    features = []
    
    for (fire_name, date), predictions in fire_date_predictions.items():
        try:
            #extract coordinates and predictions
            lats = [p[0] for p in predictions]
            lons = [p[1] for p in predictions]
            pred_values = [p[2] for p in predictions]
            
            if len(lats) > 0:
                #find high-probability pixels (threshold = 0.5)
                threshold = 0.5
                high_prob_indices = [i for i, pred in enumerate(pred_values) if pred > threshold]
                
                if len(high_prob_indices) >= 3:  # Need at least 3 points for a polygon
                    #get coordinates of high-probability pixels
                    high_prob_coords = [(lons[i], lats[i]) for i in high_prob_indices]
                    
                    #find outer perimeter
                    outer_boundary = find_outer_perimeter(high_prob_coords)
                    
                    if len(outer_boundary) >= 3:
                        #filter to outermost pixels
                        outermost = filter_outermost_pixels(outer_boundary)
                        
                        if len(outermost) >= 3:
                            #points are already sorted counterclockwise and closed in filter_outermost_pixels
                            outermost_sorted = outermost
                            
                            #calculate average prediction for this fire/date
                            avg_prediction = np.mean(pred_values)
                            
                            feature = {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [outermost_sorted]
                                },
                                "properties": {
                                    "fire_name": fire_name,
                                    "date": date,
                                    "avg_prediction": float(avg_prediction),
                                    "num_pixels": len(predictions),
                                    "num_high_prob": len(high_prob_indices),
                                    "perimeter_points": len(outermost_sorted),
                                    "threshold": threshold
                                }
                            }
                            features.append(feature)
                        else:
                            print(f"Warning: Not enough outermost pixels for {fire_name} {date} ({len(outermost)} found)")
                    else:
                        print(f"Warning: Not enough boundary pixels for {fire_name} {date} ({len(outer_boundary)} found)")
                else:
                    print(f"Warning: Not enough high-probability pixels for {fire_name} {date} ({len(high_prob_indices)} found)")
                
        except Exception as e:
            print(f"Error creating polygon for {fire_name} {date}: {e}")
            continue
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(output_file, 'w') as f:
        json.dump(geojson, f, indent=2, separators=(',', ': '))
    
    print(f"Saved {len(features)} fire perimeter polygons to {output_file}")
    print(f"Note: These are actual fire perimeter polygons based on high-probability predictions.")

def run_all_fires(target_points=10000, model_file="20200903-193223mod"):
    """Run predictions on all fires in the training data"""
    print("Running predictions on all fires in training data...")
    
    #get all fires from training_data directory
    fires = [f for f in os.listdir('training_data') if os.path.isdir(os.path.join('training_data', f)) and not f.startswith('.')]
    print(f"Found {len(fires)} fires: {fires}")
    
    all_predictions = {}
    
    for fire_name in fires:
        try:
            #get all available dates for this fire
            dates = rawdata.Day.allGoodDays(fire_name, inference=True)
            print(f"Processing {fire_name} with {len(dates)} dates: {dates}")
            
            for date in dates:
                try:
                    run_production_inference(fire_name, date, target_points, model_file, all_predictions)
                except Exception as e:
                    print(f"Error processing {fire_name} {date}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error processing fire {fire_name}: {e}")
            continue
    
    #save all predictions to GeoJSON
    if all_predictions:
        save_all_predictions_to_geojson(all_predictions)
    else:
        print("No predictions generated")
    
    return all_predictions

def create_perimeter_visualization(predictions, ptList, fire_name, date, pixel_to_latlon, output_path):
    """
    Create a visualization showing probability map with predicted perimeter line overlaid
    """
    #create a 1024x1024 probability map
    prob_map = np.zeros((1024, 1024), dtype=np.float32)
    
    #fill in the probability values
    for pt, pred in zip(ptList, predictions):
        burnName, date, (y, x) = pt
        prob_map[y, x] = pred
    
    #find high-probability pixels for perimeter
    threshold = 0.5
    high_prob_mask = prob_map > threshold
    
    if np.sum(high_prob_mask) > 0:
        #get coordinates of high-probability pixels
        high_prob_coords = list(zip(*np.where(high_prob_mask)))
        high_prob_coords = [(int(x), int(y)) for y, x in high_prob_coords]
        
        #find outer perimeter
        outer_boundary = find_outer_perimeter(high_prob_coords)
        
        if len(outer_boundary) >= 3:
            #filter to outermost pixels
            outermost = filter_outermost_pixels(outer_boundary)
            
            if len(outermost) >= 3:
                #create the visualization
                plt.figure(figsize=(12, 10))
                
                #plot probability map with custom color mapping
                #create custom colormap: yellow (low probability) to red (high probability)
                from matplotlib.colors import LinearSegmentedColormap
                
                #create custom colormap: yellow to red
                colors = ['yellow', 'orange', 'red']
                n_bins = 100
                custom_cmap = LinearSegmentedColormap.from_list('yellow_to_red', colors, N=n_bins)
                
                #plot probability map
                plt.imshow(prob_map, cmap=custom_cmap, interpolation='nearest', vmin=0, vmax=1)
                plt.colorbar(label='Predicted Burn Probability (Red=High, Yellow=Low)')
                
                #plot perimeter line
                if len(outermost) > 1:
                    #convert to arrays for plotting
                    x_coords = [p[0] for p in outermost]
                    y_coords = [p[1] for p in outermost]
                    
                    #plot the perimeter line
                    plt.plot(x_coords, y_coords, color='cyan', linewidth=3, label='Predicted Perimeter')
                    plt.plot(x_coords, y_coords, color='blue', linewidth=1, alpha=0.8)
                
                #add title and labels
                plt.title(f'Fire Prediction: {fire_name} ({date})\nProbability Map with Predicted Perimeter', fontsize=14)
                plt.xlabel('Pixel X Coordinate')
                plt.ylabel('Pixel Y Coordinate')
                plt.legend()
                
                #add statistics
                burned_pixels = np.sum(high_prob_mask)
                total_pixels = prob_map.size
                burn_percentage = (burned_pixels / total_pixels) * 100
                avg_prob = np.mean(predictions)
                
                stats_text = f'Burned Pixels: {burned_pixels:,}\nBurn Area: {burn_percentage:.2f}%\nAvg Probability: {avg_prob:.3f}'
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                #save the plot
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Perimeter visualization saved to: {output_path}")
                return True
            else:
                print(f"Warning: Not enough outermost pixels for visualization of {fire_name} {date}")
        else:
            print(f"Warning: Not enough boundary pixels for visualization of {fire_name} {date}")
    else:
        print(f"Warning: No high-probability pixels found for visualization of {fire_name} {date}")
    
    return False

def create_perimeter_overlay_visualization(predictions, ptList, fire_name, date, pixel_to_latlon, output_path):
    """
    Create a visualization showing the perimeter mask with predicted pixels overlaid on top
    """
    #load the perimeter mask
    perim_path = f"training_data/{fire_name}/perims/{date}.npy"
    if not os.path.exists(perim_path):
        print(f"Warning: Perimeter file not found at {perim_path}")
        return False
    
    try:
        perimeter_mask = np.load(perim_path)
        print(f"Loaded perimeter mask from {perim_path}, shape: {perimeter_mask.shape}")
    except Exception as e:
        print(f"Error loading perimeter mask: {e}")
        return False
    
    #create a 1024x1024 probability map
    prob_map = np.zeros((1024, 1024), dtype=np.float32)
    
    #fill in the probability values
    for pt, pred in zip(ptList, predictions):
        burnName, date, (y, x) = pt
        prob_map[y, x] = pred
    
    #create the visualization
    plt.figure(figsize=(15, 12))
    
    #create a custom colormap for the perimeter mask (gray scale)
    from matplotlib.colors import LinearSegmentedColormap
    
    #plot the perimeter mask as the background (gray scale)
    plt.subplot(1, 2, 1)
    plt.imshow(perimeter_mask, cmap='gray', interpolation='nearest', alpha=0.7)
    plt.title(f'Perimeter Mask: {fire_name} ({date})', fontsize=14)
    plt.xlabel('Pixel X Coordinate')
    plt.ylabel('Pixel Y Coordinate')
    plt.colorbar(label='Perimeter Mask')
    
    #plot the predicted pixels overlaid on the perimeter mask
    plt.subplot(1, 2, 2)
    
    #create custom colormap for predictions: yellow (low) to red (high)
    colors = ['yellow', 'orange', 'red']
    custom_cmap = LinearSegmentedColormap.from_list('yellow_to_red', colors, N=100)
    
    #create a masked array for predictions - only show where we have predictions
    import numpy.ma as ma
    masked_prob_map = ma.masked_where(prob_map == 0, prob_map)
    
    #plot perimeter mask as background
    plt.imshow(perimeter_mask, cmap='gray', interpolation='nearest', alpha=0.3)
    
    #overlay predicted pixels
    plt.imshow(masked_prob_map, cmap=custom_cmap, interpolation='nearest', vmin=0, vmax=1, alpha=0.8)
    plt.colorbar(label='Predicted Burn Probability (Red=High, Yellow=Low)')
    
    #add title and labels
    plt.title(f'Predicted Pixels Overlaid on Perimeter: {fire_name} ({date})', fontsize=14)
    plt.xlabel('Pixel X Coordinate')
    plt.ylabel('Pixel Y Coordinate')
    
    #add statistics
    high_prob_mask = prob_map > 0.5
    burned_pixels = np.sum(high_prob_mask)
    total_predicted = np.sum(prob_map > 0)
    avg_prob = np.mean(predictions) if len(predictions) > 0 else 0
    
    #calculate overlap with perimeter
    perimeter_pixels = np.sum(perimeter_mask > 0)
    overlap_mask = (high_prob_mask) & (perimeter_mask > 0)
    overlap_pixels = np.sum(overlap_mask)
    overlap_percentage = (overlap_pixels / perimeter_pixels * 100) if perimeter_pixels > 0 else 0
    
    stats_text = f'Predicted Burned: {burned_pixels:,}\nTotal Predicted: {total_predicted:,}\nAvg Probability: {avg_prob:.3f}\nPerimeter Pixels: {perimeter_pixels:,}\nOverlap: {overlap_pixels:,} ({overlap_percentage:.1f}%)'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    #save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Perimeter overlay visualization saved to: {output_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run production fire prediction (no ground truth required)')
    parser.add_argument('--fire', type=str, help='Fire name (as in data folder)')
    parser.add_argument('--date', type=str, help='Date in MMDD format')
    parser.add_argument('--points', type=int, default=10000, help='Number of points to sample (default: 10000)')
    parser.add_argument('--model', type=str, default="20260101-173820mod", help='Model file name (without .h5)')
    parser.add_argument('--all', action='store_true', help='Run predictions on all fires in training data')
    args = parser.parse_args()

    #if no arguments provided, run on all fires
    if len(sys.argv) == 1:
        print("No arguments provided. Running predictions on all fires...")
        run_all_fires(target_points=args.points, model_file=args.model)
        sys.exit(0)

    if args.all:
        run_all_fires(target_points=args.points, model_file=args.model)
        sys.exit(0)

    #validate fire and date for single fire mode
    if not args.fire or not args.date:
        print("Error: Must provide both --fire and --date for single fire mode")
        print("Usage:")
        print("  python3.10 runProduction.py                                    # Run on all fires")
        print("  python3.10 runProduction.py --all                             # Run on all fires")
        print("  python3.10 runProduction.py --fire <fire_name> --date <MMDD>  # Run on specific fire/date")
        sys.exit(1)

    fires = [f for f in os.listdir('training_data') if os.path.isdir(os.path.join('training_data', f)) and not f.startswith('.')]
    if args.fire not in fires:
        print(f"Error: Fire '{args.fire}' not found in training_data folder.")
        sys.exit(1)
    available_dates = rawdata.Day.allGoodDays(args.fire, inference=True)
    if args.date not in available_dates:
        print(f"Error: Date '{args.date}' not available for fire '{args.fire}'. Available dates: {available_dates}")
        sys.exit(1)

    run_production_inference(args.fire, args.date, args.points, args.model) 
