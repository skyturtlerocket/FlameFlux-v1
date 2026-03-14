import os
import argparse
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import io
import json
import rasterio
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import ee
import shutil

#earth engine: use EARTHENGINE_PROJECT env var or default init
_ee_project = os.environ.get("EARTHENGINE_PROJECT")
try:
    if _ee_project:
        ee.Initialize(project=_ee_project)
    else:
        ee.Initialize()
except Exception:
    ee.Authenticate()
    if _ee_project:
        ee.Initialize(project=_ee_project)
    else:
        ee.Initialize()

def loadJSON(api_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(api_url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()

def meters_per_degree_lat():
    return 111320

def meters_per_degree_lon(lat):
    from math import cos, radians
    return 111320 * cos(radians(lat))

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

def getBounds(center_lon, center_lat, area_m=30720):
    lat_meter = meters_per_degree_lat()
    lon_meter = meters_per_degree_lon(center_lat)
    half_width_deg = (area_m / 2) / lon_meter
    half_height_deg = (area_m / 2) / lat_meter
    return {
        'min_lon': center_lon - half_width_deg,
        'max_lon': center_lon + half_width_deg,
        'min_lat': center_lat - half_height_deg,
        'max_lat': center_lat + half_height_deg
    }

def coords_to_pixels_fixed(coords, map_bounds, img_width, img_height):
    pixels = []
    for lon, lat in coords:
        x_norm = (lon - map_bounds['min_lon']) / (map_bounds['max_lon'] - map_bounds['min_lon'])
        y_norm = (map_bounds['max_lat'] - lat) / (map_bounds['max_lat'] - map_bounds['min_lat'])
        x_pixel = int(x_norm * (img_width - 1))
        y_pixel = int(y_norm * (img_height - 1))
        pixels.append([x_pixel, y_pixel])
    return pixels

def fetch_weather_csv(lat, lon, date_str, out_csv, containment_percent=None):
    # date_str is MMDDYYYY
    #convert to YYYY-MM-DD
    try:
        dt = datetime.strptime(date_str, "%m%d%Y")
        date_api = dt.strftime("%Y-%m-%d")
        next_day = (dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        days_ago = (datetime.now() - dt).days
        
        #always try the regular forecast API first (works better for recent dates)
        #include both target date and next day to get hours 0-24
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=pressure_msl,temperature_2m,dew_point_2m,apparent_temperature,wind_direction_10m,wind_speed_10m,precipitation,relative_humidity_2m&start_date={date_api}&end_date={next_day}&format=csv"
        
        print(f"    Using regular forecast API ({days_ago} days ago)")
        print(f"    Fetching weather data from: {url}")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        csv_content = response.text
        if not csv_content or len(csv_content.strip()) == 0:
            print(f"    Error: Empty response from weather API")
            return False
        
        lines = csv_content.strip().split('\n')
        
        #debug: print first few lines
        print(f"    Weather API response has {len(lines)} lines")
        if len(lines) > 0:
            print(f"    First line: {lines[0][:100]}...")
        
        #find where the actual CSV data starts - look for line starting with 'time'
        data_start_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith('time,'):
                data_start_idx = i
                print(f"    Found 'time,' header at line {i}")
                break
        
        if data_start_idx is None:
            print(f"    Error: No 'time,' header found in weather response")
            print(f"    Available lines (first 10):")
            for i, line in enumerate(lines[:10]):
                print(f"      Line {i}: {line}")
            return False
        
        #extract CSV data starting from the time header
        data_csv = '\n'.join(lines[data_start_idx:])
        
        #try to read the CSV
        try:
            df = pd.read_csv(io.StringIO(data_csv))
        except Exception as e:
            print(f"    Error parsing weather CSV: {e}")
            print(f"    Raw CSV content (first 500 chars): {data_csv[:500]}")
            return False
        
        #check if 'time' column exists
        if 'time' not in df.columns:
            print(f"    Error: 'time' column not found in weather data")
            print(f"    Available columns: {list(df.columns)}")
            return False
        
        #convert time column to datetime
        try:
            df['time'] = pd.to_datetime(df['time'])
        except Exception as e:
            print(f"    Error converting 'time' column to datetime: {e}")
            print(f"    Sample time values: {df['time'].head().tolist()}")
            return False
        
        #get hours 0-23 for the target day
        mask_target = (df['time'].dt.date == dt.date()) & (df['time'].dt.hour >= 0) & (df['time'].dt.hour <= 23)
        df_target = df[mask_target].copy()
        
        #get hour 0 for the next day (will become hour 24)
        next_day_date = (dt + pd.Timedelta(days=1)).date()
        mask_next0 = (df['time'].dt.date == next_day_date) & (df['time'].dt.hour == 0)
        df_next0 = df[mask_next0].copy()
        
        print(f"    Found {len(df_target)} rows for target date (hours 0-23)")
        print(f"    Found {len(df_next0)} rows for next day hour 0 (will become hour 24)")
        
        if len(df_target) == 0:
            print(f"    Error: No data found for target date {dt.date()}")
            return False
        
        #format target day rows (hours 0-23)
        df_target['DATE'] = df_target['time'].dt.strftime('%m%d%Y')
        df_target['HOUR'] = df_target['time'].dt.hour
        df_target['LAT'] = lat
        df_target['LONG'] = lon
        
        #map columns - handle different possible column names
        column_mapping = {
            'pressure_msl (hPa)': 'MSL PRESSURE   ',
            'temperature_2m (°C)': 'TEMPERATURE   ',
            'temperature_2m (Â°C)': 'TEMPERATURE   ',  # Handle encoding issues
            'temperature_2m (Ã‚Â°C)': 'TEMPERATURE   ',
            'dew_point_2m (°C)': 'DEW POINT      ',
            'dew_point_2m (Â°C)': 'DEW POINT      ',
            'dew_point_2m (Ã‚Â°C)': 'DEW POINT      ',
            'apparent_temperature (°C)': ' TEMPERATURE   ',
            'apparent_temperature (Â°C)': ' TEMPERATURE   ',
            'apparent_temperature (Ã‚Â°C)': ' TEMPERATURE   ',
            'wind_direction_10m (°)': 'WIND DIRECTION ',
            'wind_direction_10m (Â°)': 'WIND DIRECTION ',
            'wind_direction_10m (Ã‚Â°)': 'WIND DIRECTION ',
            'wind_speed_10m (km/h)': 'WIND SPEED     ',
            'precipitation (mm)': ' PRECIPITATION',
            'relative_humidity_2m (%)': 'RELATIVE HUMIDITY'
        }
        
        #create reformatted dataframe with hours 0-23
        reformatted_df = pd.DataFrame()
        reformatted_df['DATE'] = df_target['DATE']
        reformatted_df['HOUR'] = df_target['HOUR']
        reformatted_df['LAT'] = df_target['LAT']
        reformatted_df['LONG'] = df_target['LONG']
        
        #add containment percentage
        if containment_percent is not None:
            reformatted_df['CONTAINMENT'] = containment_percent
        else:
            reformatted_df['CONTAINMENT'] = 0.0  # Default to 0% if not available
            
        for api_col, target_col in column_mapping.items():
            if api_col in df_target.columns:
                reformatted_df[target_col] = df_target[api_col]
                print(f"    Mapped {api_col} -> {target_col}")
            else:
                print(f"    Warning: Column '{api_col}' not found in weather data")
        
        #add hour 24 row if we have next day's hour 0 data
        if not df_next0.empty:
            print(f"    Adding hour 24 row from next day's hour 0")
            row24 = {
                'DATE': df_target['DATE'].iloc[0],  # Use target day's date
                'HOUR': 24,
                'LAT': lat,
                'LONG': lon
            }
            
            #add containment percentage to hour 24 row
            if containment_percent is not None:
                row24['CONTAINMENT'] = containment_percent
            else:
                row24['CONTAINMENT'] = 0.0
                
            #add weather data for hour 24
            for api_col, target_col in column_mapping.items():
                if api_col in df_next0.columns:
                    row24[target_col] = df_next0[api_col].iloc[0]
            
            #append row24 to the dataframe
            reformatted_df = pd.concat([reformatted_df, pd.DataFrame([row24])], ignore_index=True)
        else:
            print(f"    Warning: No next day hour 0 data found - missing hour 24!")
        
        #verify we have the correct number of rows (should be 25: hours 0-24)
        expected_rows = 25
        if len(reformatted_df) == expected_rows:
            print(f"    ✓ Correct number of rows: {len(reformatted_df)} (hours 0-24)")
        else:
            print(f"    Warning: Expected {expected_rows} rows, got {len(reformatted_df)}")
        
        reformatted_df.to_csv(out_csv, index=False)
        print(f"  Saved weather CSV: {out_csv}")
        return True
        
    except Exception as e:
        print(f"    Error in fetch_weather_csv: {e}")
        import traceback
        traceback.print_exc()
        return False
def fetch_landsat_and_terrain(center_lat, center_lon, fire_dir, create_date=None, img_size=(1024, 1024), area_m=30720, scale=30):
    #extract width and height from img_size tuple
    img_width, img_height = img_size
    os.makedirs(fire_dir, exist_ok=True)
    #check if all files exist
    band_names = ['band_2', 'band_3', 'band_4', 'band_5', 'ndvi', 'dem', 'slope', 'aspect']
    all_exist = all([os.path.exists(os.path.join(fire_dir, f'{name}.npy')) for name in band_names])
    if all_exist:
        print(f"  Skipping Landsat/terrain fetch for {fire_dir} (all files exist)")
        return
    print(f"  Fetching Landsat/terrain for {fire_dir} ...")
    lat_meter = meters_per_degree_lat()
    lon_meter = meters_per_degree_lon(center_lat)
    half_width_deg = (area_m / 2) / lon_meter
    half_height_deg = (area_m / 2) / lat_meter
    min_lat = center_lat - half_height_deg
    max_lat = center_lat + half_height_deg
    min_lon = center_lon - half_width_deg
    max_lon = center_lon + half_width_deg
    region = [[min_lon, min_lat], [min_lon, max_lat], [max_lon, max_lat], [max_lon, min_lat], [min_lon, min_lat]]
    bbox = [min_lon, min_lat, max_lon, max_lat]
    crs = 'EPSG:4326'
    
    #use fixed date range with quality filtering
    start_date = '2022-01-01'
    end_date = '2024-12-31'
    
    #landsat bands and NDVI - get best available image from date range
    collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1')
                  .filterBounds(ee.Geometry.Point(center_lon, center_lat))
                  .filterDate(start_date, end_date)
                  .sort('CLOUD_COVER'))  # Sort by cloud cover to get best available
    
    #get the best image (lowest cloud cover)
    image = collection.first()
    if not image:
        print(f"    Warning: No Landsat image found for date range {start_date} to {end_date}")
        return
    
    bands = ['B2', 'B3', 'B4', 'B5']
    image_bands = image.select(bands)
    ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI')
    stacked = image_bands.addBands(ndvi)
    
    #download as GeoTIFF using getDownloadURL
    try:
        url = stacked.getDownloadURL({
            'scale': scale,
            'crs': crs,
            'region': bbox,
            'format': 'GEO_TIFF'
        })
        print("    Downloading Landsat bands and NDVI from Earth Engine...")
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        tiff_path = os.path.join(fire_dir, 'landsat_temp.tif')
        with open(tiff_path, 'wb') as f:
            f.write(response.content)
        print(f"    Downloaded as {tiff_path}")
        with rasterio.open(tiff_path) as src:
            arr = src.read()  # shape: (bands, height, width)
            band_file_names = ['band_2', 'band_3', 'band_4', 'band_5', 'ndvi']
            for i, name in enumerate(band_file_names):
                band = arr[i]
                #resize to 1024x1024 if needed
                if band.shape != (img_height, img_width):
                    band_resized = resize(band, (img_height, img_width), order=1, preserve_range=True, anti_aliasing=True).astype(band.dtype)
                else:
                    band_resized = band
                np.save(os.path.join(fire_dir, f'{name}.npy'), band_resized)
                print(f"    Saved {name}.npy, shape: {band_resized.shape}")
        os.remove(tiff_path)
    except Exception as e:
        print(f"    Error processing Landsat data: {e}")
        #continue with terrain data even if Landsat fails
    
    #DEM, Slope, Aspect - Use global DEM without date filtering
    dem = ee.Image('USGS/SRTMGL1_003')
    slope = ee.Terrain.slope(dem)
    aspect = ee.Terrain.aspect(dem)
    
    #ensure all are Earth Engine images
    dem_img = ee.Image(dem)
    slope_img = ee.Image(slope)
    aspect_img = ee.Image(aspect)
    
    for img, name in zip([dem_img, slope_img, aspect_img], ['dem', 'slope', 'aspect']):
        try:
            url = img.getDownloadURL({
                'scale': scale,
                'crs': crs,
                'region': bbox,
                'format': 'GEO_TIFF'
            })
            print(f"    Downloading {name} from Earth Engine...")
            response = requests.get(url)
            response.raise_for_status()  # Check for HTTP errors
            tiff_path = os.path.join(fire_dir, f'{name}_temp.tif')
            with open(tiff_path, 'wb') as f:
                f.write(response.content)
            with rasterio.open(tiff_path) as src:
                arr = src.read(1)
                if arr.shape != (img_height, img_width):
                    arr_resized = resize(arr, (img_height, img_width), order=1, preserve_range=True, anti_aliasing=True).astype(arr.dtype)
                else:
                    arr_resized = arr
                np.save(os.path.join(fire_dir, f'{name}.npy'), arr_resized)
                print(f"    Saved {name}.npy, shape: {arr_resized.shape}")
            os.remove(tiff_path)
        except Exception as e:
            print(f"    Error processing {name}: {e}")
            continue

def fetch_satellite_hotspots(fire_dir, date_str, img_size=(1024, 1024)):
    """
    Fetch MODIS and VIIRS hotspot data and create gradient maps.
    
    Args:
        fire_dir: Directory containing fire data
        date_str: Date in MMDDYYYY format
        img_size: Image dimensions (width, height)
    
    Returns:
        bool: True if hotspots were found and processed, False otherwise
    """
    try:
        #load center.json to get bounds
        center_json_path = os.path.join(fire_dir, 'center.json')
        if not os.path.exists(center_json_path):
            print(f"    No center.json found for {fire_dir}")
            return False
            
        with open(center_json_path, 'r') as f:
            center_data = json.load(f)
        
        bounds = center_data['bounds']
        west, south, east, north = bounds['west'], bounds['south'], bounds['east'], bounds['north']
        
        #convert date from MMDDYYYY to YYYY-MM-DD
        dt = datetime.strptime(date_str, "%m%d%Y")
        api_date = dt.strftime("%Y-%m-%d")
        
        #nasa FIRMS api key: set NASA_FIRMS_API_KEY env var (get your own from https://firms.modaps.eosdis.nasa.gov/)
        api_key = os.environ.get("NASA_FIRMS_API_KEY")
        if not api_key:
            raise ValueError("NASA_FIRMS_API_KEY environment variable is required for hotspot data. Get a key at https://firms.modaps.eosdis.nasa.gov/")
        
        #create hotspot directory
        hotspot_dir = os.path.join(fire_dir, 'hotspots')
        os.makedirs(hotspot_dir, exist_ok=True)
        
        #check if hotspot file already exists
        mmdd = date_str[:4]
        hotspot_path = os.path.join(hotspot_dir, f"{mmdd}.npy")
        if os.path.exists(hotspot_path):
            print(f"    Skipped existing hotspot file: {hotspot_path}")
            return True
        
        all_hotspots = []
        
        #fetch both MODIS and VIIRS data
        instruments = ['MODIS_NRT', 'VIIRS_SNPP_NRT']
        
        for instrument in instruments:
            try:
                #construct API URL
                bounds_str = f"{west},{south},{east},{north}"
                url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{instrument}/{bounds_str}/1/{api_date}"
                
                print(f"    Fetching {instrument} hotspots for {api_date}...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                #parse CSV data
                csv_content = response.text.strip()
                if not csv_content or csv_content.startswith('No data'):
                    print(f"    No {instrument} data found for {api_date}")
                    continue
                
                #parse CSV (skip header if present)
                lines = csv_content.split('\n')
                if lines[0].startswith('latitude'):
                    lines = lines[1:]  # Skip header
                
                for line in lines:
                    if not line.strip():
                        continue
                    parts = line.split(',')
                    if len(parts) >= 13:  # Ensure we have enough columns
                        try:
                            lat = float(parts[0])
                            lon = float(parts[1])
                            frp = float(parts[12])  # FRP is the 13th column
                            confidence = parts[9]  # Confidence is the 10th column
                            
                            #only include hotspots with valid FRP values
                            if frp > 0:
                                all_hotspots.append({
                                    'lat': lat,
                                    'lon': lon,
                                    'frp': frp,
                                    'confidence': confidence,
                                    'instrument': instrument
                                })
                        except (ValueError, IndexError):
                            continue
                            
            except Exception as e:
                print(f"    Error fetching {instrument} data: {e}")
                continue
        
        if not all_hotspots:
            print(f"    No hotspots found for {api_date}")
            #create empty hotspot map
            hotspot_map = np.zeros(img_size, dtype=np.float32)
            np.save(hotspot_path, hotspot_map)
            return False
        
        print(f"    Found {len(all_hotspots)} hotspots for {api_date}")
        
        #create gradient map from hotspots
        hotspot_map = np.zeros(img_size, dtype=np.float32)
        img_width, img_height = img_size
        
        #convert hotspot coordinates to pixel coordinates
        for hotspot in all_hotspots:
            lat, lon = hotspot['lat'], hotspot['lon']
            frp = hotspot['frp']
            
            #convert lat/lon to pixel coordinates
            x_norm = (lon - west) / (east - west)
            y_norm = (north - lat) / (north - south)  # Flip Y axis
            
            x_pixel = int(x_norm * (img_width - 1))
            y_pixel = int(y_norm * (img_height - 1))
            
            #ensure pixel is within bounds
            if 0 <= x_pixel < img_width and 0 <= y_pixel < img_height:
                #add FRP value to the pixel (accumulate if multiple hotspots)
                hotspot_map[y_pixel, x_pixel] += frp
        
        #apply Gaussian smoothing to create gradient effect
        if np.sum(hotspot_map) > 0:
            #normalize and apply smoothing with larger radius
            hotspot_map = gaussian_filter(hotspot_map, sigma=16.0)  # Increased to 16.0 for very large gradients
            #normalize to 0-1 range
            max_val = np.max(hotspot_map)
            if max_val > 0:
                hotspot_map = hotspot_map / max_val
        
        #save hotspot map
        np.save(hotspot_path, hotspot_map)
        print(f"    Saved hotspot map: {hotspot_path}, shape: {hotspot_map.shape}")
        
        return True
        
    except Exception as e:
        print(f"    Error processing hotspots for {fire_dir}: {e}")
        return False

def process(api_url, fire_name=None):
    data = loadJSON(api_url)
    if data['type'] != 'FeatureCollection':
        print("Error: Expected FeatureCollection")
        return
    
    print(f"Processing {len(data['features'])} features...")
    img_size = (1024, 1024)
    area_m = 1024 * 30
    
    #alaska bounds to filter out Alaskan fires
    ALASKA_BOUNDS = {
        'min_lat': 51.2,
        'max_lat': 71.4,
        'min_lon': -173.0,  # 173°E = -173°W
        'max_lon': -130.0
    }
    
    #set up NIFC data directory
    nifc_data_dir = "training_data"
    
    #track which fires we've processed in this run
    processed_fires = set()
    fire_centers = {}  # fire_name -> (center_lat, center_lon)
    
    for i, feature in enumerate(data['features']):
        try:
            geometry = feature['geometry']
            properties = feature['properties']
            bounds = get_polygon_bounds(geometry['coordinates'], geometry['type'])
            if not bounds:
                continue
            
            #filter out Alaskan fires
            center_lon = (bounds['min_lon'] + bounds['max_lon']) / 2
            center_lat = (bounds['min_lat'] + bounds['max_lat']) / 2
            
            #check if fire is within Alaska bounds
            if (ALASKA_BOUNDS['min_lat'] <= center_lat <= ALASKA_BOUNDS['max_lat'] and 
                ALASKA_BOUNDS['min_lon'] <= center_lon <= ALASKA_BOUNDS['max_lon']):
                if 'poly_IncidentName' in properties:
                    incident_name = properties['poly_IncidentName']
                    print(f"  Skipping Alaskan fire: {incident_name} (lat: {center_lat:.3f}, lon: {center_lon:.3f})")
                continue
            
            #check if this fire has been updated in the last 24 hours
            if 'poly_DateCurrent' in properties:
                dt = datetime.fromtimestamp(properties['poly_DateCurrent'] / 1000)
                current_time = datetime.now()
                time_diff = current_time - dt
                if time_diff.total_seconds() > 24 * 3600:  # Skip if older than 24 hours
                    continue
                date_str = dt.strftime('%m%d%Y')
            else:
                continue
            
            if 'poly_IncidentName' in properties:
                incident_name = properties['poly_IncidentName']
                if fire_name is not None and incident_name != fire_name:
                    continue
            else:
                continue
            
            #filter out fires smaller than 100 acres
            if 'poly_Acres_AutoCalc' in properties:
                acres = properties['poly_Acres_AutoCalc']
                if acres is None or acres < 100:
                    print(f"  Skipping {incident_name}: {acres} acres (below 100-acre minimum)")
                    continue
            else:
                print(f"  Skipping {incident_name}: no acreage data available")
                continue
            
            #get fire creation date for Landsat data
            create_date = None
            if 'poly_CreateDate' in properties:
                create_date = properties['poly_CreateDate']
            
            #get containment percentage
            containment_percent = None
            if 'attr_PercentContained' in properties:
                containment_percent = properties['attr_PercentContained']
                if containment_percent is not None:
                    print(f"  Containment: {containment_percent}%")
            
            #track this fire as processed
            processed_fires.add(incident_name)
            
            #store or retrieve fire center
            fire_dir = os.path.join(nifc_data_dir, incident_name.replace(' ', '_'))
            if fire_dir not in fire_centers:
                center_lon = (bounds['min_lon'] + bounds['max_lon']) / 2
                center_lat = (bounds['min_lat'] + bounds['max_lat']) / 2
                fire_centers[fire_dir] = (center_lat, center_lon)
                
                #save center.json for this fire
                os.makedirs(fire_dir, exist_ok=True)
                center_json_path = os.path.join(fire_dir, 'center.json')
                
                #calculate bounds for 1024x1024 image with 30m pixels
                img_width, img_height = img_size  # img_size is (1024, 1024)
                pixel_size_m = 30
                area_m = img_width * pixel_size_m  # 30720 meters
                
                lat_meter = meters_per_degree_lat()
                lon_meter = meters_per_degree_lon(center_lat)
                half_width_deg = (area_m / 2) / lon_meter
                half_height_deg = (area_m / 2) / lat_meter
                
                bounds = {
                    'west': center_lon - half_width_deg,
                    'south': center_lat - half_height_deg,
                    'east': center_lon + half_width_deg,
                    'north': center_lat + half_height_deg
                }
                
                center_data = {
                    'center_lat': center_lat, 
                    'center_lon': center_lon,
                    'bounds': bounds
                }
                
                with open(center_json_path, 'w') as f:
                    json.dump(center_data, f, indent=2)
                
                #fetch Landsat/terrain data ONCE per fire using creation date
                fetch_landsat_and_terrain(center_lat, center_lon, fire_dir, create_date=create_date, img_size=img_size, area_m=area_m)
            else:
                center_lat, center_lon = fire_centers[fire_dir]
            
            perim_dir = os.path.join(fire_dir, 'perims')
            weather_dir = os.path.join(fire_dir, 'weather')
            os.makedirs(perim_dir, exist_ok=True)
            os.makedirs(weather_dir, exist_ok=True)
            
            #nPY filename: MMDD.npy
            mmdd = date_str[:4]
            npy_path = os.path.join(perim_dir, f"{mmdd}.npy")
            if not os.path.exists(npy_path):
                fixed_bounds = getBounds(center_lon, center_lat, area_m=area_m)
                img = np.zeros(img_size, dtype=np.uint8)
                polygons = []
                if geometry['type'] == 'Polygon':
                    polygons.append(Polygon(geometry['coordinates'][0]))
                elif geometry['type'] == 'MultiPolygon':
                    for polygon in geometry['coordinates']:
                        polygons.append(Polygon(polygon[0]))
                if not polygons:
                    continue
                fire_poly = unary_union(polygons)
                from matplotlib.path import Path
                if fire_poly.geom_type == 'Polygon':
                    polys = [fire_poly]
                else:
                    polys = fire_poly.geoms
                for poly in polys:
                    exterior = np.array(coords_to_pixels_fixed(list(poly.exterior.coords), fixed_bounds, img_size[1], img_size[0]))
                    path = Path(exterior)
                    y_grid, x_grid = np.mgrid[:img_size[0], :img_size[1]]
                    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
                    mask_flat = path.contains_points(points)
                    mask = mask_flat.reshape(img_size)
                    img[mask] = 1
                np.save(npy_path, img)
                print(f"  Saved NPY: {npy_path}")
            else:
                print(f"  Skipped existing NPY: {npy_path}")
            
            #fetch weather data with better error handling
            csv_path = os.path.join(weather_dir, f"{mmdd}.csv")
            if not os.path.exists(csv_path):
                success = fetch_weather_csv(center_lat, center_lon, date_str, csv_path, containment_percent=containment_percent)
                if not success:
                    print(f"  Warning: Failed to fetch weather data for {incident_name}")
            else:
                print(f"  Skipped existing weather CSV: {csv_path}")
                
            #fetch satellite hotspots
            try:
                fetch_satellite_hotspots(fire_dir, date_str, img_size=img_size)
            except Exception as e:
                print(f"  Warning: Failed to fetch hotspots for {incident_name}: {e}")
                
        except Exception as e:
            print(f"Error processing feature {i+1}: {e}")
            import traceback
            print(f"  Full traceback:")
            traceback.print_exc()
            continue
    
    print(f"Processing complete! Processed {len(processed_fires)} fires from the last 24 hours.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process fire perimeters and weather data into structured folders.")
    parser.add_argument('--fire', type=str, default=None, help='Filter to a specific fire name (case sensitive)')
    args = parser.parse_args()
    api_url = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson"
    process(api_url, fire_name=args.fire)