import numpy as np
import cv2
import pandas as pd

PIXEL_SIZE = 30

class Data(object):
    '''Contains an layered image of input data and an image of output data, as well as non-spatial data such as weather'''

    def __init__(self):
        self.shape = None

        self.layers = {}
        self.weather = {}
        self.output = None

    def addLayer(self, layerName, data, use=True):
        '''Add the actual layer of data(either spatial 2D or 1D weather/spatial-invariant) to our set'''
        assert len(data.shape) == 2
        if self.shape is not None:
            if self.shape != data.shape:
                raise ValueError("All spatial data must be the same dimension.")
        else:
            self.shape = data.shape
        self.layers[layerName] = data

    def addData(self, variableName, data, use=True):
        self.weather[variableName] = data

    def addOutput(self, data):
        if self.shape is not None:
            if self.shape != data.shape:
                raise ValueError("All spatial data must be the same dimension.")
        else:
            self.shape = data.shape
        self.output = data

    def stackLayers(self, layerNames=None):
        if layerNames is None:
            layers = self.layers.values()
        else:
            layers = [self.layers[name] for name in layerNames]
        stacked = np.dstack(tuple(layers))
        return stacked

    def stackWeather(self, variableNames=None):
        if variableNames is None:
            metrics = list(self.weather.values())
        else:
            metrics = [self.weather[name] for name in variableNames]
        stacked = np.array(metrics)
        return stacked

    def getOutput(self):
        '''Return a 1D array of output values, ready to be used by the model'''
        if self.output is None:
            raise ValueError("Output data must be set.")
        return self.output

    @staticmethod
    def defaultData(dateString):
        d = Data()
        
        #load data from the new training_data structure
        import os
        import glob
        
        #find the fire directory that contains this date
        fire_pattern = f'training_data/*/perims/{dateString}.npy'
        fire_files = glob.glob(fire_pattern)
        
        if not fire_files:
            raise FileNotFoundError(f"No fire data found for date {dateString}")
        
        #extract fire name from path
        fire_path = fire_files[0]
        fire_name = fire_path.split('/')[1]  # training_data/FIRE_NAME/...
        fire_dir = f'training_data/{fire_name}'
        
        #load terrain and Landsat data
        dem = np.load(os.path.join(fire_dir, 'dem.npy'))
        slope = np.load(os.path.join(fire_dir, 'slope.npy'))
        ndvi = np.load(os.path.join(fire_dir, 'ndvi.npy'))
        aspect = np.load(os.path.join(fire_dir, 'aspect.npy'))
        
        #load Landsat bands
        band_2 = np.load(os.path.join(fire_dir, 'band_2.npy'))
        band_3 = np.load(os.path.join(fire_dir, 'band_3.npy'))
        band_4 = np.load(os.path.join(fire_dir, 'band_4.npy'))
        band_5 = np.load(os.path.join(fire_dir, 'band_5.npy'))
        
        #add layers
        d.addLayer('dem', dem)
        d.addLayer('slope', slope)
        d.addLayer('ndvi', ndvi)
        d.addLayer('aspect', aspect)
        d.addLayer('band_2', band_2)
        d.addLayer('band_3', band_3)
        d.addLayer('band_4', band_4)
        d.addLayer('band_5', band_5)

        perim = Data.openStartingPerim(dateString)
        d.addLayer('perim', perim)

        #add hotspot layer
        try:
            hotspot = Data.openHotspotData(dateString)
            d.addLayer('hotspot', hotspot)
        except Exception as e:
            #if hotspot data doesn't exist, create empty layer
            print(f"Warning: No hotspot data for {dateString}, using empty layer")
            empty_hotspot = np.zeros_like(perim)
            d.addLayer('hotspot', empty_hotspot)

        weatherData = Data.createWeatherMetrics(Data.openWeatherData(dateString))
        for name, data in zip(['maxTemp', 'avgWSpeed', 'avgWDir', 'precip', 'avgHum', 'containment'], weatherData):
            d.addData(name, data)

        output = Data.openEndingPerim(dateString)
        d.addOutput(output)

        return d

    def __repr__(self):
        res = "Data("
        res += "layers:" + repr(self.layers.keys())
        res += ", data:" + repr(self.weather.keys())
        return res

    def openStartingPerim(dateString):
        import os
        import glob
        
        #find the fire directory that contains this date
        fire_pattern = f'training_data/*/perims/{dateString}.npy'
        fire_files = glob.glob(fire_pattern)
        
        if not fire_files:
            raise FileNotFoundError(f"No perimeter file found for {dateString}")
        
        #load perimeter from .npy file
        perim = np.load(fire_files[0])
        return perim

    def openEndingPerim(dateString):
        '''Get the fire perimeter on the next day'''
        import os
        import glob
        
        month, day = dateString[:2], dateString[2:]
        nextDay = str(int(day)+1).zfill(2)
        guess = month+nextDay
        
        #look for next day perimeter in training_data
        fire_pattern = f'training_data/*/perims/{guess}.npy'
        fire_files = glob.glob(fire_pattern)
        
        if not fire_files:
            # overflowed the month, that file didnt exist
            nextMonth = str(int(month)+1).zfill(2)
            guess = nextMonth+'01'
            fire_pattern = f'training_data/*/perims/{guess}.npy'
            fire_files = glob.glob(fire_pattern)
            
            if not fire_files:
                raise RuntimeError('Could not find a perimeter for the day after ' + dateString)
        
        #load perimeter from .npy file
        perim = np.load(fire_files[0])
        return perim

    def openHotspotData(dateString):
        '''Load hotspot data from the new training_data structure'''
        import os
        import glob
        
        #look for hotspot file in training_data directories
        hotspot_pattern = f'training_data/*/hotspots/{dateString}.npy'
        hotspot_files = glob.glob(hotspot_pattern)
        
        if not hotspot_files:
            raise FileNotFoundError(f"No hotspot file found for {dateString}")
        
        #use the first matching file (assuming one fire per date for now)
        hotspot_file = hotspot_files[0]
        hotspot_data = np.load(hotspot_file)
        
        #ensure it's the right shape and type
        if len(hotspot_data.shape) != 2:
            raise ValueError(f"Hotspot data should be 2D, got shape {hotspot_data.shape}")
        
        return hotspot_data

    def openWeatherData(dateString):
        import os
        import glob
        
        #find the weather file in training_data
        weather_pattern = f'training_data/*/weather/{dateString}.csv'
        weather_files = glob.glob(weather_pattern)
        
        if not weather_files:
            raise FileNotFoundError(f"No weather file found for {dateString}")
        
        #load CSV with pandas to handle the new format with containment
        df = pd.read_csv(weather_files[0])
        
        #extract weather columns (skip DATE, HOUR, LAT, LONG, CONTAINMENT)
        weather_cols = ['MSL PRESSURE   ', 'TEMPERATURE   ', 'DEW POINT      ', 
                       ' TEMPERATURE   ', 'WIND DIRECTION ', 'WIND SPEED     ', 
                       ' PRECIPITATION', 'RELATIVE HUMIDITY']
        
        #get containment percentage (should be same for all rows)
        containment = df['CONTAINMENT'].iloc[0] if 'CONTAINMENT' in df.columns else 0.0
        
        #extract weather data as numpy array
        weather_data = df[weather_cols].values.T
        
        #add containment as a new row
        containment_row = np.full((1, weather_data.shape[1]), containment)
        weather_data = np.vstack([weather_data, containment_row])
        
        return weather_data

    def createWeatherMetrics(weatherData):
        msl_pressure, temp, dewpt, temp2, wdir, wspeed, precip, hum, containment = weatherData
        avgWSpeed = sum(wspeed)/len(wspeed)
        totalPrecip = sum(precip)
        avgWDir= sum(wdir)/len(wdir)
        avgHum = sum(hum)/len(hum)
        return np.array([max(temp), avgWSpeed, avgWDir, totalPrecip, avgHum, containment[0]])
