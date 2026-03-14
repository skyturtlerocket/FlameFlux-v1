# preprocess.py
from collections import namedtuple
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    pass

try:
    from lib import util
except:
    import util
try:
    from lib import spatial_features
except ImportError:
    spatial_features = None


class PreProcessor(object):
    '''What is responsible for extracting the used data from the dataset and then
    normalizing or doing any other steps before feeding it into the network.'''

    def __init__(self, numWeatherInputs, whichLayers, AOIRadius, use_spatial_features=False, use_physics_layer=False):
        self.numWeatherInputs = numWeatherInputs
        self.whichLayers = whichLayers
        self.AOIRadius = AOIRadius
        self.use_spatial_features = use_spatial_features
        self.use_physics_layer = use_physics_layer

    def process(self, dataset, inference=False):
        '''Take a dataset and return the extracted inputs and outputs, or just inputs if inference'''
        # create dictionaries mapping from Point to actual data from that Point
        metrics = calculateWeatherMetrics(dataset)
        oneMetric = list(metrics.values())[0]
        base_weather_dim = 9
        assert len(oneMetric) == base_weather_dim, "Weather metrics must return 9 features"
        aois = getSpatialData(dataset, self.whichLayers, self.AOIRadius)
        if not inference:
            outs = getOutputs(dataset)

        # convert the dictionaries into lists, then arrays
        print("  Converting to arrays...")
        w, i, o = [], [], []
        s_for_physics = [] if self.use_physics_layer else None
        ptList = dataset.toList(dataset.points)
        total_points = len(ptList)
        print(f"    Converting {total_points:,} points to arrays...")

        #cache per (burn, date) for spatial features - avoids recomputing distance transform, centroid, wind, hotspot
        _spatial_cache = {}

        for j, pt in enumerate(ptList):
            if j % 100000 == 0 and j > 0:  # Show progress every 100k points
                print(f"      Converted {j:,}/{total_points:,} points ({j/total_points*100:.1f}%)")

            burnName, date, location = pt
            weather_row = list(metrics[burnName, date])
            if self.use_spatial_features and spatial_features is not None:
                cache_key = (burnName, date)
                if cache_key not in _spatial_cache:
                    day = dataset.data.getDay(burnName, date)
                    _spatial_cache[cache_key] = spatial_features._compute_per_burn_date_cache(burnName, date, day, dataset)
                day = dataset.data.getDay(burnName, date)
                cache = _spatial_cache[cache_key]
                sf = spatial_features.compute_spatial_features(burnName, date, location, day, dataset, cache=cache)
                weather_row.extend(sf[:7].tolist())
                if self.use_physics_layer:
                    s_for_physics.append(sf[[0, 6, 7]].tolist())
            w.append(weather_row)
            i.append(aois[burnName, date, location])
            if not inference:
                o.append(outs[burnName, date, location])

        print(f"    Converting to numpy arrays...")
        weatherInputs = np.array(w, dtype=np.float32)
        imgInputs = np.array(i)
        if not inference:
            outputs = np.array(o)
        if self.use_physics_layer and s_for_physics:
            spatialInputs = np.array(s_for_physics, dtype=np.float32)
            inputs = [weatherInputs, imgInputs, spatialInputs]
        else:
            inputs = [weatherInputs, imgInputs]
        if not inference:
            print(f"    Preprocessing complete! Arrays shape: weather={weatherInputs.shape}, images={imgInputs.shape}, outputs={outputs.shape}")
            return (inputs, outputs), ptList
        else:
            print(f"    Preprocessing complete! Arrays shape: weather={weatherInputs.shape}, images={imgInputs.shape}")
            return inputs, ptList

def calculateWeatherMetrics(dataset):
    '''Return a dictionary mapping from (burnName, date) id's to a dictionary of named weather metrics.'''
    metrics = {}
    for burnName, date in dataset.getUsedBurnNamesAndDates():
        wm = dataset.data.getWeather(burnName, date)
        precip = totalPrecipitation(wm)
        temp = maximumTemperature1(wm)
        temp2 = maximumTemperature2(wm)
        hum = averageHumidity(wm)
        winds = windMetrics(wm)
        containment = containmentValue(wm)  # NOW ACTUALLY USE IT!
        entry = [precip, temp, temp2, hum, containment] + winds  # 5 + 4 = 9 features
        metrics[(burnName, date)] = entry
    
    # now normalize all of them
    ids = list(metrics.keys())
    arr = np.array([metrics[i] for i in ids])
    normed = util.normalize(arr, axis=0)
    metrics = {i:nums for (i,nums) in zip(ids, normed)}
    return metrics

def getSpatialData(dataset, whichLayers, AOIRadius):
    # for each channel in the dataset, get all of the used data
    print("  Loading spatial layers...")
    layers = {layerName:dataset.getAllLayers(layerName) for layerName in whichLayers}
    # now normalize them
    print("  Normalizing spatial layers...")
    layers = normalizeLayers(layers)
    # now order them in the whichLayers order, stack them, and pad them
    print("  Stacking and padding layers...")
    paddedLayers = stackAndPad(layers, whichLayers, dataset, AOIRadius)
    # now extract out the aois around each point
    print("  Extracting AOIs for each point...")
    result = {}
    ptList = dataset.toList(dataset.points)
    total_points = len(ptList)
    print(f"    Processing {total_points:,} points...")
    
    for i, pt in enumerate(ptList):
        if i % 100000 == 0 and i > 0:  # Show progress every 100k points
            print(f"      Processed {i:,}/{total_points:,} points ({i/total_points*100:.1f}%)")
        
        burnName, date, location = pt
        padded = paddedLayers[(burnName, date)]
        aoi = extract(padded, location, AOIRadius)
        result[(burnName, date, location)] = aoi
    
    print(f"    Completed processing all {total_points:,} points")
    return result

def normalizeLayers(layers):
    result = {}
    for name, data in layers.items():
        if name == 'dem':
            result[name] = normalizeElevations(data)
        else:
            result[name] = normalizeNonElevations(data)
    return result

def normalizeElevations(dems):
    avgElevation = {}
    validIndicesDict = {}
    ranges = {}
    for burnName, dem in dems.items():
        validIndices = np.where(np.isfinite(dem))
        validIndicesDict[burnName] = validIndices
        validPixels = dem[validIndices]
        
        #handle case where all pixels are NaN/infinite
        if len(validPixels) == 0:
            print(f"Warning: No valid pixels found in DEM for {burnName}, using zeros")
            avgElevation[burnName] = 0.0
            ranges[burnName] = 1.0  # Use default range to avoid division by zero
        else:
            avgElevation[burnName] = np.mean(validPixels)
            ranges[burnName] = validPixels.max() - validPixels.min()
            #handle case where min == max (all pixels have same value)
            if ranges[burnName] == 0:
                ranges[burnName] = 1.0

    maxRange = max(ranges.values())
    results = {}
    for burnName, dem in dems.items():
        validIndices = validIndicesDict[burnName]
        validPixels = dem[validIndices]
        
        if len(validPixels) == 0:
            #if no valid pixels, create zero array
            results[burnName] = np.zeros_like(dem, dtype=np.float32)
        else:
            normed = util.normalize(validPixels)
            blank = np.zeros_like(dem, dtype=np.float32)
            thisRange = ranges[burnName]
            scaleFactor = thisRange/maxRange
            blank[validIndices] = scaleFactor * normed
            results[burnName] = blank
    
    return results

def normalizeNonElevations(nonDems):
    splitIndices = [0]
    validPixelsList = []
    validIndicesList = []
    names = list(nonDems.keys())
    
    #check if we have any valid pixels at all
    total_valid_pixels = 0
    for name in names:
        layer = nonDems[name]
        validIndices = np.where(np.isfinite(layer))
        validPixels = layer[validIndices]
        total_valid_pixels += len(validPixels)
    
    if total_valid_pixels == 0:
        print("Warning: No valid pixels found in any non-elevation layers, using zeros")
        results = {}
        for name in names:
            results[name] = np.zeros_like(nonDems[name], dtype=np.float32)
        return results
    
    for name in names:
        layer = nonDems[name]
        validIndices = np.where(np.isfinite(layer))
        validPixels = layer[validIndices]

        validPixelsList += validPixels.tolist()
        splitIndices.append(splitIndices[-1] + len(validPixels))
        validIndicesList.append(validIndices)

    # now layers.shape is (nburns, height, width)
    arr = np.array(validPixelsList)
    normed = util.normalize(arr)
    splitIndices = splitIndices[1:]
    splitBackUp = np.split(normed, splitIndices)
    results = {}
    for name, validIndices, normedPixels in zip(names,validIndicesList,splitBackUp):
        src = nonDems[name]
        img = np.zeros_like(src, dtype=np.float32)
        img[validIndices] = normedPixels
        results[name] = img
    return results

def getOutputs(dataset):
    result = {}
    for pt in dataset.toList(dataset.points):
        burnName, date, location = pt #location is tuples of (y, x) locations
        out = dataset.data.getOutput(burnName, date, location)
        result[(burnName, date, location)] = out

    #this prints out the date and tuples of (y, x) locations for sampled pixels
    # for loc in result:
    #     print("date ", loc[1], " pixel location: ", loc[2])

    return result

def stackAndPad(layerDict, whichLayers, dataset, AOIRadius):
    result = {}
    for burnName, date in dataset.getUsedBurnNamesAndDates():
        # guarantee that the perim mask is just 0s and 1s
        day = dataset.data.burns[burnName].days[date]
        sp = day.startingPerim
        sp[sp!=0]=1

        layers = [layerDict[layerName][burnName] for layerName in whichLayers]
        layers = [sp] + layers
        stacked = np.dstack(layers)
        r = AOIRadius
        # pad with zeros around border of image
        padded = np.pad(stacked, ((r,r),(r,r),(0,0)), 'constant')
        result[(burnName, date)] = padded
    return result

def extract(padded, location, AOIRadius):
    '''Assume padded is bordered by radius self.inputSettings.AOIRadius'''
    y,x = location
    r = AOIRadius
    lox = r+(x-r)
    hix = r+(x+r+1)
    loy = r+(y-r)
    hiy = r+(y+r+1)
    aoi = padded[loy:hiy,lox:hix]
    return aoi

# =================================================================
# utility functions

def totalPrecipitation(weatherMatrix):
    msl_pressure, temp, dewpt, temp2, wdir, wspeed, precip, hum, containment = weatherMatrix
    return sum(precip)

def averageHumidity(weatherMatrix):
    msl_pressure, temp, dewpt, temp2, wdir, wspeed, precip, hum, containment = weatherMatrix
    return sum(hum)/len(hum)

def maximumTemperature1(weatherMatrix):
    msl_pressure, temp, dewpt, temp2, wdir, wspeed, precip, hum, containment = weatherMatrix
    return max(temp)

def maximumTemperature2(weatherMatrix):
    msl_pressure, temp, dewpt, temp2, wdir, wspeed, precip, hum, containment = weatherMatrix
    return max(temp2)


def windMetrics(weatherMatrix):
    msl_pressure, temp, dewpt, temp2, wdir, wspeed, precip, hum, containment = weatherMatrix
    wDirRad = [(np.pi/180) * wDirDeg for wDirDeg in wdir]
    n, s, e, w = 0, 0, 0, 0
    for hr in range(len(wdir)):
        if wdir[hr] > 90 and wdir[hr] < 270: #from south
            s += abs(np.cos(wDirRad[hr]) * wspeed[hr])
        if wdir[hr] < 90 or wdir[hr] > 270: #from north
            n += abs(np.cos(wDirRad[hr]) * wspeed[hr])
        if wdir[hr] < 360 and wdir[hr] > 180: #from west
            w += abs(np.sin(wDirRad[hr]) * wspeed[hr])
        if wdir[hr] > 0 and wdir[hr] < 180: #from east
            e += abs(np.sin(wDirRad[hr]) * wspeed[hr])
    components = [n, s, e, w]
    return components

def containmentValue(weatherMatrix):
    msl_pressure, temp, dewpt, temp2, wdir, wspeed, precip, hum, containment = weatherMatrix
    #return the average containment value across all hours
    return sum(containment) / len(containment)

# =========================================================

if __name__ == '__main__':
    import rawdata
    import dataset
    data = rawdata.RawData.load()
    ds = dataset.Dataset(data, points=dataset.Dataset.vulnerablePixels)
    pp = PreProcessor(8, ['dem', 'ndvi', 'g'], 30)
    (inp, out), ptList = pp.process(ds)
    weather, img = inp
    print(weather[0])
    print(img[0])
    print(out[0])
