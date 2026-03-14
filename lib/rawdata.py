
from os import listdir
import numpy as np
import cv2

from lib import util

PIXEL_SIZE = 30

untrain = False

class RawData(object):

    def __init__(self, burns):
        self.burns = burns

    @staticmethod
    def load(burnNames='all', dates='all', inference=False):
        print("in rawdata load")
        untrain = False
        if burnNames == 'all':
            print('1')
            burnNames = listdir_nohidden('training_data/')
            print('1')
        if burnNames == 'untrain':
            print('2')
            untrain = True
            burnNames = listdir_nohidden('training_data/_untrained/')
            print('2')
        if dates == 'all':
            print('3')
            burns = {}
            for i, n in enumerate(burnNames):
                print(f"  Processing fire {i+1}/{len(burnNames)}: {n}")
                burn = Burn.load(n, untrain, 'all', inference=inference)
                if burn is not None:
                    burns[n] = burn
                else:
                    print(f"  Skipped fire {n} due to invalid data")
            print('3')
        else:
            # assumes dates is a dict, with keys being burnNames and vals being dates
            print('4')
            burns = {}
            for i, n in enumerate(burnNames):
                print(f"  Processing fire {i+1}/{len(burnNames)}: {n} ({len(dates[n])} dates)")
                burn = Burn.load(n, untrain, dates[n], inference=inference)
                if burn is not None:
                    burns[n] = burn
                else:
                    print(f"  Skipped fire {n} due to invalid data")
            print('4')
        return RawData(burns)

    def getWeather(self, burnName, date):
        burn = self.burns[burnName]
        day = burn.days[date]
        return day.weather

    def getOutput(self, burnName, date, location):
        burn = self.burns[burnName]
        day = burn.days[date]
        return day.endingPerim[location]

    def getDay(self, burnName, date):
        return self.burns[burnName].days[date]

    def __repr__(self):
        return "Dataset({})".format(list(self.burns.values()))

class Burn(object):

    def __init__(self, name, days, untrain=False, layers=None):
        self.name = name
        self.days = days
        self._untrain = untrain
        self.layers = layers if layers is not None else self.loadLayers()

        # what is the height and width of a layer of data
        self.layerSize = list(self.layers.values())[0].shape[:2]
        
        #add hotspot layers for each date
        self.addHotspotLayers()

    def loadLayers(self):
        folder = 'training_data/{}/'.format(self.name)
        if self._untrain:
            folder = 'training_data/_untrained/{}/'.format(self.name)
        
        #check if DEM is valid before loading other layers
        try:
            dem = util.openImg(folder+'dem.npy')
            #check if DEM has valid data (not all zeros, has finite values, has variation)
            if dem is None or np.all(dem == 0) or not np.any(np.isfinite(dem)) or np.std(dem) == 0:
                print(f"Warning: Invalid DEM data for {self.name}, skipping this fire")
                raise ValueError(f"Invalid DEM data for {self.name}")
        except Exception as e:
            print(f"Error loading DEM for {self.name}: {e}")
            raise ValueError(f"Cannot load valid DEM for {self.name}")
        
        slope = util.openImg(folder+'slope.npy')
        band_2 = util.openImg(folder+'band_2.npy')
        band_3 = util.openImg(folder+'band_3.npy')
        band_4 = util.openImg(folder+'band_4.npy')
        band_5 = util.openImg(folder+'band_5.npy')
        ndvi = util.openImg(folder+'ndvi.npy')
        aspect = util.openImg(folder+'aspect.npy')

        layers = {'dem':dem,
                'slope':slope,
                'ndvi':ndvi,
                'aspect':aspect,
                'band_4':band_4,
                'band_3':band_3,
                'band_2':band_2,
                'band_5':band_5}

        # ok, now we have to make sure that all of the NoData values are set to 0
        #the NV pixels occur outside of our AOIRadius
        #when exported from GIS they could take on a variety of values
        # susceptible = ['dem', 'r','g','b','nir',]
        for name, layer in layers.items():
            pass
        return layers

    def addHotspotLayers(self):
        '''Add hotspot layers for each date to the layers dictionary'''
        for date, day in self.days.items():
            try:
                hotspot_data = day.loadHotspotData()
                #add hotspot layer with date-specific name
                self.layers[f'hotspot_{date}'] = hotspot_data
                #also add a generic 'hotspot' layer for training compatibility
                self.layers['hotspot'] = hotspot_data
            except Exception as e:
                print(f"Warning: Could not load hotspot data for {self.name} {date}: {e}")
                #create empty hotspot layer
                empty_hotspot = np.zeros(self.layerSize, dtype=np.float32)
                self.layers[f'hotspot_{date}'] = empty_hotspot
                self.layers['hotspot'] = empty_hotspot

    @staticmethod
    def load(burnName, untrain=False, dates='all', inference=False):
        # print("Loading: ", burnName)
        # print("in load")
        # print(dates)
        if dates == 'all':
            dates = Day.allGoodDays(burnName, untrain, inference=inference)

        try:
            days = {date:Day(burnName, date, untrain, inference=inference) for date in dates}
            return Burn(burnName, days, untrain)
        except ValueError as e:
            print(f"Skipping fire {burnName} due to invalid data: {e}")
            return None

    def __repr__(self):
        return "Burn({}, {})".format(self.name, [d.date for d in self.days.values()])

class Day(object):

    def __init__(self, burnName, date, untrain=False, weather=None, startingPerim=None, endingPerim=None, inference=False):
        self.burnName = burnName
        self.date = date
        self.untrain = untrain
        self.weather = weather             if weather       is not None else self.loadWeather()
        self.startingPerim = startingPerim if startingPerim is not None else self.loadStartingPerim()
        if not inference:
            self.endingPerim = endingPerim     if endingPerim   is not None else self.loadEndingPerim()
            self.previousPerim = self._loadPreviousPerim()
        else:
            self.endingPerim = None
            self.previousPerim = None

    def loadWeather(self):
        fname = 'training_data/{}/weather/{}.csv'.format(self.burnName, self.date)
        if self.untrain:
            fname = 'training_data/_untrained/{}/weather/{}.csv'.format(self.burnName, self.date)

        #load CSV with pandas to handle the new format with containment
        import pandas as pd
        df = pd.read_csv(fname)
        
        print(f"DEBUG: Processing {self.burnName} {self.date}")
        
        #handle the specific column structure from your CSV
        #don't strip spaces - use exact column names to avoid duplicates
        weather_cols = ['MSL PRESSURE   ',    # Keep original spacing
                    'TEMPERATURE   ',      # First temperature column
                    'DEW POINT      ',
                    ' TEMPERATURE   ',     # Second temperature column (with leading space)
                    'WIND DIRECTION ',
                    'WIND SPEED     ',
                    ' PRECIPITATION',
                    'RELATIVE HUMIDITY']
        
        #verify all columns exist
        missing_cols = [col for col in weather_cols if col not in df.columns]
        if missing_cols:
            print(f"ERROR: Missing columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Missing weather columns: {missing_cols}")
        
        #get containment percentage
        containment = df['CONTAINMENT'].iloc[0] if 'CONTAINMENT' in df.columns else 0.0
        print(f"DEBUG: Containment: {containment}")
        
        #extract weather data as numpy array (transpose to get hourly data as columns)
        weather_data = df[weather_cols].values.T  # This will be 8 rows
        print(f"DEBUG: Weather data shape: {weather_data.shape} (should be 8 x 25)")
        
        #add containment as a new row (repeat for each hour)
        containment_row = np.full((1, weather_data.shape[1]), containment)
        weather_matrix = np.vstack([weather_data, containment_row])  # Now 9 rows
        
        print(f"DEBUG: Final weather matrix shape: {weather_matrix.shape} (should be 9 x 25)")
        
        return weather_matrix

    def loadHotspotData(self):
        '''Load hotspot data for this specific date'''
        try:
            fname = 'training_data/{}/hotspots/{}.npy'.format(self.burnName, self.date)
            if self.untrain:
                fname = 'training_data/_untrained/{}/hotspots/{}.npy'.format(self.burnName, self.date)
            
            hotspot_data = np.load(fname)
            
            #ensure it's the right shape and type
            if len(hotspot_data.shape) != 2:
                raise ValueError(f"Hotspot data should be 2D, got shape {hotspot_data.shape}")
            
            return hotspot_data
        except Exception as e:
            #if hotspot data doesn't exist, create empty layer
            print(f"Warning: No hotspot data for {self.burnName} {self.date}, using empty layer")
            #get the shape from the starting perimeter
            perim_shape = self.startingPerim.shape
            return np.zeros(perim_shape, dtype=np.float32)

    def loadStartingPerim(self):
        # fname = 'training_data/{}/perims/{}.tif'.format(self.burnName, self.date)
        # perim = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        fname = 'training_data/{}/perims/{}.npy'.format(self.burnName, self.date)
        if self.untrain:
            fname = 'training_data/_untrained/{}/perims/{}.npy'.format(self.burnName, self.date)

        perim = np.load(fname)
        if perim is None:
            raise RuntimeError('Could not find a perimeter for the fire {} for the day {}'.format(self.burnName, self.date))
        # perim[perim!=0] = 255
        perim[perim!=1] = 0
        return perim

    def loadEndingPerim(self):
        guess1, guess2 = Day.nextDay(self.date)
        # fname = 'data/{}/perims/{}.tif'.format(self.burnName, guess1)
        # perim = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        try:
            fname = 'training_data/{}/perims/{}.npy'.format(self.burnName, guess1)
            if self.untrain:
                fname = 'training_data/_untrained/{}/perims/{}.npy'.format(self.burnName, guess1)
            perim = np.load(fname)
        except:
        # if perim is None:
            # overflowed the month, that file didnt exist
            # fname = 'training_data/{}/perims/{}.tif'.format(self.burnName, guess2)
            # perim = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            fname = 'training_data/{}/perims/{}.npy'.format(self.burnName, guess2)
            if self.untrain:
                fname = 'training_data/_untrained/{}/perims/{}.npy'.format(self.burnName, guess2)
            perim = np.load(fname)
            if perim is None:
                raise RuntimeError('Could not open a perimeter for the fire {} for the day {} or {}'.format(self.burnName, guess1, guess2))


        perim[perim!=1] = 0
        return perim

    def _loadPreviousPerim(self):
        """Load previous day's perimeter (in-memory only, for prev_growth_indicator)."""
        guess1, guess2 = Day.prevDay(self.date)
        for guess in (g for g in (guess1, guess2) if g is not None):
            try:
                fname = 'training_data/{}/perims/{}.npy'.format(self.burnName, guess)
                if self.untrain:
                    fname = 'training_data/_untrained/{}/perims/{}.npy'.format(self.burnName, guess)
                perim = np.load(fname)
                if perim is not None and perim.shape == self.startingPerim.shape:
                    perim = perim.copy()
                    perim[perim != 1] = 0
                    return perim
            except Exception:
                pass
        return None

    def __repr__(self):
        return "Day({},{})".format(self.burnName, self.date)

    @staticmethod
    def nextDay(dateString):
        month, day = dateString[:2], dateString[2:]

        nextDay = str(int(day)+1).zfill(2)
        guess1 = month+nextDay

        nextMonth = str(int(month)+1).zfill(2)
        guess2 = nextMonth+'01'

        return guess1, guess2

    @staticmethod
    def prevDay(dateString):
        """Return (prev_date1, prev_date2) for previous calendar day (handles month boundary)."""
        month, day = dateString[:2], dateString[2:]
        if int(day) > 1:
            prev_d = str(int(day) - 1).zfill(2)
            return month + prev_d, None
        prev_month = str(int(month) - 1).zfill(2) if int(month) > 1 else '12'
        last_days = {'01': '31', '02': '28', '03': '31', '04': '30', '05': '31', '06': '30',
                     '07': '31', '08': '31', '09': '30', '10': '31', '11': '30', '12': '31'}
        prev_last = last_days.get(prev_month, '31')
        return prev_month + prev_last, None

    @staticmethod
    def allGoodDays(burnName, untrain=False, inference=False):
        '''Given a fire, return a list of all dates that we can train on (or predict on, if inference=True)'''
        if untrain:
            directory = 'training_data/_untrained/{}/'.format(burnName)
        else:
            directory = 'training_data/{}/'.format(burnName)

        weatherFiles = listdir_nohidden(directory+'weather/')
        weatherDates = [fname[:-len('.csv')] for fname in weatherFiles]

        perimFiles = listdir_nohidden(directory+'perims/')
        perimDates = [fname[:-len('.npy')] for fname in perimFiles if isValidImg(directory+'perims/'+fname)]

        if inference:
            #in inference mode, allow any day with both a perimeter and weather for the current day
            daysWithWeatherAndPerims = [d for d in perimDates if d in weatherDates]
            daysWithWeatherAndPerims.sort()
            return daysWithWeatherAndPerims
        else:
            # we can only use days which have perimeter data on the following day
            daysWithFollowingPerims = []
            for d in perimDates:
                nextDay1, nextDay2 = Day.nextDay(d)
                if nextDay1 in perimDates or nextDay2 in perimDates:
                    daysWithFollowingPerims.append(d)

            # now we have to verify that we have weather for these days as well
            daysWithWeatherAndPerims = [d for d in daysWithFollowingPerims if d in weatherDates]
            daysWithWeatherAndPerims.sort()
            return daysWithWeatherAndPerims

def isValidImg(imgName):
    if imgName.endswith('.npy'):
        try:
            arr = np.load(imgName)
            return arr is not None
        except Exception as e:
            print(f"Failed to load npy file {imgName}: {e}")
            return False
    else:
        img = cv2.imread(imgName, cv2.IMREAD_UNCHANGED)
        return img is not None

def listdir_nohidden(path):
    '''List all the files in a path that are not hidden (begin with a .)'''
    result = []

    for f in listdir(path):
        if not f.startswith('.') and not f.startswith("_"):
            result.append(f)
    return result

if __name__ == '__main__':
    raw = RawData.load()
    print(raw.burns['riceRidge'].days['0731'].weather)
