
import numpy as np
import random
import sys
import os
import argparse
import json
from typing import Dict, List, Optional

from lib import rawdata, dataset, metrics, viz, preprocess, util, model

SAMPLE_SIZE = 100000

rand = False


def _parse_csv_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(',') if v.strip()]


def _load_dates_file(path: str) -> Dict[str, List[str]]:
    with open(path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("dates file must be a JSON object: {fire: [MMDD, ...], ...}")
    out: Dict[str, List[str]] = {}
    for fire, dates in data.items():
        if not isinstance(fire, str):
            raise ValueError("dates file keys must be strings (fire names)")
        if not isinstance(dates, list) or not all(isinstance(d, str) for d in dates):
            raise ValueError(f"dates for {fire} must be a list of MMDD strings")
        out[fire] = [d.zfill(4) for d in dates]
    return out


def _load_fires_file(path: str) -> List[str]:
    with open(path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("fires file must be a JSON array of strings: [\"FireA\", \"FireB\", ...]")
    return [x.strip() for x in data if x.strip()]


def _validate_training_inputs(fires: List[str], dates_map: Dict[str, List[str]], require_next_day: bool = True) -> Dict[str, List[str]]:
    """Validate existence of weather/perim files (and optionally next-day perim) for requested fire/dates.

    Returns a filtered dates_map with only valid entries. Raises if nothing is valid.
    """

    def _next_day_str(mmdd: str) -> str:
        month, day = mmdd[:2], mmdd[2:]
        next_day = str(int(day) + 1).zfill(2)
        next_date = month + next_day
        if int(day) == 31:
            next_month = str(int(month) + 1).zfill(2)
            next_date = next_month + '01'
        return next_date

    valid: Dict[str, List[str]] = {}
    problems: List[str] = []
    for fire in fires:
        requested_dates = dates_map.get(fire, [])
        if not requested_dates:
            problems.append(f"{fire}: no dates specified")
            continue

        ok_dates: List[str] = []
        for d in requested_dates:
            d = d.zfill(4)
            perim = os.path.join('training_data', fire, 'perims', f'{d}.npy')
            weather = os.path.join('training_data', fire, 'weather', f'{d}.csv')
            if not os.path.exists(perim):
                problems.append(f"{fire}/{d}: missing perim file {perim}")
                continue
            if not os.path.exists(weather):
                problems.append(f"{fire}/{d}: missing weather file {weather}")
                continue
            if require_next_day:
                nd = _next_day_str(d)
                next_perim = os.path.join('training_data', fire, 'perims', f'{nd}.npy')
                if not os.path.exists(next_perim):
                    problems.append(f"{fire}/{d}: missing next-day perim {next_perim} (needed for labels)")
                    continue
            ok_dates.append(d)

        if ok_dates:
            valid[fire] = sorted(set(ok_dates))

    if not valid:
        msg = "No valid (fire, date) training examples after validation.\n" + "\n".join(problems[:50])
        if len(problems) > 50:
            msg += f"\n... and {len(problems) - 50} more"
        raise SystemExit(msg)

    if problems:
        print("Warning: some requested fire/dates were skipped:")
        for p in problems[:20]:
            print("  -", p)
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
    return valid

def getAvailableDates():
    """Get only the dates that runPrediction.py considers 'available' (consecutive dates)"""
    available_fires = {}
    
    #get all fire directories
    fire_dirs = [d for d in os.listdir('training_data/') 
                if os.path.isdir(os.path.join('training_data', d)) and not d.startswith('_')]
    
    for fire_name in fire_dirs:
        perim_dir = os.path.join('training_data', fire_name, 'perims')
        weather_dir = os.path.join('training_data', fire_name, 'weather')
        
        if not (os.path.exists(perim_dir) and os.path.exists(weather_dir)):
            continue
            
        #get all perimeter dates
        perim_files = [f[:-4] for f in os.listdir(perim_dir) if f.endswith('.npy')]
        weather_files = [f[:-4] for f in os.listdir(weather_dir) if f.endswith('.csv')]
        
        #find dates with both perimeter and weather
        common_dates = set(perim_files) & set(weather_files)
        
        #filter for consecutive dates (same logic as runPrediction.py)
        consecutive_dates = []
        for date in sorted(common_dates):
            #check if next day exists
            month, day = date[:2], date[2:]
            next_day = str(int(day) + 1).zfill(2)
            next_date = month + next_day
            
            #handle month overflow
            if int(day) == 31:
                next_month = str(int(month) + 1).zfill(2)
                next_date = next_month + '01'
            
            if next_date in common_dates:
                consecutive_dates.append(date)
        
        if consecutive_dates:
            available_fires[fire_name] = consecutive_dates
    
    return available_fires


def getAvailableDatesForFires(fire_names: List[str]) -> Dict[str, List[str]]:
    """Return available (trainable) dates for only the requested fires.

    Uses the same logic as getAvailableDates(): a date is usable if both perim+weather exist
    and the next day also exists (so we can build labels from next-day perimeter).
    """
    all_available = getAvailableDates()
    selected: Dict[str, List[str]] = {}
    missing = []
    for fire in fire_names:
        if fire in all_available and all_available[fire]:
            selected[fire] = all_available[fire]
        else:
            missing.append(fire)

    if missing:
        print("Warning: no trainable consecutive dates found for:")
        for f in missing:
            print("  -", f)
    if not selected:
        raise SystemExit("No trainable fires found from the provided --fires list")
    return selected

def predictFires():
    #create a new Data and make burn names those three instead of all. Pass all 3 fires
    new_data = rawdata.RawData.load(burnNames='untrain', dates='all')
    newDataSet = dataset.Dataset(new_data, dataset.Dataset.vulnerablePixels)
    pointLst = newDataSet.toList(newDataSet.points)
    pointLst = random.sample(pointLst, SAMPLE_SIZE) #SAMPLE_SIZE
    test = dataset.Dataset(new_data, pointLst)
    return test

def openDatasets():
    #use only available dates (consecutive dates)
    available_fires = getAvailableDates()
    print(f"Using {len(available_fires)} fires with consecutive dates")
    
    data = rawdata.RawData.load(burnNames=list(available_fires.keys()), dates=available_fires)
    masterDataSet = dataset.Dataset(data, dataset.Dataset.vulnerablePixels)
    # print("Built master dataset")
    # print(masterDataSet.points)
    #sample vulnerable pixels for training with a reasonable limit to prevent memory issues
    #limit to 10,000 pixels per date to balance training data and memory usage (~300k total)
    #calculate total goal: 5,000 per date × number of dates (reduced for memory)
    available_fires = getAvailableDates()
    total_dates = sum(len(dates) for dates in available_fires.values())
    total_goal = 1000 * total_dates
    print(f"Sampling {5000} pixels per date across {total_dates} dates = {total_goal} total pixels")
    ptList = masterDataSet.sample(goalNumber=total_goal, sampleEvenly=False)
    print(f"Sampled {len(ptList)} vulnerable pixels for training")
    trainPts, validatePts, testPts =  util.partition(ptList, ratios=[.8,.9])
    train = dataset.Dataset(data, trainPts)
    validate = dataset.Dataset(data, validatePts)
    test = dataset.Dataset(data, testPts)
    return train, validate, test


def openDatasetsFromSelection(
    available_fires: Dict[str, List[str]],
    pixels_per_date: int = 1000,
    max_samples: Optional[int] = None,
):
    """Build train/validate/test datasets from an explicit {fire: [dates]} selection."""
    if not available_fires:
        raise SystemExit("No fires/dates provided")
    requested_fires = list(available_fires.keys())
    print(f"Using {len(requested_fires)} fires from explicit selection")

    #load data first; this may skip fires with invalid layers
    data = rawdata.RawData.load(burnNames=requested_fires, dates=available_fires)
    loaded_fires = [f for f in requested_fires if f in data.burns]
    skipped = [f for f in requested_fires if f not in data.burns]
    if skipped:
        print("Warning: these fires were skipped during load (invalid/missing layer data):")
        for f in skipped:
            print(f"  - {f}")
    if not loaded_fires:
        raise SystemExit("No fires successfully loaded; cannot train")

    available_loaded = {f: available_fires[f] for f in loaded_fires}
    total_dates = sum(len(dates) for dates in available_loaded.values())

    masterDataSet = dataset.Dataset(data, dataset.Dataset.vulnerablePixels)

    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise SystemExit("--max-samples must be a positive integer")
        #keep total goal even (Dataset.sample requirement) and distribute evenly across fires
        if max_samples % 2 == 1:
            max_samples -= 1
        n_fires = len(loaded_fires)
        per_fire_goal = max_samples // n_fires
        if per_fire_goal % 2 == 1:
            per_fire_goal -= 1
        if per_fire_goal <= 0:
            raise SystemExit("--max-samples is too small to allocate an even sample per fire")

        print(
            f"Sampling capped by --max-samples={max_samples} and distributed evenly across {n_fires} fires: {per_fire_goal} samples/fire"
        )
        if total_dates:
            print(f"Total trainable dates (fire-days) after load: {total_dates}")

        per_fire_actual = {}
        all_pts = []
        for fire in loaded_fires:
            #build a per-fire dataset that still uses the shared RawData object
            fire_points = {fire: masterDataSet.points[fire]}
            fire_ds = dataset.Dataset(data, fire_points)

            #compute per-fire capacity and adjust goal if needed
            day2 = fire_ds.makeDay2burnedNotBurnedMap()
            limits = {day: min(len(yes), len(no)) for day, (yes, no) in day2.items()}
            capacity = sum(limits.values()) * 2
            goal = min(per_fire_goal, capacity)
            if goal % 2 == 1:
                goal -= 1
            if goal <= 0:
                print(f"Warning: {fire} has no capacity for balanced sampling; skipping")
                per_fire_actual[fire] = 0
                continue

            pts = fire_ds.sample(goalNumber=goal, sampleEvenly=False)
            per_fire_actual[fire] = len(pts)
            all_pts.extend(pts)

        random.shuffle(all_pts)
        ptList = all_pts
        print(f"Sampled {len(ptList)} total points across fires")
        for fire in sorted(per_fire_actual, key=per_fire_actual.get, reverse=True):
            print(f"  - {fire}: {per_fire_actual[fire]}")
    else:
        total_goal = int(pixels_per_date) * total_dates
        if total_goal % 2 == 1:
            total_goal -= 1
        print(f"Sampling {pixels_per_date} pixels per date across {total_dates} dates = {total_goal} total pixels")
        ptList = masterDataSet.sample(goalNumber=total_goal, sampleEvenly=False)
        print(f"Sampled {len(ptList)} vulnerable pixels for training")

    trainPts, validatePts, testPts = util.partition(ptList, ratios=[.8, .9])
    train = dataset.Dataset(data, trainPts)
    validate = dataset.Dataset(data, validatePts)
    test = dataset.Dataset(data, testPts)
    return train, validate, test

def openAndPredict(weightsFile):
    from lib import model

    test = predictFires()
    test.save('testOtherFire')
    mod = getModel(weightsFile)
    predictions = mod.predict(test)
    util.savePredictions(predictions)
    res = viz.visualizePredictions(test, predictions)
    viz.showPredictions(res)
    return test, predictions

def openAndTrain():
    from lib import model

    print("=== TRAINING NEW MODEL WITH CONTAINMENT + HOTSPOTS ===")
    print("Architecture:")
    print("- Weather inputs: 9 (includes containment)")
    print("- Spatial layers: 9 (8 original + hotspot)")
    print("- Total channels: 10 (9 spatial + 1 perimeter mask)")
    print("- AOI radius: 30 pixels (61x61 patches)")
    
    #create fresh datasets from training_data
    train, validate, test = openDatasets()
    
    #save the datasets for future use
    train.save('train_with_containment_hotspots')
    validate.save('validate_with_containment_hotspots')
    test.save('test_with_containment_hotspots')

    #validate that all required data exists
    available_fires = getAvailableDates()
    
    #check for missing hotspot files
    missing_hotspots = []
    for fire_name, dates in available_fires.items():
        for date in dates:
            hotspot_path = f'training_data/{fire_name}/hotspots/{date}.npy'
            if not os.path.exists(hotspot_path):
                missing_hotspots.append(f"{fire_name}/{date}")
    
    if missing_hotspots:
        print(f"Warning: Missing hotspot files for {len(missing_hotspots)} fire/date combinations:")
        for missing in missing_hotspots[:10]:  # Show first 10
            print(f"  - {missing}")
        if len(missing_hotspots) > 10:
            print(f"  ... and {len(missing_hotspots) - 10} more")
        
        print("Creating empty hotspot files for missing data...")
        for fire_name, dates in available_fires.items():
            for date in dates:
                hotspot_path = f'training_data/{fire_name}/hotspots/{date}.npy'
                if not os.path.exists(hotspot_path):
                    perim_path = f'training_data/{fire_name}/perims/{date}.npy'
                    if os.path.exists(perim_path):
                        perim_shape = np.load(perim_path).shape
                        empty_hotspot = np.zeros(perim_shape, dtype=np.float32)
                        os.makedirs(os.path.dirname(hotspot_path), exist_ok=True)
                        np.save(hotspot_path, empty_hotspot)
                        print(f"  Created empty hotspot: {hotspot_path}")

    print("All fires now have hotspot data for their respective dates")

    #create model with new architecture
    mod, pp = getModel()

    #verify the model architecture
    print(f"\nModel Architecture Verification:")
    print(f"- Preprocessor expects {pp.numWeatherInputs} weather inputs")
    print(f"- Preprocessor expects {len(pp.whichLayers)} spatial layers: {pp.whichLayers}")
    print(f"- AOI radius: {pp.AOIRadius} (creates {2*pp.AOIRadius+1}x{2*pp.AOIRadius+1} patches)")

    #train the model
    print(f"\nStarting training...")
    model.fireCastFit(mod, pp, train, validate, epochs=25)
    
    #test the model
    print(f"\nTesting trained model...")
    predictions, _ = model.fireCastPredict(mod, pp, test, rand)
    calculatePerformance(test, predictions)

    return test, predictions


def openAndTrainSelected(
    available_fires: Dict[str, List[str]],
    epochs: int = 25,
    pixels_per_date: int = 1000,
    max_samples: Optional[int] = None,
    use_v2: bool = False,
    skip_test: bool = False,
):
    from lib import model

    if use_v2:
        model_version = "V2 (L2 reg + fusion + callbacks)"
    else:
        model_version = "V1 (original)"
    print(f"=== TRAINING NEW MODEL - {model_version} ===")
    print(f"Fires: {len(available_fires)}")
    print(f"Total dates: {sum(len(v) for v in available_fires.values())}")
    print(f"Epochs: {epochs}" + (" (max, with EarlyStopping)" if use_v2 else ""))
    print(f"Pixels per date: {pixels_per_date}")
    if max_samples is not None:
        print(f"Max samples cap: {max_samples}")

    train, validate, test = openDatasetsFromSelection(
        available_fires,
        pixels_per_date=pixels_per_date,
        max_samples=max_samples,
    )

    #validate/ensure hotspot files exist (empty fallback is OK)
    missing_hotspots = []
    for fire_name, dates in available_fires.items():
        for date in dates:
            hotspot_path = f'training_data/{fire_name}/hotspots/{date}.npy'
            if not os.path.exists(hotspot_path):
                missing_hotspots.append(f"{fire_name}/{date}")

    if missing_hotspots:
        print(f"Warning: Missing hotspot files for {len(missing_hotspots)} fire/date combinations.")
        print("Creating empty hotspot files for missing data...")
        for fire_name, dates in available_fires.items():
            for date in dates:
                hotspot_path = f'training_data/{fire_name}/hotspots/{date}.npy'
                if not os.path.exists(hotspot_path):
                    perim_path = f'training_data/{fire_name}/perims/{date}.npy'
                    if os.path.exists(perim_path):
                        perim_shape = np.load(perim_path).shape
                        empty_hotspot = np.zeros(perim_shape, dtype=np.float32)
                        os.makedirs(os.path.dirname(hotspot_path), exist_ok=True)
                        np.save(hotspot_path, empty_hotspot)

    #select model architecture
    if use_v2:
        mod, pp = getModelV2()
    else:
        mod, pp = getModel()
        
    print(f"\nModel Architecture Verification:")
    print(f"- Preprocessor expects {pp.numWeatherInputs} weather inputs")
    print(f"- Preprocessor expects {len(pp.whichLayers)} spatial layers: {pp.whichLayers}")
    print(f"- AOI radius: {pp.AOIRadius} (creates {2*pp.AOIRadius+1}x{2*pp.AOIRadius+1} patches)")

    print("\nStarting training...")

    if use_v2:
        #v2: Use callbacks (EarlyStopping, ReduceLROnPlateau)
        history, model_path = model.fireCastFitWithCallbacks(
            mod, pp, train, validate, 
            max_epochs=int(epochs), 
            batch_size=32
        )
        print(f"\n=== V2 TRAINING SUMMARY ===")
        print(f"Final train loss: {history.history['loss'][-1]:.4f}")
        print(f"Final val loss: {history.history['val_loss'][-1]:.4f}")
        print(f"Final train acc: {history.history['accuracy'][-1]:.4f}")
        print(f"Final val acc: {history.history['val_accuracy'][-1]:.4f}")
        print(f"Best model saved to: {model_path}")
    else:
        #v1: Original training
        use_streaming = max_samples is not None
        if use_streaming:
            print("Using streaming training (generator) to reduce RAM usage")
            model.fireCastFitStreaming(mod, pp, train, validate, epochs=int(epochs), batch_size=32, shuffle=True)
        else:
            model.fireCastFit(mod, pp, train, validate, epochs=int(epochs))

    if skip_test:
        print("\nSkipping post-training testing phase (--skip-test flag)")
        return None, None
    
    print("\nTesting trained model...")
    predictions, _ = model.fireCastPredict(mod, pp, test, rand)
    calculatePerformance(test, predictions)
    return test, predictions

def calculatePerformance(test, predictions):

    fireDate = []
    samples = []
    preResu = []

    print("SIZE OF PREDICTIONS: " , len(predictions))
    for pre in predictions:
        fireDate.append(pre[1])
        samples.append(pre[2])
        preResu.append(predictions.get(pre))
    viz.getNumbers(test, samples, preResu, len(predictions), fireDate)
    res = viz.visualizePredictions(test, predictions, preResu)
    viz.showPredictions(res)



def reloadPredictions():
    predFName = "09Nov10:39.csv"
    predictions = util.openPredictions('output/predictions/'+predFName)
    test = dataset.Dataset.open("output/datasets/test.json")
    return test, predictions


def getModel(weightsFile=None, date=None):
    print('in getModel')
    numWeatherInputs = 9  # CHANGED FROM 8 TO 9 (includes containment)
    if date:
        #for specific date prediction, use date-specific hotspot layer
        usedLayers = ['dem','ndvi', 'aspect', 'slope', 'band_2', 'band_3', 'band_4', 'band_5', f'hotspot_{date}']
    else:
        #for training, use generic hotspot layer name (will be mapped to actual hotspot data)
        usedLayers = ['dem','ndvi', 'aspect', 'slope', 'band_2', 'band_3', 'band_4', 'band_5', 'hotspot']
    
    print(f"DEBUG: Using {numWeatherInputs} weather inputs and {len(usedLayers)} spatial layers")
    print(f" : Spatial layers: {usedLayers}")
    
    AOIRadius = 30  # 61x61 area of interest (30*2 + 1 = 61)
    pp = preprocess.PreProcessor(numWeatherInputs, usedLayers, AOIRadius)

    mod = model.fireCastModel(pp, weightsFile)
    return mod, pp


def getModelV2(weightsFile=None, date=None):
    """
    Get the V2 model with L2 regularization and fusion layer.
    """
    print('in getModelV2')
    numWeatherInputs = 9  # CHANGED FROM 8 TO 9 (includes containment)
    if date:
        usedLayers = ['dem','ndvi', 'aspect', 'slope', 'band_2', 'band_3', 'band_4', 'band_5', f'hotspot_{date}']
    else:
        usedLayers = ['dem','ndvi', 'aspect', 'slope', 'band_2', 'band_3', 'band_4', 'band_5', 'hotspot']
    
    print(f"V2 Model: Using {numWeatherInputs} weather inputs and {len(usedLayers)} spatial layers")
    print(f"  Spatial layers: {usedLayers}")
    print(f"  V2 Features: L2 regularization (1e-4), fusion layer (64 units), LR=5e-4")
    
    AOIRadius = 30  # 61x61 area of interest (30*2 + 1 = 61)
    pp = preprocess.PreProcessor(numWeatherInputs, usedLayers, AOIRadius)

    mod = model.fireCastModelV2(pp, weightsFile)
    return mod, pp


def example():
    try:
        modfname = sys.argv[1]
        datasetfname = sys.argv[2]
        print("working")
    except:
        print('about to import tkinter')
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
        print('done!')
        root = Tk()
        print('Tked')
        root.withdraw()
        print('withdrawn')
        modfname = askopenfilename(initialdir = "models/",title="choose a model")
        datasetfname = askopenfilename(initialdir = "output/datasets",title="choose a dataset")
        root.destroy()

    rand = False

    test = dataset.openDataset(datasetfname)
    mod,pp = getModel(modfname)

    predictions, _ = model.fireCastPredict(mod, pp, test, rand)
    calculatePerformance(test, predictions)



#uncomment openAndTrain() to train a new model
#if you create a new dataset, must change name to have "_" instead of "/" for it to work with example()
# openAndTrain()

#uncomment the next two lines to create a validation dataset
# test = predictFires()
# dataset.Dataset.save(test)

# openAndPredict('') #enter weightsFile

#uncomment example() to make test images appear
#To run: python3 main.py models/model_name output/datasets/dataset_name
#final model: 11Apr17_55.h5

def _run_cli_train(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(description="Train a new FireCast model and save a timestamped .h5 under models/.")
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--fires', default='', help='Comma-separated fire folder names under training_data (e.g. "beaverCreek,MAX")')
    parser.add_argument('--fires-file', default='', help='JSON file containing a list of fire folder names (e.g. training_fires.json). Merged with --fires.')
    parser.add_argument('--dates', default='', help='Comma-separated MMDD dates to use for ALL selected fires (e.g. "0711,0712"). If omitted, all available dates for each fire are used.')
    parser.add_argument('--dates-file', default='', help='JSON file mapping fire -> [MMDD, ...]. Overrides --dates. If omitted, all available dates for each fire are used.')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--pixels-per-date', type=int, default=1000, help='How many vulnerable pixels to sample per (fire,date).')
    parser.add_argument('--max-samples', type=int, default=None, help='Hard cap on total sampled points across all (fire,date). If set, overrides --pixels-per-date to stay under the cap.')
    parser.add_argument('--model-v2', action='store_true', help='Use improved model architecture (V2) with L2 regularization, fusion layer, EarlyStopping and ReduceLROnPlateau callbacks')
    parser.add_argument('--skip-test', action='store_true', help='Skip the post-training testing phase')
    args = parser.parse_args(argv)

    if not args.train:
        raise SystemExit("Nothing to do. Use --train (or run with 2 positional args for example mode).")

    fires = []
    if args.fires_file:
        fires.extend(_load_fires_file(args.fires_file))
    fires.extend(_parse_csv_list(args.fires))
    # de-dupe while preserving order
    seen = set()
    fires = [f for f in fires if not (f in seen or seen.add(f))]
    if not fires:
        raise SystemExit("--fires and/or --fires-file is required for training selection mode")

    if args.dates_file:
        dates_map = _load_dates_file(args.dates_file)
        valid = _validate_training_inputs(fires, dates_map, require_next_day=True)
    else:
        dates = _parse_csv_list(args.dates)
        if dates:
            dates_map = {fire: [d.zfill(4) for d in dates] for fire in fires}
            valid = _validate_training_inputs(fires, dates_map, require_next_day=True)
        else:
            #convenience mode: only fires provided, auto-pull all available trainable dates per fire
            valid = getAvailableDatesForFires(fires)

    openAndTrainSelected(
        valid, 
        epochs=args.epochs, 
        pixels_per_date=args.pixels_per_date, 
        max_samples=args.max_samples,
        use_v2=args.model_v2,
        skip_test=args.skip_test
    )


if __name__ == "__main__":
    #backward compatible behavior:
    # - 2 positional args => example mode
    # - any flags (e.g. --train ...) => argparse mode
    if any(a.startswith('-') for a in sys.argv[1:]):
        _run_cli_train(sys.argv[1:])
    elif len(sys.argv) == 1:
        print('Training a new model...')
        openAndTrain()
    elif len(sys.argv) == 3:
        example()
    elif len(sys.argv) == 2:
        print('Please include a model and a dataset.')
    else:
        print(">>> Making a new dataset")
        test = predictFires()
        dataset.Dataset.save(test, fname='beaverCreek')
