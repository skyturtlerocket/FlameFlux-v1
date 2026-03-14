#!/usr/bin/env python3.10

import sys
import os
import random
import argparse
import matplotlib.pyplot as plt
import cv2
from lib import rawdata
from lib import dataset
from lib import model
from lib import viz
from lib import preprocess
import numpy as np

def list_fires_and_dates():
    fires = [f for f in os.listdir('training_data') if os.path.isdir(os.path.join('training_data', f)) and not f.startswith('.')]
    print('Available fires:')
    total_dates = 0
    for fire in fires: 
        try:
            dates = rawdata.Day.allGoodDays(fire)
            print(f"  {fire}: {dates}")
            total_dates += len(dates)
        except Exception as e:
            print(f"  {fire}: Error reading dates ({e})")
    
    print(f"\nTotal number of dates across all fires: {total_dates}")

def run_fire_with_filename(fire_name, date, target_points=10000, eval_mode=True, model_file=None, post_process=False):
    print(f"Loading fire data for {fire_name} on date {date}...")
    
    #load data for the specified fire and date
    data = rawdata.RawData.load(burnNames=[fire_name], dates={fire_name: [date]}, inference=not eval_mode)
    
    #create dataset with vulnerable pixels around fire perimeter
    test_dataset = dataset.Dataset(data, dataset.Dataset.vulnerablePixels)
    
    print(f"Created dataset with {len(test_dataset)} total points")
    
    #sample points to reduce density while maintaining coverage
    all_points = test_dataset.toList(test_dataset.points)
    if len(all_points) > target_points:
        print(f"Sampling {target_points} points from {len(all_points)} total points (balanced sampling)")
        sampled_points = random.sample(all_points, target_points)
        test_dataset = dataset.Dataset(data, sampled_points)
    
    #show available dates
    for burnName, date in test_dataset.getUsedBurnNamesAndDates():
        points = test_dataset.points[burnName][date]
        print(f"{burnName} {date}: {len(points)} points")
    
    if model_file is None:
        model_file = "20260101-173820mod"
    print(f"Loading model: {model_file}")
    
    mod, pp = getModel(model_file, date=date)
    
    inputs, ptList = pp.process(test_dataset, inference=not eval_mode)
    flat_inputs = flatten_inputs(inputs)
    model_inputs = flat_inputs[:2]
    print("inputs type:", type(model_inputs))
    for i, arr in enumerate(model_inputs):
        print(f"Input {i}: shape={np.shape(arr)}, dtype={getattr(arr, 'dtype', type(arr))}")
    #predict
    predictions = mod.predict(model_inputs)
    
    print(f"Generated {len(predictions)} predictions")
    
    #optionally apply adaptive keep-buffer post-processing (exclude points in filter zones from metrics/viz)
    if eval_mode and post_process:
        p1 = np.asarray(predictions).flatten()
        test_dataset, predictions, ptList = _apply_post_process_filter(
            data, fire_name, date, test_dataset, ptList, p1,
            alignment=7, vulnerable_radius=dataset.Dataset.VULNERABLE_RADIUS, buffer_scale=3,
        )
        if test_dataset is None:
            print("Post-processing could not be applied (e.g. missing perims), using all points.")
        else:
            print(f"Post-process filter applied: {len(ptList)} points kept for metrics/visualization.")
    
    #calculate performance and visualize only if eval_mode
    if eval_mode:
        print("Calculating performance and generating visualizations...")
        calculatePerformance(test_dataset, predictions, target_points, ptList)
    else:
        print("Skipping performance metrics: running in inference/production mode (no ground truth available)")
    
    return test_dataset, predictions

def getModel(weightsFile=None, date=None):
    print('in getModel')
    numWeatherInputs = 9
    usedLayers = ['dem','ndvi', 'aspect', 'slope', 'band_2', 'band_3', 'band_4', 'band_5', 'hotspot']
    AOIRadius = 30
    pp = preprocess.PreProcessor(numWeatherInputs, usedLayers, AOIRadius)
    mod = model.fireCastModel(pp, weightsFile)
    return mod, pp

def calculatePerformance(test, predictions, point_count, ptList):
    fireDate = []
    samples = []
    preResu = []

    print("SIZE OF PREDICTIONS: ", len(predictions))
    for pt, pred in zip(ptList, predictions):
        fireDate.append(pt[1])
        samples.append(pt[2])
        preResu.append(pred)

    viz.getNumbers(test, samples, preResu, len(predictions), fireDate)
    res = viz.visualizePredictions(test, dict(zip(ptList, predictions)), preResu)
    savePredictionsWithFilename(res, point_count)

def savePredictionsWithFilename(predictionsRenders, point_count):
    radius = dataset.Dataset.VULNERABLE_RADIUS
    burns = {}
    for (burnName, date), render in predictionsRenders.items():
        if burnName not in burns:
            burns[burnName] = []
        burns[burnName].append((date, render))
    
    #create output/figures directory if it doesn't exist
    os.makedirs('output/figures', exist_ok=True)
    
    for burnName, frameList in burns.items():
        frameList.sort()
        fig = plt.figure(burnName, figsize=(8, 6))
        pos = (30,30)
        color = (0,0,1.0)
        size = 1
        thickness = 2
        for date, render in frameList:
            withTitle = render.copy()
            cv2.putText(withTitle,date, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness=thickness)
            im = plt.imshow(withTitle)
            fname = f"output/figures/{burnName}_{date}_radius{radius}_points{point_count}.png"
            plt.savefig(fname,bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved visualization to: {fname}")

def _apply_post_process_filter(data, fire_name, date, test_dataset, ptList, predictions,
                               alignment=7, vulnerable_radius=50, buffer_scale=3,
                               min_buffer_radius=5, max_buffer_radius=200):
    """
    Apply adaptive keep-buffer filtering (same as create_csv_prediction / compareModelAccuracy).
    Exclude points in filter_zones from metrics and visualization.
    Returns (filtered_test_dataset, filtered_predictions, filtered_ptList) or (original, original, original) on failure.
    """
    from lib.perimeter_filter import analyze_perimeter_changes
    try:
        day = data.getDay(fire_name, date)
        if day is None or day.startingPerim is None or day.endingPerim is None:
            return test_dataset, predictions, ptList
        current_mask = np.asarray(day.endingPerim).astype(bool)
        previous_mask = np.asarray(day.startingPerim).astype(bool)
        if current_mask.ndim > 2:
            current_mask = current_mask[:, :, 0]
        if previous_mask.ndim > 2:
            previous_mask = previous_mask[:, :, 0]
        H, W = current_mask.shape
        true_growth, stable_core, filter_zones, _, _ = analyze_perimeter_changes(
            current_mask, previous_mask,
            alignment_buffer_pixels=alignment,
            vulnerable_radius=vulnerable_radius,
            keep_buffer_pixels=15,
            adaptive_keep_buffer=True,
            buffer_scale=buffer_scale,
            min_buffer_radius=min_buffer_radius,
            max_buffer_radius=max_buffer_radius,
        )
        if filter_zones is None:
            return test_dataset, predictions, ptList
        kept_ptList = []
        kept_predictions = []
        for i, pt in enumerate(ptList):
            burnName, dt, (y, x) = pt
            if 0 <= y < H and 0 <= x < W and filter_zones[y, x]:
                continue
            kept_ptList.append(pt)
            kept_predictions.append(predictions[i] if hasattr(predictions, '__getitem__') else predictions[i])
        if not kept_ptList:
            return test_dataset, predictions, ptList
        points_dict = {}
        for (bn, dt, loc) in kept_ptList:
            if bn not in points_dict:
                points_dict[bn] = {}
            if dt not in points_dict[bn]:
                points_dict[bn][dt] = []
            points_dict[bn][dt].append(loc)
        test_dataset_filtered = dataset.Dataset(data, points_dict)
        pred_arr = np.asarray(kept_predictions)
        return test_dataset_filtered, pred_arr, kept_ptList
    except Exception as e:
        print(f"Post-process filter failed: {e}, using all points.")
        return test_dataset, predictions, ptList


def flatten_inputs(inputs):
    flat = []
    if isinstance(inputs, (tuple, list)):
        for item in inputs:
            if isinstance(item, (tuple, list)):
                flat.extend(flatten_inputs(item))
            else:
                flat.append(item)
    else:
        flat.append(inputs)
    return flat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run fire prediction with probability-based coloring for any fire/date')
    parser.add_argument('--fire', type=str, help='Fire name (as in training_data folder)')
    parser.add_argument('--date', type=str, help='Date in MMDD format')
    parser.add_argument('--points', type=int, default=10000, help='Number of points to sample (default: 10000)')
    parser.add_argument('--list', action='store_true', help='List available fires and dates')
    parser.add_argument('--eval', dest='eval_mode', action='store_true', help='Run in evaluation mode (requires next day perim, computes metrics)')
    parser.add_argument('--no-eval', dest='eval_mode', action='store_false', help='Run in inference/production mode (no metrics, no next day perim required)')
    parser.add_argument('--model', type=str, default='20260101-173820mod', help='Model filename (without .h5)')
    parser.add_argument('--post-process', action='store_true', dest='post_process',
                        help='Apply adaptive keep-buffer filter before metrics/viz (exclude points in filter zones)')
    parser.set_defaults(eval_mode=True)
    args = parser.parse_args()

    if args.list or (not args.fire or not args.date):
        list_fires_and_dates()
        print("\nTo run: python3.10 runPrediction.py --fire <fire_name> --date <MMDD> [--points N] [--eval/--no-eval] [--post-process]")
        sys.exit(0)

    #validate fire and date
    fires = [f for f in os.listdir('training_data') if os.path.isdir(os.path.join('training_data', f)) and not f.startswith('.')]
    if args.fire not in fires:
        print(f"Error: Fire '{args.fire}' not found in training_data folder. Use --list to see available fires.")
        sys.exit(1)
    available_dates = rawdata.Day.allGoodDays(args.fire, inference=not args.eval_mode)
    if args.date not in available_dates:
        print(f"Error: Date '{args.date}' not available for fire '{args.fire}'. Available dates: {available_dates}")
        sys.exit(1)

    run_fire_with_filename(args.fire, args.date, args.points, eval_mode=args.eval_mode,
                           model_file=args.model, post_process=args.post_process) 