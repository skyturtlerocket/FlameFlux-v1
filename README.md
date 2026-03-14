# FlameFlux

Wildfire spread prediction model for integration with the **Space Science Institute** wildfire prediction system. This repository contains the FlameFlux model, training pipeline, data ingestion, and prediction scripts.

---

## Overview

- **Main model:** `models/20260101-173820mod.h5`
- **Training entry point:** `main.py`
- **Model and data pipeline code:** `lib/`
- **Training data (reference):** `0101_training_data/` — fire-specific folders with perimeters, weather, and terrain
- **New/ingested fire data:** `training_data/` — populated by `getData.py`, used by prediction scripts

---

## Repository layout

```
FlameFlux/
├── main.py                 # Model training
├── getData.py              # Fetch new fire data from NIFC/WFIGS API
├── create_csv_prediction.py # Run predictions on recent fires → CSV + images
├── runPrediction.py        # Single fire/date prediction (eval or inference), figures + metrics
├── runProduction.py        # Batch production inference (all fires or one fire/date)
├── view_npy.py             # View .npy training/perimeter data
├── lib/                    # Model and data pipeline
│   ├── model.py            # Model architecture
│   ├── preprocess.py       # Preprocessing and spatial features
│   ├── rawdata.py          # Raw data loading (perims, weather, layers)
│   ├── dataset.py          # Dataset and vulnerable-pixel sampling
│   ├── viz.py              # Visualization helpers
│   ├── util.py
│   ├── metrics.py
│   ├── perimeter_filter.py
│   └── ...
├── models/                 # Saved .h5 models
│   └── 20260101-173820mod.h5
├── 0101_training_data/     # Reference training data (fire folders)
├── training_data/          # New fire data (from getData.py)
├── output/                 # All script outputs
│   ├── csv/                # create_csv_prediction.py CSVs
│   ├── csv_images/         # create_csv_prediction.py PNGs
│   ├── figures/            # runPrediction.py figures
│   ├── images/             # runProduction.py perimeter viz
│   ├── predictions_*.csv   # runProduction.py per-fire/date predictions
│   ├── predicted_perimeter_*.geojson
│   ├── predicted_perimeter_*_overlay.png
│   ├── runProduction_*.log
│   └── ...
└── requirements.txt
```

---

## Setup

```bash
python3 -m venv myvenv
source myvenv/bin/activate   # or: myvenv\Scripts\activate on Windows
pip install -r requirements.txt
```

Use Python 3.10 for scripts that specify it (e.g. `python3.10` in examples below).

**Environment variables (do not commit secrets):**

- **`NASA_FIRMS_API_KEY`** — Required by `getData.py` for hotspot data. Get a key at [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/). Set locally, e.g. `export NASA_FIRMS_API_KEY=your_key`.
- **`EARTHENGINE_PROJECT`** — Optional. Google Earth Engine / GCP project ID for `getData.py`. If unset, Earth Engine uses your default project after `ee.Authenticate()`.

---

## Scripts and usage

### 1. View training / perimeter data (`view_npy.py`)

Inspect `.npy` arrays (terrain, perimeters) under `data/`, `training_data/`, or `0101_training_data/`.

```bash
# List available fires and example files
python3 view_npy.py --list

# View a specific file by path
python3 view_npy.py --file training_data/Yellow/perims/0311.npy
python3 view_npy.py --file 0101_training_data/SomeFire/dem.npy

# By fire + type (uses directory "data/" by default)
python3 view_npy.py --fire beaverCreek --type dem
python3 view_npy.py --fire beaverCreek --type ndvi --cmap RdYlGn

# Perimeter for a date
python3 view_npy.py --fire beaverCreek --date 0711

# Custom colormap
python3 view_npy.py --file training_data/Yellow/ndvi.npy --cmap RdYlGn
```

---

### 2. Fetch new fire data (`getData.py`)

Pulls current perimeters and metadata from the NIFC WFIGS API, then fetches Landsat/terrain and builds fire folders under **`training_data/`**. Only fires updated in the last 24 hours are processed.

```bash
# Fetch all eligible recent fires into training_data/
python3 getData.py
```

New data layout per fire: `training_data/<Fire_Name>/` with `perims/`, `weather/`, `hotspots/`, terrain `.npy` files, and `center.json`.

---

### 3. Model training (`main.py`)

Train a new model; outputs a timestamped `.h5` under `models/`.

```bash
# Train on all available fires/dates (auto-discovered from training_data)
python3 main.py

# Train with explicit fire selection and options
python3 main.py --train --fires "Yellow,Cherry" --epochs 25 --pixels-per-date 1000

# Use a fires list from a JSON file (and optional dates file)
python3 main.py --train --fires-file training_fires.json --dates-file training_dates.json

# Limit total samples (e.g. for memory)
python3 main.py --train --fires "Yellow" --max-samples 50000 --epochs 20
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--train` | Run training (required for CLI training mode) |
| `--fires` | Comma-separated fire folder names under `training_data` |
| `--fires-file` | JSON file: list of fire names |
| `--dates` | Comma-separated MMDD dates for all selected fires |
| `--dates-file` | JSON file: `{"FireName": ["MMDD", ...]}` |
| `--epochs` | Training epochs (default: 25) |
| `--pixels-per-date` | Vulnerable pixels to sample per (fire, date) (default: 1000) |
| `--max-samples` | Cap total sampled points (overrides pixels-per-date) |
| `--skip-test` | Skip post-training test phase |

---

### 4. Run predictions on recent fires → CSV + images (`create_csv_prediction.py`)

Fetches recent fires from the NIFC API, runs the model on fires that exist in `training_data/`, applies perimeter-based filtering, and writes **`output/csv/<fire>.csv`** and **`output/csv_images/<fire>.png`**.

No CLI arguments: it uses built-in API URL and 24‑hour recency + 100‑acre minimum.

```bash
python3 create_csv_prediction.py
```

Logs and errors go to timestamped files under `output/` (e.g. `getCSVPredictions_*.log`).

---

### 5. Single fire/date prediction and evaluation (`runPrediction.py`)

Run the model for one fire and one date. **Eval mode** (default) requires the next-day perimeter and computes metrics; **inference mode** does not. Outputs go to **`output/figures/`**.

```bash
# List available fires and dates (from training_data)
python3 runPrediction.py --list

# Eval mode (default): need next-day perim, get metrics + figure
python3 runPrediction.py --fire Yellow --date 0311

# Inference only (no ground truth, no next-day perim)
python3 runPrediction.py --fire Yellow --date 0311 --no-eval

# Custom point count and model
python3 runPrediction.py --fire Yellow --date 0311 --points 5000 --model 20260101-173820mod

# Apply adaptive keep-buffer post-processing before metrics/viz
python3 runPrediction.py --fire Yellow --date 0311 --post-process
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--fire` | Fire name (folder under `training_data`) |
| `--date` | Date in MMDD format |
| `--points` | Points to sample (default: 10000) |
| `--list` | List fires and dates, then exit |
| `--eval` | Eval mode (default): metrics + viz |
| `--no-eval` | Inference only |
| `--model` | Model name without .h5 (default: 20260101-173820mod) |
| `--post-process` | Apply perimeter filter before metrics/viz |

---

### 6. Production batch predictions (`runProduction.py`)

Runs inference on **all** fires in `training_data` (or a single fire/date). Does **not** require ground truth. Writes CSVs, PNGs, GeoJSON perimeters, and logs under **`output/`**.

```bash
# Run on all fires (default when no args, or with --all)
python3 runProduction.py
python3 runProduction.py --all

# Single fire and date
python3 runProduction.py --fire Yellow --date 0311

# Custom points and model
python3 runProduction.py --fire Yellow --date 0311 --points 5000 --model 20260101-173820mod
```

**Outputs (examples):**

- `output/runProduction_<timestamp>.log`, `output/runProduction_<timestamp>_errors.log`
- `output/predictions_<fire>_<date>.csv`, `output/predictions_<fire>_<date>.png`
- `output/predicted_perimeter_<fire>_<date>.geojson`, `output/predicted_perimeter_<fire>_<date>_overlay.png`
- `output/images/perimeter_viz_*.png`, `output/images/perimeter_overlay_*.png`
- `output/all_predictions.geojson` (when applicable)

**Flags:**

| Flag | Description |
|------|-------------|
| `--fire` | Fire name (for single fire/date run) |
| `--date` | MMDD (for single fire/date run) |
| `--all` | Run on all fires in `training_data` |
| `--points` | Points per run (default: 10000) |
| `--model` | Model name without .h5 (default: 20260101-173820mod) |

---

## Output summary

| Script | Main outputs |
|--------|----------------|
| **create_csv_prediction.py** | `output/csv/<fire>.csv`, `output/csv_images/<fire>.png` |
| **runPrediction.py** | `output/figures/<fire>_<date>_radius50_points<N>.png`, metrics to console |
| **runProduction.py** | `output/predictions_*.csv`, `output/predictions_*.png`, `output/predicted_perimeter_*.geojson`, `output/predicted_perimeter_*_overlay.png`, `output/runProduction_*.log` |

---

## Model and data notes

- **Inputs:** Terrain (DEM, slope, aspect, NDVI, Landsat bands), weather (e.g. with containment), optional hotspot layer; 61×61 patches (AOI radius 30 px).
- **Output:** Burn probability per vulnerable pixel; perimeter products are derived from thresholding and contouring.
- **Data layout:** Each fire folder under `training_data/` or `0101_training_data/` should contain `perims/<MMDD>.npy`, `weather/<MMDD>.csv`, terrain `.npy` files, and (for training) next-day perimeter for labels.
- **Fires &lt;100 acres** are excluded when fetching via `getData.py` and in recent-fire processing in `create_csv_prediction.py`.
