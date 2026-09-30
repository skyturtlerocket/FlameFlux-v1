"""Run FlameFlux v2 on one fetched fire: 24 h growth, hourly arrival time, and perimeter exports.

    python runModel.py --fire Dome          # reads cache/Dome/, writes output/Dome/

Outputs per run (stamped with the issue hour):
    <stamp>.geojson   observed perimeter, 24 h perimeter and growth, hourly perimeters, open gate points
    <stamp>.npz       arrival-time raster (hours; NaN = not reached) and 1 h / 5 h perimeter stacks
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import rasterio.features
import rasterio.transform
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, mapping, shape
from shapely.ops import unary_union
from skimage.measure import find_contours
from xgboost import XGBClassifier, XGBRegressor

here = os.path.dirname(os.path.abspath(__file__))
cacheDir = os.path.join(here, "cache")
outputDir = os.path.join(here, "output")
modelDir = os.path.join(here, "model")
pixelAcres = 30.0 * 30.0 / 4046.8564224
eight = np.ones((3, 3), dtype=np.uint8)


# ---------------------------------------------------------------- perimeter history
# Successive perimeters jitter, so each is shifted to best cover the cumulative
# burned core. If no shift keeps half the core burned, two fires share the name.
historyDates = 4  # t0 plus three prior observations
maxShift, maxTotalShift = 40, 80
minCoverage, minGain = 0.5, 0.005
minSearchArea, maxAreaRatio = 1500, 8


def shift(mask, dy, dx):
    """Translate a mask by whole pixels, zero-filling the vacated edge."""
    out = np.roll(np.roll(np.asarray(mask), dy, axis=0), dx, axis=1)
    if dy > 0:
        out[:dy, :] = 0
    elif dy < 0:
        out[dy:, :] = 0
    if dx > 0:
        out[:, :dx] = 0
    elif dx < 0:
        out[:, dx:] = 0
    return out


def coverage(ref, cur):
    area = int(ref.sum())
    return 1.0 if area == 0 else float((ref & cur).sum()) / area


def bestShift(ref, cur):
    """Coarse (4 px) then fine (1 px) search for the shift of cur that best covers ref."""
    base = coverage(ref, cur)
    refArea, curArea = int(ref.sum()), int(cur.sum())
    if refArea < minSearchArea or curArea == 0 or curArea > maxAreaRatio * refArea:
        return (0, 0), base, base
    best, bestCov = (0, 0), base
    for dy in range(-maxShift, maxShift + 1, 4):
        for dx in range(-maxShift, maxShift + 1, 4):
            cov = coverage(ref, shift(cur, dy, dx))
            if cov > bestCov:
                best, bestCov = (dy, dx), cov
    cy, cx = best
    for dy in range(cy - 6, cy + 7):
        for dx in range(cx - 6, cx + 7):
            if abs(dy) <= maxShift and abs(dx) <= maxShift:
                cov = coverage(ref, shift(cur, dy, dx))
                if cov > bestCov:
                    best, bestCov = (dy, dx), cov
    return best, base, bestCov


def align(masks):
    """Chain-align masks. Returns (aligned masks, final shift), or None if contaminated."""
    ref = masks[0]
    aligned, dy, dx = [ref], 0, 0
    for raw in masks[1:]:
        cur = shift(raw, dy, dx) if (dy or dx) else raw
        (ddy, ddx), base, best = bestShift(ref, cur)
        if best < minCoverage:
            return None
        if best - base < minGain or abs(dy + ddy) > maxTotalShift or abs(dx + ddx) > maxTotalShift:
            ddy, ddx = 0, 0
        dy, dx = dy + ddy, dx + ddx
        current = shift(raw, dy, dx) if (dy or dx) else raw
        aligned.append(current)
        ref = ref | current
    return aligned, (dy, dx)


def loadHistory(fireDir, today):
    folder = os.path.join(fireDir, "perims")
    dates = sorted(pd.to_datetime(n[:-4], format="%Y%m%d") for n in os.listdir(folder) if n.endswith(".npy"))
    return [np.load(os.path.join(folder, d.strftime("%Y%m%d") + ".npy")).astype(bool) for d in dates if d <= today]


def arrivalIndex(history):
    """Observation index each pixel first burned (-1 unburned), t0 index, degraded flag.

    Degraded means a single observation or a contaminated sequence."""
    recent = history[-historyDates:]
    result = align(recent) if len(recent) >= 2 else None
    if result is None:
        return np.where(recent[-1], 0, -1).astype(np.int16), 0, True
    aligned, (dy, dx) = result
    index = np.full(aligned[0].shape, -1, np.int16)
    burned = np.zeros(aligned[0].shape, bool)
    for step, mask in enumerate(aligned):
        burned |= mask
        index[burned & (index < 0)] = step
    # Keep the full known scar so growth is never forecast into old burn.
    scar = np.logical_or.reduce(history)
    index[(shift(scar, dy, dx) if (dy or dx) else scar) & (index < 0)] = 0
    return index, len(aligned) - 1, False


# ---------------------------------------------------------------- sample
windowMargin, windowRound, windowMax = 160, 32, 4096
weatherColumns = ["air_temp_c", "relative_humidity_pct", "precip_mm", "wind_speed_kmh",
                  "cloud_cover_pct", "wind_u", "wind_v", "containment_pct"]


def forecastWindow(t0):
    """Square crop around t0 plus the forecast margin: (y0, y1, x0, x1)."""
    ys, xs = np.nonzero(t0)
    cy = int(round((int(ys.min()) + int(ys.max())) / 2))
    cx = int(round((int(xs.min()) + int(xs.max())) / 2))
    extent = max(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)) + 2 * windowMargin
    size = min(int(np.ceil(extent / windowRound) * windowRound), windowMax, *t0.shape)
    y0 = max(0, min(cy - size // 2, t0.shape[0] - size))
    x0 = max(0, min(cx - size // 2, t0.shape[1] - size))
    return y0, y0 + size, x0, x0 + size


def loadWeather(fireDir, issued, containment):
    """24 x 8 hourly weather from issue time, plus the mean wind vector."""
    frame = pd.read_csv(os.path.join(fireDir, "weather.csv"), parse_dates=["datetime"])
    frame = frame[(frame["datetime"] >= issued) & (frame["datetime"] < issued + pd.Timedelta(hours=24))].copy()
    if len(frame) < 24:
        raise ValueError(f"weather has {len(frame)} of 24 forecast hours")
    # Direction is where wind comes from; u/v point where it goes.
    theta = np.deg2rad(frame["wind_direction_deg"].to_numpy(np.float64))
    speed = frame["wind_speed_kmh"].to_numpy(np.float64)
    frame["wind_u"], frame["wind_v"] = -speed * np.sin(theta), -speed * np.cos(theta)
    frame["containment_pct"] = containment
    return frame[weatherColumns].to_numpy(np.float32), float(frame["wind_u"].mean()), float(frame["wind_v"].mean())


def buildSample(fireDir, index, t0Index, weather):
    """Model channels cropped to the forecast window."""
    y0, y1, x0, x1 = forecastWindow(index >= 0)
    crop = lambda a: a[y0:y1, x0:x1]
    load = lambda name: crop(np.load(os.path.join(fireDir, name + ".npy")).astype(np.float32))
    burn = crop(index)
    burned = burn >= 0
    grew = lambda lag: burned & (burn > 0) & (burn >= t0Index - lag + 1)
    since = np.zeros(burn.shape, np.float32)
    since[burned] = (t0Index - burn[burned]).astype(np.float32)
    fuelPath = os.path.join(fireDir, "fuel.npy")
    aspect = np.deg2rad(load("aspect"))
    sequence, windU, windV = weather
    return {
        "window": (y0, y1, x0, x1), "t0": burned,
        "grew1": grew(1), "grew2": grew(2), "grew3": grew(3),
        "timeSinceBurned": np.log1p(since),
        "historyDepth": np.float32(min(t0Index, 3) / 3.0),
        "dem": load("dem"), "slope": load("slope"),
        "aspectSin": np.sin(aspect).astype(np.float32), "aspectCos": np.cos(aspect).astype(np.float32),
        "ndvi": load("ndvi"), "band4": load("band4"),
        # Roads are not fetched; training used the grid size as "no road nearby".
        "roadDist": np.full(burn.shape, np.log1p(np.float32(max(index.shape))), np.float32),
        "fuel": crop(np.load(fuelPath)) if os.path.exists(fuelPath) else None,
        "windU": np.float32(windU), "windV": np.float32(windV),
        "containment": np.float32(sequence[0, 7]), "weather": sequence,
    }


# ---------------------------------------------------------------- front features
behindBands = ((0.0, 4.0), (4.0, 12.0), (12.0, 32.0))
aheadBands = ((0.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 32.0),
              (32.0, 64.0), (64.0, 128.0), (128.0, 256.0))
# Scott & Burgan FBFM40 benchmark spread rates (chains/hour); other codes are unburnable.
baseRos = {
    101: 15, 102: 35, 103: 45, 104: 75, 105: 85, 106: 100, 107: 110, 108: 120, 109: 170,
    121: 15, 122: 25, 123: 35, 124: 50,
    141: 10, 142: 15, 143: 25, 144: 35, 145: 50, 146: 35, 147: 50, 148: 60, 149: 80,
    161: 5, 162: 10, 163: 15, 164: 20, 165: 25,
    181: 1, 182: 2, 183: 3, 184: 4, 185: 5, 186: 6, 187: 7, 188: 8, 189: 10,
    201: 10, 202: 20, 203: 30, 204: 40,
}


def fuelSpeed(fuel):
    """Sustained spread rate (m/min): one third of the benchmark rate."""
    speed = np.zeros(fuel.shape, dtype=np.float32)
    for code, ros in baseRos.items():
        speed[fuel == code] = ros * (20.1168 / 60.0) * (1.0 / 3.0)
    return speed


def windAlignment(t0, windU, windV):
    """Cosine between each pixel's outward direction from the fire and the wind."""
    _, (iy, ix) = distance_transform_edt(~t0, return_indices=True)
    ys, xs = np.indices(t0.shape)
    dy, dx = (ys - iy).astype(np.float32), (xs - ix).astype(np.float32)
    dist = np.sqrt(dy * dy + dx * dx)
    dist[dist == 0] = 1.0
    dy, dx = dy / dist, dx / dist
    windU, windV = np.full(t0.shape, windU, np.float32), np.full(t0.shape, windV, np.float32)
    speed = np.sqrt(windU * windU + windV * windV)
    speed = np.where(speed > 1e-6, speed, 1.0)
    return np.where(t0, 0.0, dx * (windU / speed) - dy * (windV / speed)).astype(np.float32)


def resample(contour, spacing):
    closed = np.vstack([contour, contour[0]])
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    length = float(cumulative[-1])
    targets = np.linspace(0.0, length, max(3, int(np.ceil(length / max(spacing, 1.0)))), endpoint=False)
    return np.column_stack([np.interp(targets, cumulative, closed[:, 0]),
                            np.interp(targets, cumulative, closed[:, 1])]).astype(np.float32)


def frontGeometry(t0, spacing):
    """Evenly spaced points on the outer boundary of every fire component."""
    labels, count = label(t0, structure=eight)
    parts = {k: [] for k in ("points", "normals", "tangents", "curvature", "component", "prev", "next", "area")}
    offset = 0
    for component in range(1, count + 1):
        mask = labels == component
        contours = find_contours(mask.astype(np.float32), 0.5)
        if not contours or len(max(contours, key=len)) < 3:
            continue
        points = resample(max(contours, key=len), spacing)
        tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
        normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
        probe = np.rint(points + 2.0 * normal).astype(int)  # flip normals that point inward
        normal[mask[np.clip(probe[:, 0], 0, t0.shape[0] - 1), np.clip(probe[:, 1], 0, t0.shape[1] - 1)]] *= -1.0
        n = len(points)
        parts["points"].append(points)
        parts["normals"].append(normal.astype(np.float32))
        parts["tangents"].append(tangent.astype(np.float32))
        parts["curvature"].append(np.linalg.norm(np.roll(tangent, -1, axis=0) - np.roll(tangent, 1, axis=0),
                                                 axis=1).astype(np.float32) * 0.5)
        parts["component"].append(np.full(n, component - 1, dtype=np.int32))
        parts["prev"].append(offset + np.roll(np.arange(n), 1))
        parts["next"].append(offset + np.roll(np.arange(n), -1))
        parts["area"].append(np.full(n, mask.sum(), dtype=np.float32))
        offset += n
    if not offset:
        raise ValueError("perimeter has no usable boundary")
    return {k: np.concatenate(v) for k, v in parts.items()}


def segmentMean(values, owner, mask, n):
    index = owner[mask]
    if not len(index):
        return np.zeros(n, dtype=np.float32)
    weights = np.broadcast_to(np.asarray(values, np.float64), mask.shape)[mask]
    return (np.bincount(index, weights=weights, minlength=n)
            / np.maximum(np.bincount(index, minlength=n), 1)).astype(np.float32)


def segmentMinDistance(mask, owner, distance, n):
    result = np.full(n, np.inf, dtype=np.float32)
    if mask.any():
        np.minimum.at(result, owner[mask], distance[mask])
    result[~np.isfinite(result)] = 512.0
    return result


def bandFraction(mask, owner, lo, hi, spacing, n):
    return (np.bincount(owner[mask], minlength=n) / max((hi - lo) * spacing, 1.0)).clip(0, 1).astype(np.float32)


def extractFront(s, spacing):
    """Perimeter segments, each pixel's nearest segment, and 115 base features per segment."""
    t0 = s["t0"]
    geo = frontGeometry(t0, spacing)
    n = len(geo["points"])
    ys, xs = np.indices(t0.shape)
    distance, owner = cKDTree(geo["points"]).query(np.column_stack([ys.ravel(), xs.ravel()]), workers=-1)
    owner = owner.reshape(t0.shape).astype(np.int32)
    distance = distance.reshape(t0.shape).astype(np.float32)

    zeros = np.zeros(t0.shape, np.float32)
    speed = fuelSpeed(s["fuel"]) if s["fuel"] is not None else zeros
    burnable = (speed > 0).astype(np.float32) if s["fuel"] is not None else zeros
    behind = [s["grew1"], s["grew2"], s["grew3"], s["timeSinceBurned"]]
    ahead = [speed, burnable, s["roadDist"], s["containment"], s["dem"], s["slope"], s["aspectSin"],
             s["aspectCos"], s["ndvi"], s["band4"], windAlignment(t0, s["windU"], s["windV"])]
    acres = lambda m: np.full(n, np.log1p(m.sum() * pixelAcres), dtype=np.float32)
    features = [
        geo["normals"][:, 0], geo["normals"][:, 1], geo["tangents"][:, 0], geo["tangents"][:, 1],
        geo["curvature"], np.log1p(geo["area"]),
        acres(t0), acres(s["grew1"]), acres(s["grew2"]), acres(s["grew3"]),
        np.full(n, float(s["historyDepth"] > 0), dtype=np.float32),
        np.full(n, geo["component"].max() + 1, dtype=np.float32),
        np.full(n, float(s["fuel"] is not None), dtype=np.float32),
    ]
    for lo, hi in behindBands:  # inside the perimeter: where it recently grew
        band = t0 & (distance >= lo) & (distance < hi)
        features.append(bandFraction(band, owner, lo, hi, spacing, n))
        features += [segmentMean(c, owner, band, n) for c in behind]
    for grew in (s["grew1"], s["grew2"], s["grew3"]):
        features.append(segmentMinDistance(t0 & grew, owner, distance, n))
    for lo, hi in aheadBands:  # outside the perimeter: what lies in the growth path
        band = ~t0 & (distance >= lo) & (distance < hi)
        features.append(bandFraction(band, owner, lo, hi, spacing, n))
        features += [segmentMean(c, owner, band, n) for c in ahead]
    return {"features": np.stack(features, axis=1).astype(np.float32), "owner": owner, "distance": distance,
            "t0": t0, "points": geo["points"], "normals": geo["normals"],
            "prev": geo["prev"].astype(np.int64), "next": geo["next"].astype(np.int64)}


def componentSize(mask):
    components, count = label(mask, structure=eight)
    if not count:
        return np.zeros(mask.shape, dtype=np.float32)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return sizes[components].astype(np.float32)


def historyFeatures(s, front):
    """24 recent-growth features per segment that separate real advance from thin alignment slivers."""
    n = len(front["points"])
    owner = front["owner"].astype(np.int64)
    distance = front["distance"].astype(np.float16).astype(np.float32)  # trained at half precision
    near = front["t0"] & (distance < 32)
    count = lambda m: np.bincount(owner[m], weights=np.ones(m.sum(), np.float32), minlength=n).astype(np.float32)
    exposure = np.maximum(np.bincount(owner[near], minlength=n).astype(np.float32), 1.0)
    columns, support = [], []
    for grew in (s["grew1"], s["grew2"], s["grew3"]):
        raw = grew & near
        size = componentSize(raw)
        kept = raw & (size >= 6)
        rawCount, keptCount = count(raw), count(kept)
        core1, core2 = count(binary_erosion(kept, iterations=1)), count(binary_erosion(kept, iterations=2))
        largest = np.zeros(n, dtype=np.float32)
        if kept.any():
            np.maximum.at(largest, owner[kept], size[kept])
        columns += [rawCount / exposure, keptCount / exposure, core1 / exposure, core2 / exposure,
                    np.log1p(keptCount), np.log1p(largest), 1.0 - core1 / np.maximum(keptCount, 1.0)]
        support.append(keptCount > 0)
    columns += [np.stack(support).sum(axis=0).astype(np.float32),
                np.full(n, float(s["historyDepth"]), dtype=np.float32),
                support[0].astype(np.float32) + (support[1] & support[2]).astype(np.float32)]
    return np.stack(columns, axis=1).astype(np.float32)


# ---------------------------------------------------------------- segment gate
def loadHotspots(fireDir):
    path = os.path.join(fireDir, "firms.csv")
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path)
    if not len(frame):
        return None
    clock = frame["acq_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    frame["time"] = pd.to_datetime(frame["acq_date"].astype(str) + " " + clock.str[:2] + ":" + clock.str[2:],
                                   format="%Y-%m-%d %H:%M", utc=True, errors="coerce")
    return frame.dropna(subset=["time", "latitude", "longitude"])


def nearestSquared(a, b):
    """For each point in a, squared distance to the closest point in b."""
    out = np.empty(len(a), dtype=np.result_type(a, b))
    for start in range(0, len(a), 4096):
        chunk = a[start:start + 4096]
        out[start:start + 4096] = ((chunk[:, None, 0] - b[None, :, 0]) ** 2
                                   + (chunk[:, None, 1] - b[None, :, 1]) ** 2).min(axis=1)
    return out


def hotspotPixels(detections, grid, window, issued, hours):
    times = detections["time"].dt.tz_convert("UTC").dt.tz_localize(None)
    recent = detections[(times <= issued) & (times >= issued - pd.Timedelta(hours=hours))]
    if not len(recent):
        return np.zeros((0, 2), np.float32)
    rows, cols = rasterio.transform.rowcol(gridTransform(grid), recent["longitude"].to_numpy(float),
                                           recent["latitude"].to_numpy(float))
    return np.column_stack([np.asarray(rows, float) - window[0],
                            np.asarray(cols, float) - window[2]]).astype(np.float32)


def hotspotGate(front, detections, grid, window, issued, config):
    """(open mask or None, reach floor). None: too few hotspots to decide.

    Uses hotspots near the front from the last 8 h (24 h if 8 h is empty). Segments
    near a hotspot open; hotspots outside the perimeter set a minimum reach."""
    points = front["points"]
    floor = np.zeros(len(points), np.float32)
    if detections is None:
        return None, floor
    radius = config["hotspotRadius"]
    for hours in (config["hotspotHours"], config["hotspotFallbackHours"]):
        pixels = hotspotPixels(detections, grid, window, issued, hours)
        if len(pixels):
            pixels = pixels[nearestSquared(pixels, points) <= radius ** 2]
        if len(pixels):
            break
    if len(pixels) < config["hotspotMinDetections"]:
        return None, floor

    anchors = []
    for pixel in pixels:
        row, col = np.rint(pixel).astype(int)
        if 0 <= row < front["t0"].shape[0] and 0 <= col < front["t0"].shape[1] and not front["t0"][row, col]:
            squared = np.sum((points - pixel) ** 2, axis=1)
            segment = int(np.argmin(squared))
            if np.sqrt(squared[segment]) <= radius:
                anchors.append((segment, float(np.sqrt(squared[segment]))))
    if len(anchors) >= config["anchorMinDetections"]:
        for segment, distance in anchors:
            floor[segment] = max(floor[segment], distance)
    opened = np.sqrt(nearestSquared(points.astype(float), pixels)) <= radius
    return (opened if opened.any() else None), floor


def recentGrowthGate(front, s):
    """Open segments whose flank burned within the last two observations."""
    recent = s["grew2"] & front["t0"] & (front["distance"] < 8.0)
    opened = np.zeros(len(front["points"]), dtype=bool)
    opened[np.unique(front["owner"][recent])] = True
    return opened if opened.any() else np.ones(len(opened), bool)  # nothing grew: let the model decide


# ---------------------------------------------------------------- v2 model
def loadModels(folder=modelDir):
    with open(os.path.join(folder, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(folder, "norm.json")) as f:
        norm = {k: np.asarray(v, np.float32) if k.endswith(("Mean", "Std")) else v for k, v in json.load(f).items()}
    reach, guard = XGBRegressor(), XGBClassifier()
    reach.load_model(os.path.join(folder, "reach.ubj"))
    guard.load_model(os.path.join(folder, "dayGuard.ubj"))
    return {"config": config, "norm": norm, "reach": reach, "dayGuard": guard}


def segmentWeather(weather, normals, norm):
    """Per segment and hour: normalized weather plus wind along/across the outward normal."""
    normalized = (weather - norm["weatherMean"]) / np.maximum(norm["weatherStd"], 1e-6)
    east, north = normals[:, 1, None], -normals[:, 0, None]
    u, v = weather[None, :, 5], weather[None, :, 6]
    along, cross = east * u + north * v, -north * u + east * v
    dry = np.clip((100.0 - weather[None, :, 1]) / 70.0, 0.0, 1.5)
    directional = np.stack([along / 10.0, cross / 10.0, np.maximum(along, 0.0) / 10.0,
                            np.maximum(-along, 0.0) / 10.0, np.maximum(along, 0.0) / 10.0 * dry], axis=-1)
    repeated = np.broadcast_to(normalized[None], (len(normals), *normalized.shape))
    return np.concatenate([repeated, directional], axis=-1).astype(np.float32)


def neighbor(links, steps):
    out = np.arange(len(links), dtype=np.int64)
    for _ in range(steps):
        out = links[out]
    return out


def modelInputs(front, s, gate, norm):
    """497 features per segment: base, history, weather, and gate/neighbor context."""
    base = (front["features"] - norm["featureMean"]) / norm["featureStd"]
    # 16 zero columns: lagged hotspot inputs that the live model treats as unavailable.
    context = np.concatenate([np.zeros((len(base), 16), np.float32), historyFeatures(s, front)], axis=1)
    context = (context - norm["contextMean"]) / norm["contextStd"]
    forcing = segmentWeather(s["weather"], front["normals"], norm)
    forcing = np.concatenate((forcing.mean(1), forcing.min(1), forcing.max(1)), axis=1).astype(np.float32)
    timestamp = np.repeat(np.asarray([[1, 1, 0, 1, 0, 0]], np.float32), len(base), axis=0)  # exact issue time
    x = np.concatenate((base, context, forcing, timestamp), axis=1).astype(np.float32)

    active = (gate > 0.5).astype(np.float32)
    columns = [active[:, None], np.full((len(active), 1), active.mean(), np.float32),
               np.full((len(active), 1), np.log1p(active.sum()), np.float32),
               np.full((len(active), 1), np.log1p(len(active)), np.float32)]
    for hop in (1, 2, 4, 8, 16):
        left, right = neighbor(front["prev"], hop), neighbor(front["next"], hop)
        columns.append(((active + active[left] + active[right]) / 3.0)[:, None])
    selected = norm["neighborColumns"]
    for hop in (1, 4, 12):
        left, right = neighbor(front["prev"], hop), neighbor(front["next"], hop)
        columns.append((x[left][:, selected] + x[:, selected] + x[right][:, selected]) / 3.0)
    return np.nan_to_num(np.concatenate([x, *columns], axis=1), nan=0.0, posinf=1e5, neginf=-1e5).astype(np.float32)


def daySummary(x, gate):
    """Fire-day summary of the gated segments for the day guard."""
    active = gate > 0.15
    rows = x[active] if active.any() else x
    return np.concatenate([np.nanmean(rows, axis=0), np.nanstd(rows, axis=0),
                           np.nanmin(rows, axis=0), np.nanmax(rows, axis=0),
                           np.asarray([np.log1p(len(gate)), np.log1p(active.sum()), active.mean(), 1.0],
                                      np.float32)]).astype(np.float32)[None]


def predict(models, s, front, detections, grid, issued, degraded, firstObservation):
    """Per-segment 24 h reach and the final gate.

    A fire with observed progression always forecasts. A first observation
    forecasts only when hotspots support it; a contaminated history never does."""
    config = models["config"]
    opened, floor = hotspotGate(front, detections, grid, s["window"], issued, config)
    source = "hotspot"
    coldStart = bool(firstObservation and opened is not None)
    if opened is None:
        opened, source = recentGrowthGate(front, s), "recentGrowth"
    eligible = (not degraded) or coldStart
    gate = opened.astype(np.float32) if eligible else np.zeros(len(opened), np.float32)

    x = modelInputs(front, s, gate, models["norm"])
    dayProbability = float(models["dayGuard"].predict_proba(daySummary(x, gate))[0, 1])
    reach = np.maximum(np.expm1(models["reach"].predict(x)), 0).astype(np.float32) * config["reachMultiplier"]
    modelGuard = eligible and dayProbability >= config["dayGuardThreshold"]
    anchored = eligible and bool(np.any(floor > 0))
    if anchored:
        # Hotspots outside the perimeter are a measured lower bound on reach.
        reach = np.maximum(reach, floor)
        if modelGuard or coldStart:
            gate[floor > 0] = 1.0
        else:
            gate = (floor > 0).astype(np.float32)  # guard closed: open only where hotspots are
    if not (eligible and (modelGuard or anchored or coldStart)):
        gate[:] = 0.0
    return reach, gate, {"gateSource": source if eligible else "none", "dayGrowthProbability": dayProbability,
                         "dayGuardOpen": bool(eligible and (modelGuard or anchored or coldStart)),
                         "coldStart": coldStart, "anchored": anchored}


# ---------------------------------------------------------------- arrival time
def connected(candidate, t0):
    """Keep only candidate pixels 8-connected to the current fire."""
    labels, _ = label(candidate | t0, structure=eight)
    sources = np.unique(labels[t0])
    return np.isin(labels, sources[sources > 0]) & candidate


def hourlyGrowth(front, reach, gate, threshold):
    """24 nested growth masks, assuming each segment spreads at a constant rate."""
    owner, distance, t0 = front["owner"].astype(np.int64), front["distance"], front["t0"]
    openPixels = (gate > threshold)[owner] & ~t0
    fractions = np.arange(1, 25, dtype=np.float32) / 24
    grown = np.zeros_like(t0)
    hourly = []
    for fraction in fractions:
        limit = (reach * fraction)[owner]
        grown = grown | connected(openPixels & (distance <= limit + (0.05 + 1e-3 * limit)), t0)
        hourly.append(grown.copy())
    return np.stack(hourly)


def arrivalTime(hourly, t0):
    """Hours until each pixel burns: 0 inside the fire, NaN if not reached in 24 h."""
    arrival = np.full(t0.shape, np.nan, dtype=np.float32)
    arrival[t0] = 0.0
    reached = hourly[-1]
    arrival[reached] = (np.argmax(hourly, axis=0) + 1)[reached].astype(np.float32)
    return arrival


def perimeters(arrival, interval):
    """Cumulative burned area every `interval` hours, always ending at 24 h."""
    hours = list(range(interval, 25, interval))
    if hours[-1] != 24:
        hours.append(24)
    return np.asarray(hours, np.int16), np.stack([np.isfinite(arrival) & (arrival <= h) for h in hours])


# ---------------------------------------------------------------- exports
def gridTransform(grid):
    return rasterio.transform.from_bounds(*grid["bounds"], grid["width"], grid["height"])


def toGeometry(mask, grid):
    """Grid mask to a (Multi)Polygon in EPSG:4326, simplified to half a pixel."""
    if not mask.any():
        return None
    polygons = [shape(g) for g, v in rasterio.features.shapes(
        mask.astype(np.uint8), mask=mask, transform=gridTransform(grid)) if v == 1]
    return unary_union(polygons).simplify(0.5 * 30 / 111_320, preserve_topology=True)


def exportGeojson(path, window, t0Full, hourly, front, gate, grid, props, threshold):
    y0, y1, x0, x1 = window
    place = lambda m: np.pad(m, ((y0, grid["height"] - y1), (x0, grid["width"] - x1)))
    growth = place(hourly[-1])
    distance = distance_transform_edt(~t0Full)
    props = {**props,
             "growthAcres": round(float(growth.sum()) * pixelAcres, 2),
             "growthPct": round(100 * float(growth.sum()) / max(int(t0Full.sum()), 1), 3),
             "maxReachPx": round(float(distance[growth].max()), 3) if growth.any() else 0.0}
    features = []

    def add(mask, layer, extra=None):
        geometry = toGeometry(mask, grid)
        if geometry is not None:
            features.append({"type": "Feature", "geometry": mapping(geometry),
                             "properties": {"layer": layer, **props, **(extra or {})}})

    issued = pd.Timestamp(props["issued"])
    add(t0Full, "observedPerimeter")
    add(t0Full | growth, "perimeter24h")
    add(growth, "growth24h")
    for hour, mask in enumerate(hourly, start=1):
        add(t0Full | place(mask), f"perimeterHour{hour:02d}",
            {"hour": hour, "validAt": (issued + pd.Timedelta(hours=hour)).isoformat()})
    points = front["points"][gate > threshold]
    if len(points):
        lon, lat = rasterio.transform.xy(gridTransform(grid), points[:, 0] + y0, points[:, 1] + x0, offset="center")
        features.append({"type": "Feature", "geometry": mapping(MultiPoint(list(zip(lon, lat)))),
                         "properties": {"layer": "openFront", **props,
                                        "openSegments": len(points), "totalSegments": len(gate)}})
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)


# ---------------------------------------------------------------- run one fire
def runFire(fireDir, models=None, outDir=outputDir):
    """Forecast one fetched fire. Returns the GeoJSON path, or None if skipped."""
    models = models or loadModels()
    config = models["config"]
    with open(os.path.join(fireDir, "fire.json")) as f:
        fire = json.load(f)
    with open(os.path.join(fireDir, "grid.json")) as f:
        grid = json.load(f)
    issued = pd.Timestamp(fire["issued"])
    today = pd.Timestamp(fire["captured"]).tz_localize(None).normalize()

    history = loadHistory(fireDir, today)
    index, t0Index, degraded = arrivalIndex(history)
    weather = loadWeather(fireDir, issued, fire["containment"])
    s = buildSample(fireDir, index, t0Index, weather)
    if not degraded:
        # Skip fires whose latest observed growth is under 0.10% of prior area.
        growthPct = 100.0 * s["grew1"].sum() / max(int((s["t0"] & ~s["grew1"]).sum()), 1)
        if growthPct < config["latestGrowthMinPct"]:
            print(f"  skipped: latest growth {growthPct:.4f}% < {config['latestGrowthMinPct']}%")
            return None

    front = extractFront(s, config["frontSpacing"])
    detections = loadHotspots(fireDir)
    reach, gate, info = predict(models, s, front, detections, grid, issued, degraded,
                                firstObservation=degraded and len(history) <= 1)
    hourly = hourlyGrowth(front, reach, gate, config["gateThreshold"])
    arrival = arrivalTime(hourly, s["t0"])
    hours1, perimeters1 = perimeters(arrival, 1)
    hours5, perimeters5 = perimeters(arrival, 5)

    folder = os.path.join(outDir, fire["fire"])
    os.makedirs(folder, exist_ok=True)
    stamp = issued.strftime("%Y%m%dT%H0000Z")
    np.savez_compressed(os.path.join(folder, stamp + ".npz"), arrivalHours=arrival, window=np.asarray(s["window"]),
                        hours1=hours1, perimeters1=perimeters1, hours5=hours5, perimeters5=perimeters5,
                        reach=reach, gate=gate, frontPoints=front["points"])
    props = {"fire": fire["name"], "irwin": fire["irwin"], "model": "v2",
             "issued": issued.isoformat() + "Z", "validTo": (issued + pd.Timedelta(hours=24)).isoformat() + "Z",
             "perimeterCaptured": fire["captured"], "historyDates": len(history), "degradedHistory": degraded,
             "containment": fire["containment"], "acres": fire["acres"], **info}
    path = os.path.join(folder, stamp + ".geojson")
    exportGeojson(path, s["window"], index >= 0, hourly, front, gate, grid, props, config["gateThreshold"])
    print(f"  {info['gateSource']} gate, day guard {'open' if info['dayGuardOpen'] else 'closed'} "
          f"(p={info['dayGrowthProbability']:.2f}), +{hourly[-1].sum() * pixelAcres:,.0f} ac -> {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fire", required=True, help="Fire folder name under the cache (spaces as underscores)")
    parser.add_argument("--cache", default=cacheDir)
    parser.add_argument("--out", default=outputDir)
    args = parser.parse_args()
    runFire(os.path.join(args.cache, args.fire.replace(" ", "_")), outDir=args.out)


if __name__ == "__main__":
    main()
