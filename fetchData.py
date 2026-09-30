"""Fetch everything FlameFlux v2 needs for active WFIGS fires into cache/<fire>/.

    python fetchData.py                  # every qualifying active fire
    python fetchData.py --fire "Dome"    # one fire (WFIGS incident name)

Needs EARTHENGINE_PROJECT (terrain, vegetation), LFPS_EMAIL (fuel) and,
optionally, NASA_FIRMS_MAP_KEY (hotspots; without it the model gates on perimeter history).
"""
import argparse
import io
import json
import os
import shutil
import time
import zipfile
from datetime import datetime, timezone
from math import ceil, cos, radians

import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.transform
import requests
from shapely.geometry import shape

here = os.path.dirname(os.path.abspath(__file__))
cacheDir = os.path.join(here, "cache")

# ---------------------------------------------------------------- grid
# Square EPSG:4326 grid at 30 m/px, sized to the fire plus forecast headroom.
pixelMeters = 30
metersPerDegree = 111_320
minSize, maxSize, roundTo = 1024, 4096, 64
gridMargin = 160
gridAllowance = 1.15  # extra room when a fire outgrows its grid


def extent(geometry):
    minx, miny, maxx, maxy = shape(geometry).bounds
    lon, lat = (minx + maxx) / 2, (miny + maxy) / 2
    height = (maxy - miny) * metersPerDegree / pixelMeters
    width = (maxx - minx) * metersPerDegree * cos(radians(lat)) / pixelMeters
    return height, width, lon, lat


def gridFromCenter(lon, lat, size):
    halfLat = size * pixelMeters / 2 / metersPerDegree
    halfLon = size * pixelMeters / 2 / (metersPerDegree * cos(radians(lat)))
    return {"bounds": [lon - halfLon, lat - halfLat, lon + halfLon, lat + halfLat],
            "center": [lon, lat], "height": size, "width": size}


def outgrown(grid, geometry):
    minx, miny, maxx, maxy = shape(geometry).bounds
    west, south, east, north = grid["bounds"]
    perLat = metersPerDegree / pixelMeters
    perLon = metersPerDegree * cos(radians(grid["center"][1])) / pixelMeters
    return any(v + gridMargin > 0 for v in (
        (maxy - north) * perLat, (south - miny) * perLat,
        (west - minx) * perLon, (maxx - east) * perLon))


def loadGrid(fireDir, geometry):
    """Saved grid, or a new one if the fire is new or outgrew it (which clears its cache)."""
    path = os.path.join(fireDir, "grid.json")
    height, width, lon, lat = extent(geometry)
    needed = int(ceil((max(height, width) + 2 * gridMargin) / roundTo) * roundTo)
    size = max(minSize, min(int(ceil(needed * gridAllowance / roundTo) * roundTo), maxSize))
    if os.path.exists(path):
        with open(path) as f:
            previous = json.load(f)
        if not outgrown(previous, geometry):
            return previous
        size = min(maxSize, max(size, previous["width"]))
        shutil.rmtree(fireDir)
    grid = gridFromCenter(lon, lat, size)
    os.makedirs(fireDir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(grid, f)
    return grid


def gridTransform(grid):
    return rasterio.transform.from_bounds(*grid["bounds"], grid["width"], grid["height"])


def rasterize(geometry, grid):
    mask = rasterio.features.rasterize(
        [(shape(geometry), 1)], out_shape=(grid["height"], grid["width"]),
        transform=gridTransform(grid), fill=0, dtype=np.uint8)
    return mask > 0


# ---------------------------------------------------------------- perimeters
wfigs = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
currentUrl = f"{wfigs}/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query"
archiveUrl = f"{wfigs}/WFIGS_Daily_Perimeters_Public/FeatureServer/0/query"
alaska = (51.2, 71.4, -173.0, -130.0)  # minLat, maxLat, minLon, maxLon


def utc(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def fetchCurrent():
    params = {"where": "1=1", "returnGeometry": "true", "f": "geojson",
              "outFields": "poly_IncidentName,poly_DateCurrent,poly_PolygonDateTime,"
                           "poly_Acres_AutoCalc,poly_IRWINID,poly_CreateDate,"
                           "attr_PercentContained"}
    for attempt in range(4):  # the layer is large and sometimes arrives truncated
        try:
            response = requests.get(currentUrl, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            if data.get("features"):
                return data
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError("WFIGS current perimeters unavailable")


def discover(onlyFire=None, maxAgeHours=24, maxPerimeterDays=30, minAcres=100):
    """Recently updated, large-enough, non-Alaskan WFIGS fires."""
    now = datetime.now(timezone.utc)
    fires = []
    for feature in fetchCurrent()["features"]:
        props = feature["properties"]
        name = props.get("poly_IncidentName")
        if not name or (onlyFire and name != onlyFire) or props.get("poly_DateCurrent") is None:
            continue
        updated = utc(props["poly_DateCurrent"])
        if (now - updated).total_seconds() / 3600 > maxAgeHours:
            continue
        # PolygonDateTime is the capture time; DateCurrent is only the edit time.
        captured = utc(props["poly_PolygonDateTime"]) if props.get("poly_PolygonDateTime") else updated
        if (now - captured).total_seconds() / 86400 > maxPerimeterDays:
            continue
        acres = props.get("poly_Acres_AutoCalc")
        if acres is None or acres < minAcres:
            continue
        minx, miny, maxx, maxy = shape(feature["geometry"]).bounds
        lon, lat = (minx + maxx) / 2, (miny + maxy) / 2
        if alaska[0] <= lat <= alaska[1] and alaska[2] <= lon <= alaska[3]:
            continue
        fires.append({
            "name": name, "fire": name.strip().replace(" ", "_"),
            "geometry": feature["geometry"], "irwin": props.get("poly_IRWINID") or "",
            "updated": updated.isoformat(), "captured": captured.isoformat(), "acres": acres,
            "start": (utc(props["poly_CreateDate"]) if props.get("poly_CreateDate") else updated).isoformat(),
            "containment": props.get("attr_PercentContained") or 0.0,
        })
    return fires


def fetchArchive(irwin):
    """Every archived perimeter for an incident, one per capture date."""
    params = {"where": f"poly_IRWINID='{irwin}'", "orderByFields": "poly_DateCurrent ASC",
              "outFields": "poly_DateCurrent,poly_PolygonDateTime", "f": "geojson"}
    response = requests.get(archiveUrl, params=params, timeout=60)
    response.raise_for_status()
    byDate = {}
    for feature in response.json().get("features", []):
        props = feature["properties"]
        stamp = props.get("poly_PolygonDateTime") or props.get("poly_DateCurrent")
        if feature.get("geometry") and stamp is not None:
            byDate[pd.to_datetime(stamp, unit="ms").normalize()] = feature["geometry"]
    return sorted(byDate.items())


def fetchHistory(fireDir, fire, grid):
    """Rasterize today's perimeter and any archived dates into perims/YYYYMMDD.npy."""
    folder = os.path.join(fireDir, "perims")
    os.makedirs(folder, exist_ok=True)
    path = lambda date: os.path.join(folder, date.strftime("%Y%m%d") + ".npy")
    today = pd.Timestamp(fire["captured"]).tz_localize(None).normalize()
    np.save(path(today), rasterize(fire["geometry"], grid))
    if fire["irwin"]:
        try:
            for date, geometry in fetchArchive(fire["irwin"]):
                if date < today and not os.path.exists(path(date)):
                    np.save(path(date), rasterize(geometry, grid))
        except requests.RequestException as error:
            print(f"  archive fetch failed ({error}); using cached history")


# ---------------------------------------------------------------- terrain + vegetation (Earth Engine)
eeByteCap = 45_000_000  # getDownloadURL rejects requests over 50 MB
eeBytesPerPixelBand = 5  # float32 data plus mask


def initEarthEngine():
    import ee
    project = os.environ.get("EARTHENGINE_PROJECT")
    ee.Initialize(project=project) if project else ee.Initialize()
    return ee


def downloadRegion(image, bounds, height, width):
    west, south, east, north = bounds
    region = {"type": "Polygon", "coordinates": [[[west, south], [east, south], [east, north],
                                                  [west, north], [west, south]]]}
    for attempt in range(4):
        # Pinning dimensions returns exactly the grid; scale would not.
        url = image.getDownloadURL({"dimensions": f"{width}x{height}", "crs": "EPSG:4326",
                                    "region": region, "format": "GEO_TIFF"})
        response = requests.get(url, timeout=600)
        if response.status_code not in (429, 500, 502, 503, 504) or attempt == 3:
            response.raise_for_status()
            break
        time.sleep(2 ** attempt)
    with rasterio.open(io.BytesIO(response.content)) as src:
        return src.read().astype(np.float32)


def downloadToGrid(image, grid, bands):
    """Download an EE image onto the grid, tiling to stay under the size cap."""
    height, width = grid["height"], grid["width"]
    west, south, east, north = grid["bounds"]
    image = image.toFloat()
    tiles = max(1, int(np.ceil(np.sqrt(bands * eeBytesPerPixelBand * height * width / eeByteCap))))
    out = np.empty((bands, height, width), dtype=np.float32)
    rowEdges = [int(round(i * height / tiles)) for i in range(tiles + 1)]
    colEdges = [int(round(i * width / tiles)) for i in range(tiles + 1)]
    for r in range(tiles):
        for c in range(tiles):
            r0, r1, c0, c1 = rowEdges[r], rowEdges[r + 1], colEdges[c], colEdges[c + 1]
            bounds = (west + (east - west) * c0 / width, north - (north - south) * r1 / height,
                      west + (east - west) * c1 / width, north - (north - south) * r0 / height)
            out[:, r0:r1, c0:c1] = downloadRegion(image, bounds, r1 - r0, c1 - c0)
    return out


def fetchTerrain(ee, fireDir, grid):
    """SRTM elevation, slope and aspect."""
    dem = ee.Image("USGS/SRTMGL1_003")
    for name, image in (("dem", dem), ("slope", ee.Terrain.slope(dem)), ("aspect", ee.Terrain.aspect(dem))):
        np.save(os.path.join(fireDir, f"{name}.npy"), downloadToGrid(image, grid, 1)[0])


def clearMask(image):
    """Landsat QA_PIXEL: dilated cloud, cirrus, cloud and shadow bits."""
    qa = image.select("QA_PIXEL")
    bad = qa.bitwiseAnd(1 << 1).neq(0)
    for bit in (2, 3, 4):
        bad = bad.Or(qa.bitwiseAnd(1 << bit).neq(0))
    return bad.Not()


def fetchVegetation(ee, fireDir, grid, fireStart):
    """Red band and NDVI from the most recent clear pre-fire Landsat scene.

    Ladder: a 98%-clear scene in the last 90 days, a 90%-clear one, a 98%-clear
    one in the last 18 months, else a median of the 3 newest >=70%-clear scenes."""
    start = pd.Timestamp(fireStart).tz_convert("UTC").tz_localize(None).normalize()
    west, south, east, north = grid["bounds"]
    region = ee.Geometry.Rectangle([west, south, east, north])
    day = lambda offset: ee.Date(str(start - offset)[:10])
    end = day(pd.Timedelta(days=1))

    def clearFraction(image):
        # Unmasked so area outside the scene counts as not clear.
        fraction = clearMask(image).unmask(0).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=90,
            maxPixels=1e9, bestEffort=True).get("QA_PIXEL")
        return image.set("clear", ee.Number(ee.Algorithms.If(fraction, fraction, 0)))

    scale = lambda image: image.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5"]).multiply(0.0000275).add(-0.2)
    scenes = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
              .filterBounds(region).filterDate(day(pd.DateOffset(months=18)), end)
              .map(clearFraction).sort("system:time_start", False))
    recent = scenes.filterDate(day(pd.Timedelta(days=90)), end)

    bands = None
    for pool, bar in ((recent, 0.98), (recent, 0.90), (scenes, 0.98)):
        hit = pool.filter(ee.Filter.gte("clear", bar))
        if hit.size().getInfo() > 0:
            bands = scale(ee.Image(hit.first()))
            break
    if bands is None:
        pool = recent.filter(ee.Filter.gte("clear", 0.70))
        if pool.size().getInfo() == 0:
            pool = scenes.filter(ee.Filter.gte("clear", 0.70))
        if pool.size().getInfo() == 0:
            raise RuntimeError("no usable pre-fire Landsat scene")
        bands = pool.limit(3).map(lambda im: scale(im).updateMask(clearMask(im))).median()
    bands = bands.rename(["B2", "B3", "B4", "B5"])
    image = bands.select(["B4"]).addBands(bands.normalizedDifference(["B5", "B4"]).rename("NDVI"))

    stack = downloadToGrid(image, grid, 2)
    for band in stack:  # fill gaps with the band median
        gap = ~np.isfinite(band)
        if gap.all():
            band[:] = 0.0
        elif gap.any():
            band[gap] = float(np.median(band[~gap]))
    np.save(os.path.join(fireDir, "band4.npy"), stack[0])
    np.save(os.path.join(fireDir, "ndvi.npy"), stack[1])


# ---------------------------------------------------------------- fuel (LANDFIRE)
lfps = "https://lfps.usgs.gov/api/job"


def fetchFuel(fireDir, grid):
    """LANDFIRE FBFM40 fuel model, nearest-neighbor warped onto the grid."""
    from osgeo import gdal
    gdal.UseExceptions()
    if not os.environ.get("LFPS_EMAIL"):
        raise RuntimeError("set LFPS_EMAIL (LANDFIRE requires a contact email)")
    west, south, east, north = grid["bounds"]
    job = requests.get(f"{lfps}/submit", timeout=60, params={
        "Email": os.environ["LFPS_EMAIL"], "Layer_List": "LF2023_FBFM40",
        "Area_of_Interest": f"{west} {south} {east} {north}", "Output_Projection": "5070"})
    job.raise_for_status()
    jobId = job.json()["jobId"]
    for _ in range(180):
        status = requests.get(f"{lfps}/status", params={"JobId": jobId}, timeout=60).json()
        if status.get("status") in ("Succeeded", "Failed", "Canceled"):
            break
        time.sleep(10)
    if status.get("status") != "Succeeded":
        raise RuntimeError(f"LANDFIRE job {jobId}: {status.get('status')}")
    archive = requests.get(status["outputFile"], timeout=300)
    archive.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(archive.content)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".tif"))
        tif = os.path.join(fireDir, "fuel.tif")
        with open(tif, "wb") as f:
            f.write(z.read(name))
    warped = gdal.Warp("", tif, format="MEM", dstSRS="EPSG:4326", outputBounds=(west, south, east, north),
                       width=grid["width"], height=grid["height"], resampleAlg="near")
    fuel = warped.GetRasterBand(1).ReadAsArray().astype(np.int16)
    os.remove(tif)
    np.save(os.path.join(fireDir, "fuel.npy"), np.where(fuel < 0, 0, fuel))


# ---------------------------------------------------------------- weather (open-meteo)
forecastUrl = "https://api.open-meteo.com/v1/forecast"


def fetchWeather(fireDir, grid, issued):
    """Hourly forecast at the grid center covering the 24 h after issue."""
    lon, lat = grid["center"]
    end = issued + pd.Timedelta(hours=24)
    response = requests.get(forecastUrl, timeout=60, params={
        "latitude": lat, "longitude": lon, "timezone": "UTC",
        "start_date": issued.strftime("%Y-%m-%d"), "end_date": end.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,"
                  "wind_speed_10m,wind_direction_10m,cloud_cover"})
    response.raise_for_status()
    hourly = response.json()["hourly"]
    pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "air_temp_c": hourly["temperature_2m"],
        "relative_humidity_pct": hourly["relative_humidity_2m"],
        "precip_mm": hourly["precipitation"],
        "wind_speed_kmh": hourly["wind_speed_10m"],
        "wind_direction_deg": hourly["wind_direction_10m"],
        "cloud_cover_pct": hourly["cloud_cover"],
    }).to_csv(os.path.join(fireDir, "weather.csv"), index=False)


# ---------------------------------------------------------------- hotspots (NASA FIRMS)
firmsUrl = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
firmsSources = ("VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "VIIRS_SNPP_NRT", "MODIS_NRT")


def fetchHotspots(fireDir, grid):
    """VIIRS and MODIS detections over the grid in the last 3 days."""
    key = os.environ.get("NASA_FIRMS_MAP_KEY", "").strip()
    path = os.path.join(fireDir, "firms.csv")
    if os.path.exists(path):
        os.remove(path)
    if not key:
        print("  NASA_FIRMS_MAP_KEY not set; the model will gate on perimeter history")
        return
    area = ",".join(f"{v:.6f}" for v in grid["bounds"])
    frames = []
    for source in firmsSources:
        try:
            response = requests.get(f"{firmsUrl}/{key}/{source}/{area}/3", timeout=45)
            response.raise_for_status()
            frame = pd.read_csv(io.StringIO(response.text))
            if len(frame):
                frames.append(frame)
        except Exception as error:
            print(f"  FIRMS {source} failed ({type(error).__name__})")
    if frames:
        frame = pd.concat(frames, ignore_index=True, sort=False)
        subset = [c for c in ("latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument")
                  if c in frame]
        frame.drop_duplicates(subset=subset).to_csv(path, index=False)


# ---------------------------------------------------------------- per fire
def fetchFire(fire, root=cacheDir, ee=None):
    """Populate cache/<fire>/ and return the fire directory."""
    fireDir = os.path.join(root, fire["fire"])
    grid = loadGrid(fireDir, fire["geometry"])
    fetchHistory(fireDir, fire, grid)
    has = lambda name: os.path.exists(os.path.join(fireDir, name))
    if not (has("dem.npy") and has("slope.npy") and has("aspect.npy")):
        ee = ee or initEarthEngine()
        fetchTerrain(ee, fireDir, grid)
    if not has("ndvi.npy"):
        ee = ee or initEarthEngine()
        fetchVegetation(ee, fireDir, grid, fire["start"])
    if not has("fuel.npy"):
        try:
            fetchFuel(fireDir, grid)
        except Exception as error:  # the model still runs, without fuel features
            print(f"  fuel fetch failed, forecasting without fuel: {error}")
    issued = pd.Timestamp.now(tz="UTC").tz_localize(None)
    fetchWeather(fireDir, grid, issued)
    fetchHotspots(fireDir, grid)
    info = {k: v for k, v in fire.items() if k != "geometry"}
    info["issued"] = issued.isoformat()
    with open(os.path.join(fireDir, "fire.json"), "w") as f:
        json.dump(info, f, indent=2)
    return fireDir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fire", help="WFIGS incident name (default: every active fire)")
    parser.add_argument("--cache", default=cacheDir)
    args = parser.parse_args()
    fires = discover(args.fire)
    print(f"{len(fires)} fire(s): {[f['name'] for f in fires]}")
    for fire in fires:
        print(f"fetching {fire['name']}")
        try:
            print(f"  -> {fetchFire(fire, args.cache)}")
        except Exception as error:
            print(f"  failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
