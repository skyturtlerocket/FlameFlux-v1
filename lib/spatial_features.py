"""
Spatial awareness features for wildfire prediction.
Computes 7 per-point features for spatial context.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, center_of_mass


def dominant_wind_direction(weather_matrix):
    """
    Compute dominant wind direction in degrees from weather matrix.
    Uses wind-weighted average of u,v components.
    Returns direction in degrees [0, 360) where wind is blowing FROM.
    """
    msl_pressure, temp, dewpt, temp2, wdir, wspeed, precip, hum, containment = weather_matrix
    #wind direction: meteorological convention, direction FROM which wind blows
    #convert to radians: 0 = North, 90 = East, 180 = South, 270 = West
    wdir_rad = np.radians(np.array(wdir, dtype=np.float64))
    #unit vector components (wind blows FROM direction, so u = -sin, v = -cos for "to" direction)
    #for "from" direction: u = sin(wdir), v = cos(wdir) gives direction wind comes from
    u = np.sin(wdir_rad) * np.array(wspeed, dtype=np.float64)
    v = np.cos(wdir_rad) * np.array(wspeed, dtype=np.float64)
    u_avg = np.mean(u)
    v_avg = np.mean(v)
    # atan2(u, v) gives angle from North; convert to degrees [0, 360)
    angle_rad = np.arctan2(u_avg, v_avg)
    angle_deg = np.degrees(angle_rad) % 360
    return float(angle_deg)


def _circular_mask(radius):
    """Create boolean circular mask of given radius (in pixels)."""
    r = int(radius)
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = x*x + y*y <= r*r
    return mask


def _ray_intersects_perimeter(perim, py, px, dy, dx, max_steps=500):
    """
    Step along unit direction (dy, dx) from (py, px) and check if we hit perimeter.
    Returns True if perimeter pixel hit, False otherwise.
    """
    h, w = perim.shape
    perim_bool = perim.astype(bool)
    step = 0.5  # sub-pixel stepping for accuracy
    for i in range(max_steps):
        t = i * step
        ty = int(round(py + dy * t))
        tx = int(round(px + dx * t))
        if ty < 0 or ty >= h or tx < 0 or tx >= w:
            return False
        if perim_bool[ty, tx]:
            return True
    return False


def _ray_distance_to_perimeter(perim, py, px, dy, dx, max_steps=500):
    """
    Step from (py, px) in direction (dy, dx) until hitting perimeter.
    Returns distance in pixels, or max_steps * step if no hit.
    """
    h, w = perim.shape
    perim_bool = perim.astype(bool)
    step = 0.5
    for i in range(max_steps):
        t = i * step
        ty = int(round(py + dy * t))
        tx = int(round(px + dx * t))
        if ty < 0 or ty >= h or tx < 0 or tx >= w:
            return float(i * step)  # went off edge
        if perim_bool[ty, tx]:
            return float(t)
    return float(max_steps * step)


def _compute_per_burn_date_cache(burn_name, date, day, dataset, get_hotspot_fn=None):
    """
    Compute quantities that depend only on (burn, date), not on point location.
    Cached and reused for all points in the same (burn, date).
    """
    sp = day.startingPerim
    dist_from_fire = distance_transform_edt(~sp.astype(bool))
    cy, cx = center_of_mass(sp)
    wm = dataset.data.getWeather(burn_name, date)
    wind_deg = dominant_wind_direction(wm)
    wind_rad = np.radians(wind_deg)
    downwind_dy = -np.cos(wind_rad)
    downwind_dx = np.sin(wind_rad)
    upwind_dy = np.cos(wind_rad)
    upwind_dx = -np.sin(wind_rad)
    if get_hotspot_fn:
        hs = get_hotspot_fn(burn_name, date)
    else:
        try:
            hs = day.loadHotspotData()
        except Exception:
            hs = np.zeros_like(sp, dtype=np.float32)
    prev_perim = getattr(day, 'previousPerim', None)
    aspect_layer = dataset.data.burns[burn_name].layers.get('aspect') if burn_name in dataset.data.burns else None
    return {
        'sp': sp,
        'dist_from_fire': dist_from_fire,
        'cy': cy, 'cx': cx,
        'wind_deg': wind_deg,
        'upwind_dy': upwind_dy, 'upwind_dx': upwind_dx,
        'downwind_dy': downwind_dy, 'downwind_dx': downwind_dx,
        'hs': hs,
        'prev_perim': prev_perim,
        'aspect_layer': aspect_layer,
        'h': sp.shape[0], 'w': sp.shape[1],
    }


def compute_spatial_features(burn_name, date, location, day, dataset,
                             get_hotspot_fn=None, get_previous_perim_fn=None, cache=None):
    """
    Compute 7 spatial awareness features for a single point.
    If cache is provided (from _compute_per_burn_date_cache), reuses per-(burn,date) computations.

    Returns 8-element numpy array (float32):
    [0] distance_to_perimeter (pixels)
    [1] bearing_from_fire_centroid (radians, [0, 2*pi])
    [2] local_perimeter_density (count in 10-px radius, normalized)
    [3] upwind_distance (pixels to perimeter in upwind direction)
    [4] downwind_fire_presence (1 if fire downwind, 0 else)
    [5] hotspot_density_local (mean in 10-px radius)
    [6] aspect_alignment (cos(terrain_aspect - wind_direction), [-1, 1])
    [7] prev_growth_indicator (1 if in previous day perimeter, 0 else) - appended for physics layer

    Note: Plan specifies 7 features for weather expansion; prev_growth is the 8th for physics.
    We return 8 elements: first 7 go to weather (16=9+7), indices 0,6,7 go to physics (distance, aspect_alignment, prev_growth).
    """
    y, x = location
    if cache is not None:
        sp = cache['sp']
        dist_from_fire = cache['dist_from_fire']
        cy, cx = cache['cy'], cache['cx']
        wind_deg = cache['wind_deg']
        upwind_dy, upwind_dx = cache['upwind_dy'], cache['upwind_dx']
        downwind_dy, downwind_dx = cache['downwind_dy'], cache['downwind_dx']
        hs = cache['hs']
        prev_perim = cache['prev_perim']
        aspect_layer = cache['aspect_layer']
        h, w = cache['h'], cache['w']
    else:
        sp = day.startingPerim
        h, w = sp.shape[:2]
        dist_from_fire = distance_transform_edt(~sp.astype(bool))
        cy, cx = center_of_mass(sp)
        wm = dataset.data.getWeather(burn_name, date)
        wind_deg = dominant_wind_direction(wm)
        upwind_dy = np.cos(np.radians(wind_deg))
        upwind_dx = -np.sin(np.radians(wind_deg))
        downwind_dy = -np.cos(np.radians(wind_deg))
        downwind_dx = np.sin(np.radians(wind_deg))
        if get_hotspot_fn:
            hs = get_hotspot_fn(burn_name, date)
        else:
            try:
                hs = day.loadHotspotData()
            except Exception:
                hs = np.zeros_like(sp, dtype=np.float32)
        prev_perim = getattr(day, 'previousPerim', None)
        aspect_layer = dataset.data.burns[burn_name].layers.get('aspect') if burn_name in dataset.data.burns else None

    # 1. distance_to_perimeter
    distance_to_perimeter = float(dist_from_fire[y, x])

    # 2. bearing_from_fire_centroid
    dy_pt = y - cy
    dx_pt = x - cx
    bearing = np.arctan2(dx_pt, -dy_pt)  # -dy so 0 = North
    if bearing < 0:
        bearing += 2 * np.pi
    bearing_from_fire_centroid = float(bearing)

    # 3. local_perimeter_density (count perimeter pixels in 10-px radius)
    r = 10
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    patch = sp[y0:y1, x0:x1].astype(np.float32)
    mask = _circular_mask(r)
    m_h, m_w = mask.shape
    #crop mask to match patch (patch can be smaller at edges)
    my0 = r - (y - y0)
    mx0 = r - (x - x0)
    m_h_use = min(patch.shape[0], m_h - my0)
    m_w_use = min(patch.shape[1], m_w - mx0)
    m_patch = mask[my0:my0 + m_h_use, mx0:mx0 + m_w_use]
    if m_patch.shape == patch.shape:
        count = float(np.sum(patch[m_patch]))
        total = float(np.sum(m_patch))
    else:
        count = float(np.sum(patch))
        total = float(patch.size)
    local_perimeter_density = count / max(1.0, total)

    # 4. upwind_distance
    upwind_distance = _ray_distance_to_perimeter(sp, y, x, upwind_dy, upwind_dx)

    # 5. downwind_fire_presence
    hit = _ray_intersects_perimeter(sp, y, x, downwind_dy, downwind_dx)
    downwind_fire_presence = 1.0 if hit else 0.0

    # 6. hotspot_density_local (hs from cache or computed above)
    hs_patch = hs[max(0,y-r):min(h,y+r+1), max(0,x-r):min(w,x+r+1)]
    valid = np.isfinite(hs_patch)
    hotspot_density_local = float(np.mean(hs_patch[valid])) if np.any(valid) else 0.0

    # 7. aspect_alignment: cos(terrain_aspect - wind_direction)
    if aspect_layer is not None and 0 <= y < h and 0 <= x < w:
        asp = aspect_layer[y, x]
        if np.isfinite(asp):
            diff_rad = np.radians(asp - wind_deg)
            aspect_alignment = float(np.clip(np.cos(diff_rad), -1.0, 1.0))
        else:
            aspect_alignment = 0.0
    else:
        aspect_alignment = 0.0

    # 8. prev_growth_indicator (prev_perim from cache or computed above)
    if get_previous_perim_fn and prev_perim is None:
        prev_perim = get_previous_perim_fn(burn_name, date, day)
    if prev_perim is not None and 0 <= y < prev_perim.shape[0] and 0 <= x < prev_perim.shape[1]:
        prev_growth_indicator = 1.0 if prev_perim[y, x] == 1 else 0.0
    else:
        prev_growth_indicator = 0.0

    return np.array([
        distance_to_perimeter,
        bearing_from_fire_centroid,
        local_perimeter_density,
        upwind_distance,
        downwind_fire_presence,
        hotspot_density_local,
        aspect_alignment,
        prev_growth_indicator,
    ], dtype=np.float32)
