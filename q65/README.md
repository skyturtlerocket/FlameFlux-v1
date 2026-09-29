# q65

24-hour wildfire perimeter forecast for active US fires. q65 predicts how far each
segment of the current perimeter will advance in the next 24 h, then turns that into
hourly arrival times and perimeters.

## Setup

```bash
pip install -r requirements.txt
earthengine authenticate
export EARTHENGINE_PROJECT=<google cloud project with Earth Engine>
export LFPS_EMAIL=<your email>          # LANDFIRE requires a contact address
export NASA_FIRMS_MAP_KEY=<firms key>   # optional, see below
```

## Run

```bash
python pipeline.py                  # every active fire: fetch, forecast, arrival time, export
python pipeline.py --fire "Dome"    # one fire

python fetchData.py --fire "Dome"   # data only  -> cache/Dome/
python runModel.py --fire Dome      # model only -> output/Dome/
```

## Scripts

| file | does |
|---|---|
| `fetchData.py` | finds active WFIGS fires and caches the perimeter history, SRTM terrain, Landsat NDVI and red band, LANDFIRE fuel, open-meteo forecast and FIRMS hotspots |
| `runModel.py` | aligns the perimeter history, builds per-segment features, gates the front, predicts 24 h reach, computes arrival time, writes the exports |
| `pipeline.py` | runs both for every active fire |
| `model/` | `reach.ubj` (segment reach), `dayGuard.ubj` (fire-day growth classifier), `norm.json`, `config.json` |

Fires qualify when WFIGS updated them in the last 24 h, the perimeter is at most 30 days
old, they cover at least 100 acres and are outside Alaska. A fire whose latest observed
growth is under 0.10% of its prior area is skipped.

## Output

`output/<fire>/<issue hour>.geojson` (EPSG:4326), one feature per `layer`:

- `observedPerimeter`: the WFIGS perimeter the forecast starts from
- `perimeter24h`, `growth24h`: forecast perimeter and new growth after 24 h
- `perimeterHour01` … `perimeterHour24`: hourly perimeters
- `openFront`: perimeter points the gate allowed to grow

`output/<fire>/<issue hour>.npz`: `arrivalHours` (hours until each pixel burns; 0 inside
the fire, NaN if not reached), `perimeters1`/`perimeters5` (burned area every 1 h / 5 h),
and `window` (row/column crop of the fire grid).

## How it works

1. Perimeter history (up to 4 observations) is aligned to remove mapping jitter.
2. The perimeter is split into segments every 4 px (120 m). Each segment gets
   geometry, recent-growth, fuel, terrain, vegetation and wind features.
3. Gate: segments within 480 m of a FIRMS hotspot from the last 8 h (24 h if none) open.
   With fewer than 5 near-front hotspots, segments that grew in the last two
   observations open instead.
4. The day guard decides whether the fire grows at all; the reach model sets how far
   each open segment advances. Hotspots outside the perimeter set a minimum reach.
5. Arrival time assumes each segment spreads at a constant rate over the 24 h.

## Caveats

- A zero-growth forecast can be the day guard's call, not evidence the fire stopped.
- A fire's first observation forecasts only with hotspot support.
- Hourly timing is interpolated from a 24 h endpoint; it has no sub-daily validation.
- Spotting is not modeled. Road distance is not fetched.
