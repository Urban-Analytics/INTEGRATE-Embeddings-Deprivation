# Multi-city Street View data store

Standalone pipeline for sampling a city's road network and downloading Google
Street View imagery into an H5 store whose schema matches the existing
Manchester store (`notebooks/1-SampleStreetNetwork.ipynb`).

## Quick start

```bash
export GOOGLE_STREETVIEW_API_KEY="..."
jupyter lab sample_city.ipynb
```

Set `CITY_NAME` (and optionally `POLYGON_PATH`, `SPACING_M`, `MAX_POINTS`) in
the second code cell, then run top to bottom. The metadata pass is free; the
image-download cell defaults to `dry_run=True` so you see the cost estimate
before spending anything.

## Files

| File | Purpose |
|---|---|
| `paths.py` | Per-city path helpers and API-key lookup. |
| `city_pipeline.py` | Boundary, road graph, sampling, metadata, download, H5 build, merge. |
| `sample_city.ipynb` | End-to-end driver notebook. |
| `explore_city.ipynb` | Read-only: inspect a built store (stats, maps, spot-check vs real Street View). |
| `data/<slug>/` | Per-city state (created on first run). |

## Per-city directory layout

```
data/<city_slug>/
├── boundary.geojson         # cached city polygon
├── road_graph.graphml       # cached OSMnx drive graph
├── points.parquet           # master point table (append-only across rounds)
├── samples_manifest.parquet # one row per sampling round
├── pano_metadata.parquet    # results of the free metadata endpoint
├── attempted.parquet        # per-(point, heading) download log
├── street_images/           # on-disk JPEG cache (resume buffer)
└── street_data.h5           # final H5, schema below
```

## H5 schema

Manchester-compatible datasets — same names, same dtypes:

- `point_id` (int64), `latitude` (float64), `longitude` (float64)
- `date` (S10) — e.g. `b"2024-08"`
- `image_paths` (N×K, S512) — paths relative to the city root
- `images_present` (N×K, bool)
- `images_jpeg` (N×K, variable-length uint8) — raw JPEG bytes

Extra per-row datasets (cheap; preserve information from the SV API):

- `sampled_lat`, `sampled_lon` (float64) — point pre-snap
- `pano_lat`, `pano_lon` (float64) — point post-snap (where SV actually is)
- `pano_id` (S128), `copyright` (S128), `status` (S32)
- `city` (S64), `round_id` (int32), `heading` (N×K, int32)

H5 attributes: `city`, `city_slug`, `image_size`, `headings`, `built_at`.

## Adding to an existing city

Rerun cells 3+ with a smaller `SPACING_M` (denser) or a different boundary.
The pipeline:

1. Re-samples candidates at the new spacing.
2. Deduplicates against existing points (6 dp lat/lon).
3. Assigns fresh `point_id`s strictly greater than the existing max.
4. Appends a row to `samples_manifest.parquet` with the new `round_id`.
5. Only fetches metadata + images for the new points.
6. Rebuilds the H5 to include the new rows.

## Resuming after a crash

Just rerun the notebook. The metadata pass skips points already in
`pano_metadata.parquet`; the image download skips (point, heading) pairs
already logged successful in `attempted.parquet` (or already present on disk).
Worst case you waste a few seconds checking caches.

## Combining cities

```python
import city_pipeline as cp
from paths import CityPaths
cp.merge_city_h5s(
    [CityPaths.for_city(n).h5 for n in ["Leeds, UK", "Sheffield, UK"]],
    out_path="data/combined_street_data.h5",
)
```

Rows in the combined H5 carry their original `city` and `point_id`.

## Cost note

Google Street View Static API images cost ~$7 per 1000 (May 2026). At 50 m
spacing × 4 headings, a medium UK city is 50–100k images = $350–$700. Use
`SPACING_M=200` or `MAX_POINTS=...` for a first-pass test.

## Dependencies

Beyond what the parent project already uses, this folder pulls in
`tenacity` (HTTP retries). All others (`osmnx`, `geopandas`, `h5py`, `requests`,
`tqdm`, `pyarrow`) are in the project's existing environment.
