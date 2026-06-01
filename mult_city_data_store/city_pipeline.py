"""Multi-city Street View pipeline.

Stages, each individually resumable and cached on disk:

1.  ``load_boundary``      – place name or polygon file -> shapely polygon.
2.  ``build_road_graph``   – OSMnx ``drive`` graph, cached as GraphML.
3.  ``sample_points_along_edges`` – deterministic spacing along each edge.
4.  ``register_sampling_round``   – append candidates to the master point
    table with stable, non-colliding ``point_id``s (dedupes new candidates
    against existing points at 6dp lat/lon).
5.  ``fetch_metadata_for_points`` – calls the free Street View metadata
    endpoint and caches results in ``pano_metadata.parquet``.
6.  ``download_images``    – paid Static API call; one JPEG per (point,
    heading) on disk; progress logged in ``attempted.parquet``.
7.  ``build_h5``           – packs everything into a per-city H5 with the
    same schema as the existing Manchester store (plus extra fields like
    ``pano_id``, ``city``, ``round_id``, ``status``).
8.  ``merge_city_h5s``     – combines per-city stores on demand.

Designed so that running any stage twice is safe and free.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import h5py
import numpy as np
import osmnx as ox
import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.auto import tqdm

from paths import CityPaths


# --- constants --------------------------------------------------------------

DEFAULT_HEADINGS: tuple[int, ...] = (0, 90, 180, 270)
DEFAULT_IMAGE_SIZE = "640x640"
DEFAULT_PITCH = 0
DEFAULT_SPACING_M = 50.0

META_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMG_URL = "https://maps.googleapis.com/maps/api/streetview"

# Per-1000 request prices (May 2026 — update if Google changes them). Used
# only for the cost estimator; not for billing.
PRICE_PER_1000_IMAGES_USD = 7.0


# --- stage 1: boundary ------------------------------------------------------

def load_boundary(
    city_name: str,
    polygon_path: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Return a single-row GeoDataFrame (EPSG:4326) with the city polygon.

    If ``polygon_path`` is given, it's loaded from disk (GeoJSON/Shapefile/etc.)
    and the geometry is unioned; otherwise OSM's geocoder is queried. The
    polygon path is the escape hatch for cities whose default OSM boundary is
    wrong (e.g. too large, includes water, etc.).
    """
    if polygon_path is not None:
        gdf = gpd.read_file(polygon_path).to_crs(4326)
        merged = gdf.geometry.union_all()
        return gpd.GeoDataFrame({"name": [city_name]}, geometry=[merged], crs=4326)
    gdf = ox.geocode_to_gdf(city_name)
    return gdf[["geometry"]].assign(name=city_name).to_crs(4326)


# --- stage 2: road graph ----------------------------------------------------

def build_road_graph(boundary_gdf: gpd.GeoDataFrame, paths: CityPaths):
    """Load the cached graph if present, otherwise download and cache it."""
    paths.ensure_dirs()
    if paths.road_graph.exists():
        return ox.load_graphml(paths.road_graph)
    poly = boundary_gdf.geometry.union_all()
    graph = ox.graph_from_polygon(poly, network_type="drive", simplify=True)
    # Edge ``length`` (metres) is computed by osmnx during graph build, but
    # belt-and-braces in case the OSMnx version doesn't auto-populate it.
    graph = ox.distance.add_edge_lengths(graph)
    ox.save_graphml(graph, paths.road_graph)
    boundary_gdf.to_file(paths.boundary_geojson, driver="GeoJSON")
    return graph


# --- stage 3: deterministic edge sampling -----------------------------------

def sample_points_along_edges(
    graph,
    spacing_m: float = DEFAULT_SPACING_M,
    dedupe_two_way: bool = True,
) -> gpd.GeoDataFrame:
    """One row per candidate point: ``point_local_id``, ``geometry``, edge keys.

    The local id is just a row index used until the points are merged into the
    master table — at that point they get globally unique ``point_id``s.
    Returned CRS is EPSG:4326 (lat/lon) for direct use with the SV API.

    A two-way street is stored in the OSM drive graph as two directed edges
    (``u->v`` and ``v->u``) whose geometry is an exact reverse of one another.
    With ``dedupe_two_way=True`` (the default) each *physical* road is sampled
    only once: each edge's coordinate sequence is canonicalised so a geometry
    and its reverse map to the same key, and the second time we see a key we
    skip it. Divided carriageways and genuinely separate roads have distinct
    (not reversed-identical) geometry, so they are correctly kept. Set this to
    ``False`` to sample every directed edge — this roughly doubles the points
    (and the image cost) on every two-way road and interleaves the two offset
    point sets into an irregular spacing; kept as an option mainly to reproduce
    or compare against the old behaviour.
    """
    edges_gdf = ox.graph_to_gdfs(graph, nodes=False)
    if "length" not in edges_gdf.columns:
        raise RuntimeError("Edges missing 'length' — was add_edge_lengths called?")

    records: list[dict] = []
    seen_geoms: set[tuple] = set()  # canonical geometry keys already sampled
    n_roads = 0       # distinct roads considered (valid geometry/length)
    n_too_short = 0   # of those, how many were too short to place any point
    short_len_m = 0.0  # running total length of the skipped short roads
    for (u, v, key), row in edges_gdf.iterrows():
        geom = row.geometry
        length_m = float(row["length"])
        if geom is None or length_m <= 0:
            continue
        if dedupe_two_way:
            # Collapse an edge and its reverse to one road. Round to 7dp (~1cm)
            # so float noise in the reversed coordinates doesn't defeat the
            # match, then take the orientation-independent (min of
            # forward/backward) key.
            coords = tuple((round(x, 7), round(y, 7)) for x, y in geom.coords)
            canon = min(coords, coords[::-1])
            if canon in seen_geoms:
                continue
            seen_geoms.add(canon)
        n_roads += 1
        # Points at spacing/2, 3*spacing/2, ... up to length. Guarantees a
        # gap of >= spacing/2 from each endpoint, so adjacent edges don't pile
        # points on top of each other at intersections.
        positions = np.arange(spacing_m / 2.0, length_m, spacing_m)
        if len(positions) == 0:
            # Road shorter than spacing/2 — no point placed (it's disregarded).
            n_too_short += 1
            short_len_m += length_m
            continue
        for d in positions:
            frac = float(d) / length_m
            pt = geom.interpolate(frac, normalized=True)
            records.append({
                "edge_u": int(u),
                "edge_v": int(v),
                "edge_key": int(key),
                "edge_offset_m": float(d),
                "edge_length_m": length_m,
                "geometry": pt,
            })

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=4326)
    # Stable order so the same graph + spacing always produces the same ids.
    gdf = gdf.sort_values(["edge_u", "edge_v", "edge_key", "edge_offset_m"]).reset_index(drop=True)
    gdf["sampled_lat"] = gdf.geometry.y.round(6)
    gdf["sampled_lon"] = gdf.geometry.x.round(6)
    # Within-batch dedupe (rare, but possible at intersection junctions).
    gdf = gdf.drop_duplicates(subset=["sampled_lat", "sampled_lon"]).reset_index(drop=True)

    # Report short roads that fell below spacing/2 and so received no point.
    pct = 100.0 * n_too_short / n_roads if n_roads else 0.0
    print(
        f"Sampled {len(gdf)} points from {n_roads} roads. "
        f"{n_too_short} roads ({pct:.1f}%) shorter than {spacing_m / 2:.0f} m were "
        f"skipped with no point ({short_len_m / 1000:.1f} km of road unsampled)."
    )
    return gdf


# --- stage 4: sampling round registration -----------------------------------

_POINT_COLS = [
    "point_id", "round_id", "sampled_lat", "sampled_lon",
    "edge_u", "edge_v", "edge_key", "edge_offset_m", "edge_length_m",
]
_MANIFEST_COLS = ["round_id", "timestamp", "spacing_m", "n_new_points", "id_lo", "id_hi"]


def _read_parquet_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


def register_sampling_round(
    candidates: gpd.GeoDataFrame,
    paths: CityPaths,
    spacing_m: float,
) -> pd.DataFrame:
    """Merge new candidates into the master point table.

    Deduplicates against any existing ``(sampled_lat, sampled_lon)`` at 6dp
    so re-running with a denser spacing only adds genuinely new points.
    Returns the *new* rows (the ones actually appended).
    """
    paths.ensure_dirs()
    existing = _read_parquet_or_empty(paths.points_parquet, _POINT_COLS)
    manifest = _read_parquet_or_empty(paths.samples_manifest, _MANIFEST_COLS)

    if len(existing):
        key_existing = set(zip(existing["sampled_lat"], existing["sampled_lon"]))
        mask_new = ~candidates.apply(
            lambda r: (r["sampled_lat"], r["sampled_lon"]) in key_existing,
            axis=1,
        )
        new = candidates.loc[mask_new].copy()
    else:
        new = candidates.copy()

    if new.empty:
        print(f"No new points: all {len(candidates)} candidates already in master table.")
        return new

    round_id = 1 if manifest.empty else int(manifest["round_id"].max()) + 1
    next_id = 0 if existing.empty else int(existing["point_id"].max()) + 1
    new = new.reset_index(drop=True)
    new["point_id"] = np.arange(next_id, next_id + len(new), dtype=np.int64)
    new["round_id"] = round_id

    new_records = new[_POINT_COLS].copy()
    combined = pd.concat([existing, new_records], ignore_index=True)
    combined.to_parquet(paths.points_parquet, index=False)

    manifest_row = pd.DataFrame([{
        "round_id": round_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spacing_m": float(spacing_m),
        "n_new_points": int(len(new)),
        "id_lo": int(new["point_id"].min()),
        "id_hi": int(new["point_id"].max()),
    }])
    pd.concat([manifest, manifest_row], ignore_index=True).to_parquet(
        paths.samples_manifest, index=False
    )
    print(
        f"Round {round_id}: added {len(new)} new points "
        f"(ids {new['point_id'].min()}..{new['point_id'].max()}); "
        f"{len(candidates) - len(new)} dedupe-skipped."
    )
    return new_records


# --- stage 5: metadata endpoint (free) --------------------------------------

# Retry on connection-level errors; HTTP-level "OVER_QUERY_LIMIT" is handled
# inside the request function because requests doesn't raise on a 200 with a
# JSON error body.
_RETRYABLE_HTTP = (requests.ConnectionError, requests.Timeout)


@retry(
    retry=retry_if_exception_type(_RETRYABLE_HTTP),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _meta_request(session: requests.Session, lat: float, lon: float, api_key: str) -> dict:
    resp = session.get(
        META_URL,
        params={"location": f"{lat},{lon}", "key": api_key},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_metadata_for_points(
    paths: CityPaths,
    api_key: str,
    only_missing: bool = True,
    sleep_between: float = 0.0,
) -> pd.DataFrame:
    """Run the (free) metadata pass for each point.

    Stores ``pano_id``, snapped ``pano_lat/lon``, ``date``, ``copyright`` and
    raw ``status`` per point. ``only_missing=True`` is the normal mode: skip
    points that already have a metadata record. Pass ``False`` to refresh.
    """
    points = _read_parquet_or_empty(paths.points_parquet, _POINT_COLS)
    if points.empty:
        raise RuntimeError("No points yet — run sampling first.")

    cols = ["point_id", "status", "pano_id", "pano_lat", "pano_lon", "date", "copyright", "fetched_at"]
    have = (
        pd.read_parquet(paths.pano_metadata)
        if paths.pano_metadata.exists()
        else pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    )
    have_ids = set(have["point_id"].astype(int)) if len(have) else set()

    todo = points if not only_missing else points[~points["point_id"].isin(have_ids)]
    if todo.empty:
        print(f"Metadata already fetched for all {len(points)} points.")
        return have

    new_rows: list[dict] = []
    with requests.Session() as session:
        for row in tqdm(todo.itertuples(index=False), total=len(todo), desc="metadata"):
            try:
                m = _meta_request(session, float(row.sampled_lat), float(row.sampled_lon), api_key)
            except Exception as ex:  # network exhaustion after retries
                m = {"status": f"REQUEST_FAILED:{type(ex).__name__}"}
            loc = m.get("location") or {}
            new_rows.append({
                "point_id": int(row.point_id),
                "status": str(m.get("status", "UNKNOWN")),
                "pano_id": m.get("pano_id"),
                "pano_lat": float(loc.get("lat")) if loc.get("lat") is not None else None,
                "pano_lon": float(loc.get("lng")) if loc.get("lng") is not None else None,
                "date": m.get("date"),
                "copyright": m.get("copyright"),
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            if sleep_between:
                time.sleep(sleep_between)

    out = pd.concat([have, pd.DataFrame(new_rows)], ignore_index=True)
    out = out.drop_duplicates(subset=["point_id"], keep="last").reset_index(drop=True)
    out.to_parquet(paths.pano_metadata, index=False)
    n_ok = int((out["status"] == "OK").sum())
    print(f"Metadata: {n_ok}/{len(out)} points have a pano (status=OK).")
    return out


# --- stage 6: image download (paid) -----------------------------------------

_ATTEMPTED_COLS = ["point_id", "heading", "status", "http_status", "bytes", "attempted_at", "error"]


def _read_attempted(paths: CityPaths) -> pd.DataFrame:
    if paths.attempted.exists():
        return pd.read_parquet(paths.attempted)
    return pd.DataFrame({c: pd.Series(dtype="object") for c in _ATTEMPTED_COLS})


@retry(
    retry=retry_if_exception_type(_RETRYABLE_HTTP),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _image_request(session: requests.Session, params: dict) -> requests.Response:
    resp = session.get(IMG_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp


def download_images(
    paths: CityPaths,
    api_key: str,
    headings: Sequence[int] = DEFAULT_HEADINGS,
    image_size: str = DEFAULT_IMAGE_SIZE,
    pitch: int = DEFAULT_PITCH,
    max_points: int | None = None,
    dry_run: bool = True,
) -> pd.DataFrame:
    """Download the four-heading image set for every point with status=OK.

    ``dry_run=True`` (the default) prints what would be fetched and the cost
    estimate, but makes no paid API calls. Skips points whose images are
    already on disk *and* logged successful in ``attempted.parquet``, so this
    is the resume function: rerun after a crash, pay nothing extra.
    """
    points = _read_parquet_or_empty(paths.points_parquet, _POINT_COLS)
    if not paths.pano_metadata.exists():
        raise RuntimeError("Run fetch_metadata_for_points first.")
    meta = pd.read_parquet(paths.pano_metadata)
    eligible = points.merge(meta[meta["status"] == "OK"], on="point_id", how="inner")
    if max_points is not None:
        eligible = eligible.head(int(max_points))

    attempted = _read_attempted(paths)
    done_keys = set(
        zip(
            attempted.loc[attempted["status"] == "ok", "point_id"].astype(int),
            attempted.loc[attempted["status"] == "ok", "heading"].astype(int),
        )
    ) if len(attempted) else set()

    todo: list[tuple[int, str, int]] = []  # (point_id, pano_id, heading)
    for row in eligible.itertuples(index=False):
        for h in headings:
            if (int(row.point_id), int(h)) in done_keys:
                continue
            if paths.image_path(int(row.point_id), int(h)).exists():
                # Image present on disk but no log row — record retroactively.
                continue
            todo.append((int(row.point_id), str(row.pano_id), int(h)))

    n_requests = len(todo)
    est_cost = n_requests / 1000.0 * PRICE_PER_1000_IMAGES_USD
    print(
        f"Eligible points: {len(eligible)}; image requests to make: {n_requests}; "
        f"estimated cost: ${est_cost:.2f} (at ${PRICE_PER_1000_IMAGES_USD}/1000)."
    )
    if dry_run:
        print("dry_run=True — no images downloaded. Set dry_run=False to proceed.")
        return attempted

    if not todo:
        print("Nothing to download.")
        return attempted

    new_log: list[dict] = []
    with requests.Session() as session:
        for point_id, pano_id, heading in tqdm(todo, desc="images"):
            out_path = paths.image_path(point_id, heading)
            params = {
                "size": image_size,
                "pano": pano_id,
                "heading": str(int(heading)),
                "pitch": str(int(pitch)),
                "key": api_key,
            }
            entry = {
                "point_id": point_id, "heading": heading,
                "attempted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "http_status": None, "bytes": 0, "error": None,
            }
            try:
                resp = _image_request(session, params)
                entry["http_status"] = resp.status_code
                content = resp.content
                # Google returns a generic "no imagery" JPEG when pano lookup
                # via lat/lon fails. We always pass a real pano_id, so a
                # 200 here should be a real image; still, sanity-check.
                if not content or len(content) < 1000:
                    entry["status"] = "error"
                    entry["error"] = "response too small to be a real image"
                else:
                    out_path.write_bytes(content)
                    entry["bytes"] = len(content)
                    entry["status"] = "ok"
            except Exception as ex:
                entry["status"] = "error"
                entry["error"] = f"{type(ex).__name__}: {ex}"
            new_log.append(entry)

    new_df = pd.DataFrame(new_log)
    out = pd.concat([attempted, new_df], ignore_index=True)
    # Keep the most recent attempt per (point_id, heading).
    out = (
        out.sort_values("attempted_at")
        .drop_duplicates(subset=["point_id", "heading"], keep="last")
        .reset_index(drop=True)
    )
    out.to_parquet(paths.attempted, index=False)
    n_ok = int((new_df["status"] == "ok").sum())
    print(f"Downloaded {n_ok}/{len(new_df)} images this run; total cost ~${n_ok / 1000 * PRICE_PER_1000_IMAGES_USD:.2f}.")
    return out


# --- stage 7: H5 build ------------------------------------------------------

def build_h5(
    paths: CityPaths,
    city_name: str,
    headings: Sequence[int] = DEFAULT_HEADINGS,
    image_size: str = DEFAULT_IMAGE_SIZE,
) -> Path:
    """(Re)build the per-city H5 from the on-disk state.

    Row count = number of points that have been *attempted at least once*
    (either via the metadata endpoint or an image download). Failed/no-pano
    points are kept as rows with ``images_present[i, :] = False`` so we
    preserve the record that they were tried (matches Manchester behaviour).
    """
    points = _read_parquet_or_empty(paths.points_parquet, _POINT_COLS)
    if points.empty:
        raise RuntimeError("No points yet.")
    meta = (
        pd.read_parquet(paths.pano_metadata)
        if paths.pano_metadata.exists() else
        pd.DataFrame(columns=["point_id", "status", "pano_id", "pano_lat", "pano_lon", "date", "copyright"])
    )
    attempted = _read_attempted(paths)

    # Include only points we've actually tried. If the user only sampled and
    # never ran metadata, the H5 would be empty — flag that.
    attempted_ids = set(meta["point_id"].astype(int)) if len(meta) else set()
    rows = points[points["point_id"].isin(attempted_ids)].copy()
    if rows.empty:
        raise RuntimeError("No points have metadata yet — run fetch_metadata_for_points first.")
    rows = rows.sort_values("point_id").reset_index(drop=True)
    rows = rows.merge(meta, on="point_id", how="left", suffixes=("", "_meta"))

    N = len(rows)
    n_h = len(headings)
    out_path = paths.h5
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use latin-1 for variable strings stored as fixed-length S<N> bytes;
    # callers should decode with the same codec. Manchester uses ASCII for
    # short fields like dates — we keep that for direct compatibility.
    with h5py.File(out_path, "w") as f:
        f.attrs["city"] = city_name
        f.attrs["city_slug"] = paths.slug
        f.attrs["image_size"] = image_size
        f.attrs["headings"] = np.asarray(headings, dtype=np.int32)
        f.attrs["built_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # Manchester-compatible datasets ------------------------------------
        f.create_dataset("point_id", data=rows["point_id"].astype(np.int64).to_numpy())
        f.create_dataset("latitude", data=rows["sampled_lat"].astype(np.float64).to_numpy())
        f.create_dataset("longitude", data=rows["sampled_lon"].astype(np.float64).to_numpy())
        f.create_dataset(
            "date",
            data=np.asarray(
                [str(d or "").encode("ascii", "ignore")[:10] for d in rows.get("date", [""] * N)],
                dtype="S10",
            ),
        )
        f.create_dataset("image_paths", (N, n_h), dtype="S512")
        f.create_dataset("images_present", (N, n_h), dtype="bool")
        img_ds = f.create_dataset(
            "images_jpeg", shape=(N, n_h), dtype=h5py.vlen_dtype(np.dtype("uint8"))
        )

        # Extra per-row context (cheap, lossless) ---------------------------
        f.create_dataset("sampled_lat", data=rows["sampled_lat"].astype(np.float64).to_numpy())
        f.create_dataset("sampled_lon", data=rows["sampled_lon"].astype(np.float64).to_numpy())
        f.create_dataset(
            "pano_lat",
            data=rows["pano_lat"].astype("float64").fillna(np.nan).to_numpy(),
        )
        f.create_dataset(
            "pano_lon",
            data=rows["pano_lon"].astype("float64").fillna(np.nan).to_numpy(),
        )
        f.create_dataset(
            "pano_id",
            data=np.asarray([str(p or "").encode("utf-8")[:128] for p in rows["pano_id"]], dtype="S128"),
        )
        f.create_dataset(
            "copyright",
            data=np.asarray([str(c or "").encode("utf-8")[:128] for c in rows.get("copyright", [""] * N)], dtype="S128"),
        )
        f.create_dataset(
            "status",
            data=np.asarray([str(s or "").encode("ascii", "ignore")[:32] for s in rows["status"]], dtype="S32"),
        )
        f.create_dataset(
            "city",
            data=np.asarray([city_name.encode("utf-8")[:64]] * N, dtype="S64"),
        )
        f.create_dataset("round_id", data=rows["round_id"].astype(np.int32).to_numpy())
        f.create_dataset("heading", data=np.tile(np.asarray(headings, dtype=np.int32), (N, 1)))

        # Fill image slots --------------------------------------------------
        # Index attempted by (point_id, heading) for fast lookup.
        att_ok = attempted[attempted["status"] == "ok"]
        att_key = set(zip(att_ok["point_id"].astype(int), att_ok["heading"].astype(int)))

        empty = np.array([], dtype=np.uint8)
        for i, point_id in enumerate(rows["point_id"].astype(int).to_numpy()):
            for j, h in enumerate(headings):
                path = paths.image_path(point_id, int(h))
                rel_path = str(path.relative_to(paths.root)).encode("utf-8")[:512]
                f["image_paths"][i, j] = rel_path
                if (int(point_id), int(h)) in att_key and path.exists():
                    f["images_jpeg"][i, j] = np.frombuffer(path.read_bytes(), dtype=np.uint8)
                    f["images_present"][i, j] = True
                else:
                    img_ds[i, j] = empty
                    f["images_present"][i, j] = False
            if i and i % 5000 == 0:
                print(f"  packed {i}/{N} rows")

    n_present = 0
    with h5py.File(out_path, "r") as f:
        n_present = int(f["images_present"][:].any(axis=1).sum())
    print(
        f"Wrote {out_path} — {N} rows, {n_present} with >=1 image "
        f"({100.0 * n_present / max(N, 1):.1f}%)."
    )
    return out_path


# --- stage 8: cross-city merge ---------------------------------------------

def merge_city_h5s(h5_paths: Iterable[Path], out_path: Path) -> Path:
    """Concatenate per-city H5 stores into one combined file.

    All inputs must share the same ``headings`` length and ``image_size``.
    The combined file has a ``city`` row dataset so any subset can be
    re-extracted by filtering.
    """
    h5_paths = [Path(p) for p in h5_paths]
    if not h5_paths:
        raise ValueError("No inputs")

    # Two passes: collect sizes + sanity-check schemas, then copy.
    n_total = 0
    n_h = None
    image_size = None
    for p in h5_paths:
        with h5py.File(p, "r") as f:
            n = f["point_id"].shape[0]
            nh = f["images_jpeg"].shape[1]
            if n_h is None:
                n_h = nh
                image_size = f.attrs.get("image_size", DEFAULT_IMAGE_SIZE)
            elif nh != n_h:
                raise ValueError(f"{p} has {nh} heading slots, expected {n_h}")
            n_total += n

    fixed_scalar = {
        "point_id": np.int64, "latitude": np.float64, "longitude": np.float64,
        "sampled_lat": np.float64, "sampled_lon": np.float64,
        "pano_lat": np.float64, "pano_lon": np.float64,
        "round_id": np.int32,
    }
    fixed_str = {"date": "S10", "pano_id": "S128", "copyright": "S128", "status": "S32", "city": "S64"}

    with h5py.File(out_path, "w") as out:
        out.attrs["image_size"] = image_size
        out.attrs["built_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for k, dt in fixed_scalar.items():
            out.create_dataset(k, (n_total,), dtype=dt)
        for k, dt in fixed_str.items():
            out.create_dataset(k, (n_total,), dtype=dt)
        out.create_dataset("image_paths", (n_total, n_h), dtype="S512")
        out.create_dataset("images_present", (n_total, n_h), dtype="bool")
        out.create_dataset("heading", (n_total, n_h), dtype=np.int32)
        out.create_dataset(
            "images_jpeg", shape=(n_total, n_h), dtype=h5py.vlen_dtype(np.dtype("uint8"))
        )

        offset = 0
        for p in h5_paths:
            with h5py.File(p, "r") as f:
                n = f["point_id"].shape[0]
                sl = slice(offset, offset + n)
                for k in list(fixed_scalar) + list(fixed_str):
                    if k in f:
                        out[k][sl] = f[k][:]
                out["image_paths"][sl] = f["image_paths"][:]
                out["images_present"][sl] = f["images_present"][:]
                out["heading"][sl] = f["heading"][:]
                # vlen needs row-wise assignment.
                for i in range(n):
                    for j in range(n_h):
                        out["images_jpeg"][offset + i, j] = f["images_jpeg"][i, j]
                offset += n
                print(f"  merged {p.name}: {n} rows")
    print(f"Combined H5 written to {out_path} ({n_total} rows).")
    return out_path


# --- small utilities --------------------------------------------------------

def cost_estimate(n_points: int, n_headings: int = 4) -> str:
    n_req = n_points * n_headings
    usd = n_req / 1000.0 * PRICE_PER_1000_IMAGES_USD
    return f"{n_points} points x {n_headings} headings = {n_req} requests; ~${usd:.2f} USD"


def show_h5_summary(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        n = f["point_id"].shape[0]
        present = f["images_present"][:]
        info = {
            "rows": n,
            "rows_with_any_image": int(present.any(axis=1).sum()),
            "total_images": int(present.sum()),
            "city": f.attrs.get("city", "?"),
            "image_size": f.attrs.get("image_size", "?"),
            "headings": list(np.asarray(f.attrs.get("headings", []))),
        }
    print(json.dumps(info, indent=2, default=str))
    return info
