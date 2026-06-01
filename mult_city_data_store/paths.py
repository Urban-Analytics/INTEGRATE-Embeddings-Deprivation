"""Per-city path helpers for the multi-city Street View pipeline.

Each city gets its own directory under ``mult_city_data_store/data/<city_slug>/``
containing the road graph, the master point table, the attempted-downloads log,
the on-disk JPEG cache, and the final H5 store. Slugs are derived
deterministically from the city name so reruns end up in the same folder.
"""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# Directory containing this file. Used as the anchor for all per-city paths so
# the pipeline can be invoked from anywhere (notebook, shell, tests).
PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"


def slugify(name: str) -> str:
    """Produce a filesystem-safe slug from a free-form city name.

    "Leeds, UK" -> "leeds_uk"; "Kingston upon Hull" -> "kingston_upon_hull".
    Stable across runs so the same input always maps to the same folder.
    """
    norm = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    norm = re.sub(r"[^A-Za-z0-9]+", "_", norm).strip("_").lower()
    if not norm:
        raise ValueError(f"City name {name!r} produced an empty slug")
    return norm


@dataclass(frozen=True)
class CityPaths:
    """All file paths for a single city. Construct via ``CityPaths.for_city``."""

    slug: str
    root: Path

    @classmethod
    def for_city(cls, city_name: str, base_dir: Path | None = None) -> "CityPaths":
        slug = slugify(city_name)
        root = (base_dir or DATA_DIR) / slug
        return cls(slug=slug, root=root)

    # --- per-city files -----------------------------------------------------
    @property
    def boundary_geojson(self) -> Path:
        return self.root / "boundary.geojson"

    @property
    def road_graph(self) -> Path:
        return self.root / "road_graph.graphml"

    @property
    def points_parquet(self) -> Path:
        # Master point table; append-only across sampling rounds.
        return self.root / "points.parquet"

    @property
    def samples_manifest(self) -> Path:
        # One row per sampling round (round_id, spacing, n_new_points, id range).
        return self.root / "samples_manifest.parquet"

    @property
    def pano_metadata(self) -> Path:
        # Result of the free metadata endpoint; keyed by point_id.
        return self.root / "pano_metadata.parquet"

    @property
    def attempted(self) -> Path:
        # Per-point download status log (ok / no_pano / error).
        return self.root / "attempted.parquet"

    @property
    def images_dir(self) -> Path:
        return self.root / "street_images"

    @property
    def h5(self) -> Path:
        return self.root / "street_data.h5"

    def __repr__(self) -> str:
        lines = [f"CityPaths(slug={self.slug!r})", f"  root : {self.root}"]
        props = [
            ("boundary_geojson", self.boundary_geojson),
            ("road_graph",       self.road_graph),
            ("points_parquet",   self.points_parquet),
            ("samples_manifest", self.samples_manifest),
            ("pano_metadata",    self.pano_metadata),
            ("attempted",        self.attempted),
            ("images_dir",       self.images_dir),
            ("h5",               self.h5),
        ]
        width = max(len(name) for name, _ in props)
        for name, path in props:
            exists = "✓" if path.exists() else "✗"
            lines.append(f"  {name:<{width}} {exists}  {path}")
        return "\n".join(lines)

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def image_path(self, point_id: int, heading: int) -> Path:
        return self.images_dir / f"point{int(point_id)}_heading{int(heading)}.jpg"


def get_api_key() -> str:
    """Read the Street View Static API key from the environment.

    Kept here so the rest of the pipeline never touches the raw value.
    """
    key = os.environ.get("GOOGLE_STREETVIEW_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GOOGLE_STREETVIEW_API_KEY is not set. Export it in your shell, "
            "e.g. `export GOOGLE_STREETVIEW_API_KEY=...` before launching Jupyter."
        )
    return key
