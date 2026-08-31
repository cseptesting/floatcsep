"""
Python subprocess handlers for complex data processing.
Called by Next.js API routes via subprocess.

Performance optimizations:
- File-based JSON caching to avoid reparsing forecast files
- Vectorized numpy operations for cell data generation
- Uses orjson for faster JSON serialization when available
"""
import sys
import hashlib
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np

# Try to use orjson for faster serialization, fall back to standard json
try:
    import orjson
    def json_dumps(obj):
        return orjson.dumps(obj).decode('utf-8')
    def json_loads(s):
        return orjson.loads(s)
except ImportError:
    import json
    def json_dumps(obj):
        return json.dumps(obj, separators=(',', ':'))  # Compact output
    def json_loads(s):
        return json.loads(s)

# Standard json for file operations that need pretty printing
import json as std_json

from floatcsep.utils.file_io import GriddedForecastParsers, CatalogForecastParsers, CatalogParser

# Cache directory for processed forecast JSON
CACHE_DIR = Path(__file__).parent / ".cache" / "forecast_cache"


def get_cache_path(forecast_path: str, is_catalog_fc: bool) -> Path:
    """Generate a cache file path based on the forecast file."""
    # Create hash of the forecast path for cache key
    path_hash = hashlib.md5(forecast_path.encode()).hexdigest()[:12]
    fc_type = "catalog" if is_catalog_fc else "gridded"
    return CACHE_DIR / f"{fc_type}_{path_hash}.json"


def is_cache_valid(cache_path: Path, source_path: Path) -> bool:
    """Check if cache is valid (exists and newer than source)."""
    if not cache_path.exists():
        return False
    if not source_path.exists():
        return False
    return cache_path.stat().st_mtime > source_path.stat().st_mtime


def load_catalog_data(catalog_path: str, app_root: str) -> Dict[str, Any]:
    """Load catalog and return event data as JSON."""
    path = Path(app_root) / catalog_path

    try:
        if path.suffix.lower() == ".json":
            try:
                catalog = CatalogParser.json(str(path))
            except std_json.JSONDecodeError:
                catalog = CatalogParser.ascii(str(path))
        else:
            catalog = CatalogParser.ascii(str(path))
    except Exception as e:
        return {"error": str(e)}

    if catalog is None or catalog.event_count == 0:
        return {"events": [], "count": 0, "bbox": None}

    lons = catalog.get_longitudes()
    lats = catalog.get_latitudes()
    mags = catalog.get_magnitudes()
    dts = catalog.get_datetimes()
    event_ids = catalog.get_event_ids()

    events = []
    for i in range(len(lons)):
        events.append({
            "lon": float(lons[i]),
            "lat": float(lats[i]),
            "magnitude": float(mags[i]),
            "time": dts[i].isoformat(),
            "event_id": str(event_ids[i]) if isinstance(event_ids[i], (bytes, bytearray)) else event_ids[i].decode('utf-8') if hasattr(event_ids[i], 'decode') else str(event_ids[i]),
        })

    # Calculate bbox manually to handle antimeridian correctly
    if len(lons) > 0 and len(lats) > 0:
        min_lat = float(np.min(lats))
        max_lat = float(np.max(lats))
        min_lon = float(np.min(lons))
        max_lon = float(np.max(lons))

        # Check if data crosses the antimeridian
        has_positive = any(lon > 90 for lon in lons)
        has_negative = any(lon < -90 for lon in lons)

        if has_positive and has_negative:
            positive_lons = [lon for lon in lons if lon > 0]
            negative_lons = [lon for lon in lons if lon < 0]

            if positive_lons and negative_lons:
                west = float(min(positive_lons))
                east = float(max(negative_lons))
                bbox = [west, min_lat, east, max_lat]
            else:
                bbox = [min_lon, min_lat, max_lon, max_lat]
        else:
            bbox = [min_lon, min_lat, max_lon, max_lat]
    else:
        bbox = None

    return {
        "events": events,
        "count": catalog.event_count,
        "bbox": bbox,
    }


def load_forecast_data(
    forecast_path: str,
    app_root: str,
    region_data: Dict[str, Any],
    is_catalog_fc: bool = False,
) -> Dict[str, Any]:
    """Load forecast and return gridded data as JSON.
    
    Uses file-based caching to avoid reparsing on subsequent loads.
    """
    path = Path(app_root) / forecast_path

    if not path.exists():
        return {"error": f"Forecast file not found: {forecast_path}"}

    # Check cache first
    cache_path = get_cache_path(forecast_path, is_catalog_fc)
    if is_cache_valid(cache_path, path):
        try:
            with open(cache_path, 'r') as f:
                return json_loads(f.read())
        except Exception:
            pass  # Cache read failed, regenerate

    suffix = path.suffix.lower()

    try:
        if is_catalog_fc:
            from pycsep.core.regions import CartesianGrid2D

            if not region_data or not region_data.get('bbox'):
                return {"error": "Region data required for catalog forecasts"}

            bbox = region_data['bbox']
            dh = region_data.get('dh', 0.1)

            region = CartesianGrid2D.from_origins(
                origins=None,
                dh=dh,
                mask=None,
            )

            cf = CatalogForecastParsers.csv(
                str(path),
                region=region,
                filter_spatial=True,
                apply_filters=True,
                store=False,
            )

            gf = cf.get_expected_rates(verbose=False)
            rates = np.asarray(gf.data, dtype='float32')
            region_out = getattr(gf, 'region', region)
        else:
            # Gridded forecast
            if suffix == ".dat":
                rates, region_out, mags = GriddedForecastParsers.dat(str(path))
            elif suffix in (".xml", ".gml"):
                rates, region_out, mags = GriddedForecastParsers.xml(str(path))
            elif suffix in (".csv", ".txt"):
                rates, region_out, mags = GriddedForecastParsers.csv(str(path))
            elif suffix in (".h5", ".hdf5"):
                rates, region_out, mags = GriddedForecastParsers.hdf5(str(path))
            else:
                return {"error": f"Unsupported forecast file extension: {suffix}"}

        # Sum across magnitude bins to get total rate per cell
        total_rates = rates.sum(axis=1).astype('float32')

        # Get cell coordinates
        origins = region_out.origins()
        dh = float(region_out.dh)

        # Vectorized cell center calculation
        lon_c = origins[:, 0] + 0.5 * dh
        lat_c = origins[:, 1] + 0.5 * dh

        # Filter to non-zero rates using numpy mask (vectorized)
        mask = total_rates > 0
        lon_c_filtered = lon_c[mask]
        lat_c_filtered = lat_c[mask]
        rates_filtered = total_rates[mask]

        # Build cell data using list comprehension (faster than loop with append)
        # Convert to Python floats in one go
        cells = [
            {"lon": float(lon_c_filtered[i]), "lat": float(lat_c_filtered[i]), "rate": float(rates_filtered[i])}
            for i in range(len(lon_c_filtered))
        ]

        # Calculate vmin/vmax in log10 space
        with np.errstate(divide='ignore', invalid='ignore'):
            log_rates = np.log10(rates_filtered)

        finite = np.isfinite(log_rates)
        if np.any(finite):
            vmin = float(np.nanmin(log_rates[finite]))
            vmax = float(np.nanmax(log_rates[finite]))
        else:
            vmin = 0.0
            vmax = 1.0

        result = {
            "cells": cells,
            "dh": dh,
            "vmin": vmin,
            "vmax": vmax,
        }

        # Write to cache
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                f.write(json_dumps(result))
        except Exception:
            pass  # Cache write failed, continue without caching

        return result

    except Exception as e:
        return {"error": f"Failed to load forecast: {str(e)}"}


if __name__ == "__main__":
    # Command-line interface for subprocess calls
    if len(sys.argv) < 2:
        print(json_dumps({"error": "No command specified"}))
        sys.exit(1)

    command = sys.argv[1]

    if command == "load_catalog":
        if len(sys.argv) < 4:
            print(json_dumps({"error": "Missing arguments for load_catalog"}))
            sys.exit(1)
        catalog_path = sys.argv[2]
        app_root = sys.argv[3]
        result = load_catalog_data(catalog_path, app_root)
        print(json_dumps(result))

    elif command == "load_forecast":
        if len(sys.argv) < 6:
            print(json_dumps({"error": "Missing arguments for load_forecast"}))
            sys.exit(1)
        forecast_path = sys.argv[2]
        app_root = sys.argv[3]
        region_arg = sys.argv[4]
        if Path(region_arg).exists():
            with open(region_arg, 'r') as f:
                region_data = json_loads(f.read())
        else:
            region_data = json_loads(region_arg)

        is_catalog_fc = sys.argv[5].lower() == "true"
        result = load_forecast_data(forecast_path, app_root, region_data, is_catalog_fc)
        print(json_dumps(result))

    else:
        print(json_dumps({"error": f"Unknown command: {command}"}))
        sys.exit(1)
