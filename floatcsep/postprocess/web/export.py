"""
Writes an experiment as a static dashboard folder: the manifest, the catalog, the forecasts,
the evaluation results and the front-end files. The folder can be served by any static web
server (``floatcsep serve``, ``python -m http.server``, GitHub Pages, ...).
"""

import base64
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from floatcsep.utils.helpers import timewindow2str
from .manifest import build_manifest
from .meta import citation, fair_scores, find_cache

log = logging.getLogger("floatLogger")

STATIC = Path(__file__).resolve().parent / "static"
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def _dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"), allow_nan=False)


def _round(arr: np.ndarray, nd: int) -> List[float]:
    return np.round(np.asarray(arr, dtype=float), nd).tolist()


def _finite(value: Any) -> Any:
    """Replaces non-finite numbers by None so the JSON stays standard."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _outline(origins: np.ndarray, dh: float) -> Optional[Dict[str, Any]]:
    """Union of the grid cells as a GeoJSON geometry (needs shapely, a pyCSEP dependency)."""
    try:
        from shapely.geometry import box, mapping
        from shapely.ops import unary_union
    except ImportError:
        return None
    eps = dh * 0.02
    cells = [box(x - eps, y - eps, x + dh + eps, y + dh + eps) for x, y in origins]
    geom = unary_union(cells).buffer(-eps, join_style=2).simplify(dh * 0.01)
    return mapping(geom)


def export_grid(region: Any, out: Path) -> Optional[Dict[str, Any]]:
    if region is None:
        return None
    origins = np.asarray(region.origins(), dtype=float)
    dh = float(region.dh)
    grid = {"dh": dh, "origins": _round(origins, 5), "outline": _outline(origins, dh)}
    _dump(grid, out / "grid.json")
    return {"file": "grid.json", "n_cells": int(len(origins))}


def _columns(cat: Any) -> Dict[str, Any]:
    d = cat.data
    return {
        "id": [i.decode() if isinstance(i, bytes) else str(i) for i in d["id"]],
        "t": np.asarray(d["origin_time"], dtype=np.int64).tolist(),
        "lat": _round(d["latitude"], 4),
        "lon": _round(d["longitude"], 4),
        "depth": _round(d["depth"], 2),
        "mag": _round(d["magnitude"], 2),
    }


def _iso(v: Any) -> Any:
    from .manifest import _iso as iso
    return iso(v) if hasattr(v, "isoformat") else v


def export_catalog(experiment: Any, manifest: Dict[str, Any], out: Path) -> Optional[Dict[str, Any]]:
    """
    Writes the testing and input catalogs as floatCSEP produced them.

    The testing catalog is read from the files stored in ``results/<window>/catalog/``; each
    event keeps the list of windows whose testing catalog contains it. The input catalog is
    the one handed to the models (``input_config.catalog``), produced with the same
    function floatCSEP uses to write it into each model's input folder, up to the start of
    the last window. Events of the input catalog for a given window are those before its
    start.
    """
    repo = experiment.catalog_repo
    rcfg = experiment.region_config or {}
    windows = manifest["time_windows"]

    events: Dict[Any, int] = {}
    test = {k: [] for k in ("id", "t", "lat", "lon", "depth", "mag")}
    test["w"] = []
    missing = []
    for wi, win in enumerate(windows):
        try:
            cat = repo.get_test_cat(win["key"])
        except Exception:
            missing.append(win["label"])
            continue
        cols = _columns(cat)
        for j in range(len(cols["id"])):
            key = (cols["id"][j], cols["t"][j])
            if key not in events:
                events[key] = len(test["id"])
                for c in ("id", "t", "lat", "lon", "depth", "mag"):
                    test[c].append(cols[c][j])
                test["w"].append([])
            test["w"][events[key]].append(wi)
    if missing:
        log.warning(f"No stored testing catalog for {len(missing)} window(s), e.g. {missing[0]}")
    test["n"] = len(test["id"])
    test["filters"] = {
        "region": manifest["region"].get("name"),
        "mag_min": rcfg.get("mag_min"),
        "mag_max": rcfg.get("mag_max"),
        "depth_min": rcfg.get("depth_min"),
        "depth_max": rcfg.get("depth_max"),
        "time": "Within each window",
    }
    test["stored"] = "results/<window>/catalog/"

    inp = None
    td = manifest["experiment"]["exp_class"] == "Time-Dependent"
    if td or experiment.input_config:
        try:
            cat = repo.get_input_cat(windows[-1]["key"])
            inp = _columns(cat)
            inp["n"] = len(inp["id"])
            cfg = repo.input_cat_config
            reg = cfg.get("region")
            inp["filters"] = {
                "region": getattr(reg, "name", None) if reg is not None else None,
                "mag_min": cfg.get("mag_min"),
                "mag_max": cfg.get("mag_max"),
                "depth_min": cfg.get("depth_min"),
                "depth_max": cfg.get("depth_max"),
                "start_date": _iso(cfg.get("start_date")),
                "time": "Before each window start",
            }
            inp["stored"] = "Model input folders"
        except Exception as exc:
            log.warning(f"Could not build the input catalog: {exc}")
            inp = None

    data = {"testing": test, "input": inp}
    _dump(data, out / "catalog.json")
    return {
        "file": "catalog.json",
        "n_testing": test["n"],
        "n_input": inp["n"] if inp else None,
        "source": str(repo.cat_path) if repo.cat_path else None,
    }


def _encode_rates(rates: np.ndarray) -> Dict[str, Any]:
    """
    Cumulative rates per cell and magnitude threshold, as quantized log10 values.

    Parameters
    ----------
    rates : ndarray, shape (n_cells, n_mags)
        Expected number of events per cell and magnitude bin.

    Returns
    -------
    dict
        ``vmin``/``vmax`` in log10 units, ``totals`` per threshold, and ``data`` as a
        base64 little-endian uint16 array (row-major, cells x thresholds). 0 means no rate,
        1..65535 map linearly onto [vmin, vmax].
    """
    cum = np.cumsum(rates[:, ::-1], axis=1)[:, ::-1]
    pos = cum > 0
    with np.errstate(divide="ignore"):
        lg = np.where(pos, np.log10(np.where(pos, cum, 1.0)), np.nan)
    if not pos.any():
        return {"vmin": 0.0, "vmax": 1.0, "totals": _round(cum.sum(axis=0), 4), "data": ""}
    vmin, vmax = float(np.nanmin(lg)), float(np.nanmax(lg))
    if vmax <= vmin:
        vmax = vmin + 1.0
    q = np.zeros(lg.shape, dtype=np.uint16)
    q[pos] = np.round((lg[pos] - vmin) / (vmax - vmin) * 65534).astype(np.uint16) + 1
    return {
        "vmin": vmin,
        "vmax": vmax,
        "totals": _round(cum.sum(axis=0), 4),
        "data": base64.b64encode(q.astype("<u2").tobytes()).decode("ascii"),
    }


def _read_rates(fc: Any):
    """Expected rates, region and magnitudes of a gridded or catalog-based forecast."""
    if fc is None:
        return None, None, None
    if hasattr(fc, "get_expected_rates"):
        fc = fc.get_expected_rates(verbose=False)
    rates = np.asarray(fc.data, dtype=float)
    rates = np.where(np.isfinite(rates), rates, 0.0)
    return rates, fc.region, np.asarray(fc.magnitudes, dtype=float)


def export_forecasts(experiment: Any, manifest: Dict[str, Any], out: Path) -> None:
    region = getattr(experiment, "region", None)
    ref = np.asarray(region.origins(), dtype=float) if region is not None else None
    windows = manifest["time_windows"]

    for mi, (model, mrec) in enumerate(zip(experiment.models, manifest["models"])):
        for win in windows:
            tstr = win["key"]
            try:
                rates, fregion, mags = _read_rates(model.get_forecast(tstr, region=region))
            except ValueError as exc:
                if "more catalogs than" not in str(exc):
                    log.warning(f"Could not read forecast {model.name} / {tstr}: {exc}")
                    continue
                log.warning(
                    f"{model.name} / {tstr}: {exc}. Reading it again with the number of "
                    f"catalogs found in the file (catalog_id may start at 1)."
                )
                try:
                    fc = model.repository.load_forecast(tstr, name=model.name, region=region)
                    rates, fregion, mags = _read_rates(fc)
                    mrec.setdefault("notes", []).append(
                        f"Window {int(win['id'][1:]) + 1}: more catalogs in the file than n_sims"
                    )
                except Exception as exc2:
                    log.warning(f"Could not read forecast {model.name} / {tstr}: {exc2}")
                    continue
            except Exception as exc:
                log.warning(f"Could not read forecast {model.name} / {tstr}: {exc}")
                continue
            if rates is None:
                continue

            rec = {
                "model": mrec["id"],
                "window": win["id"],
                "mags": _round(mags, 3),
                "n_cells": int(rates.shape[0]),
                **_encode_rates(rates),
            }
            origins = np.asarray(fregion.origins(), dtype=float)
            if ref is not None and origins.shape == ref.shape and np.allclose(origins, ref):
                rec["grid"] = "region"
            else:
                rec["grid"] = {"dh": float(fregion.dh), "origins": _round(origins, 5)}
            rel = f"forecasts/{mrec['id']}_{win['id']}.json"
            _dump(rec, out / rel)
            mrec["forecasts"][win["id"]] = rel
            log.info(f"\tForecast {model.name} / {win['label']}")


def _summary(rec: Dict[str, Any], test_type: Optional[str]) -> Dict[str, Any]:
    dist = rec.get("test_distribution")
    obs = rec.get("observed_statistic")
    out = {
        "name": rec.get("name"),
        "observed_statistic": obs,
        "quantile": rec.get("quantile"),
        "status": rec.get("status"),
        "min_mw": rec.get("min_mw"),
    }
    if isinstance(dist, list) and dist and isinstance(dist[0], str):
        out["dist"] = {"type": dist[0], "params": dist[1:]}
    elif test_type == "comparative" and isinstance(dist, list) and len(dist) == 2:
        out["ci"] = dist
    elif isinstance(dist, list) and dist and isinstance(obs, (int, float)):
        arr = np.asarray(dist, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            out["n_sim"] = int(arr.size)
            out["dist_q"] = _round(np.percentile(arr, [2.5, 5, 25, 50, 75, 95, 97.5]), 5)
    return out


def export_results(experiment: Any, manifest: Dict[str, Any], out: Path) -> Dict[str, Any]:
    reg = experiment.registry
    wid = {w["key"]: w["id"] for w in manifest["time_windows"]}
    tid = {t["name"]: t for t in manifest["tests"]}
    mid = {m["name"]: m["id"] for m in manifest["models"]}
    index: List[Dict[str, Any]] = []

    for wkey, tests in (reg.results or {}).items():
        for tname, models in tests.items():
            for mname, rel in models.items():
                path = reg.abs(reg.run_dir, rel)
                if not path.exists() or tname not in tid or mname not in mid:
                    continue
                try:
                    with open(path) as f:
                        rec = json.load(f)
                except Exception as exc:
                    log.warning(f"Could not read result {path}: {exc}")
                    continue
                t = tid[tname]
                entry = {"window": wid.get(wkey, wkey), "test": t["id"], "model": mid[mname]}
                entry.update(_summary(rec, t.get("type")))
                dist = rec.get("test_distribution")
                if "dist_q" in entry:
                    frel = f"results/{entry['window']}_{t['id']}_{entry['model']}.json"
                    _dump({"test_distribution": _finite(dist)}, out / frel)
                    entry["dist_file"] = frel
                elif isinstance(dist, list) and "ci" not in entry and "dist" not in entry:
                    entry["test_distribution"] = dist
                index.append(_finite(entry))

    _dump(index, out / "results" / "index.json")
    return {"index": "results/index.json", "n": len(index)}


def export_figures(experiment: Any, manifest: Dict[str, Any], out: Path) -> Dict[str, Any]:
    reg = experiment.registry
    figs = reg.figures or {}
    wid = {w["key"]: w["id"] for w in manifest["time_windows"]}
    mid = {m["name"]: m["id"] for m in manifest["models"]}
    tid = {t["name"]: t["id"] for t in manifest["tests"]}
    result: Dict[str, Any] = {"windows": {}}

    def copy(rel: Any, dest: str) -> Optional[str]:
        src = reg.abs(reg.run_dir, rel)
        if not src.exists():
            return None
        (out / dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / dest)
        return dest

    for key in ("main_catalog_map", "main_catalog_time"):
        if key in figs:
            d = copy(figs[key], f"figures/{Path(str(figs[key])).name}")
            if d:
                result[key] = d

    for wkey, w in wid.items():
        wf = figs.get(wkey) or {}
        entry: Dict[str, Any] = {"tests": {}, "forecasts": {}}
        for k in ("catalog_map", "catalog_time"):
            if k in wf:
                d = copy(wf[k], f"figures/{w}/{Path(str(wf[k])).name}")
                if d:
                    entry[k] = d
        for tname, t in tid.items():
            if tname in wf:
                d = copy(wf[tname], f"figures/{w}/{Path(str(wf[tname])).name}")
                if d:
                    entry["tests"][t] = d
            for mname, m in mid.items():
                k = f"{tname}_{mname}"
                if k in wf:
                    d = copy(wf[k], f"figures/{w}/{Path(str(wf[k])).name}")
                    if d:
                        entry["tests"][f"{t}:{m}"] = d
        for mname, m in mid.items():
            fk = (wf.get("forecasts") or {}).get(mname)
            if fk:
                d = copy(fk, f"figures/{w}/{Path(str(fk)).name}")
                if d:
                    entry["forecasts"][m] = d
        result["windows"][w] = entry
    return result


def export_static(out: Path) -> None:
    for src in STATIC.rglob("*"):
        if src.is_file():
            dest = out / src.relative_to(STATIC)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    logos = out / "logos"
    logos.mkdir(exist_ok=True)
    if ARTIFACTS.exists():
        for src in ARTIFACTS.glob("*.png"):
            shutil.copy2(src, logos / src.name)


def export_experiment(
    experiment: Any,
    out_dir: Optional[str] = None,
    forecasts: bool = True,
    results: bool = True,
    figures: bool = True,
) -> Path:
    """
    Exports an experiment as a self-contained static dashboard.

    Parameters
    ----------
    experiment : floatcsep.experiment.Experiment
        A staged experiment whose results exist on disk.
    out_dir : str, optional
        Output folder. Defaults to ``<run_dir>/dashboard``.
    forecasts, results, figures : bool
        Which parts to export besides the manifest and the catalog.

    Returns
    -------
    pathlib.Path
        The output folder.
    """
    reg = experiment.registry
    out = Path(out_dir) if out_dir else Path(reg.run_dir) / "dashboard"
    out.mkdir(parents=True, exist_ok=True)
    log.info(f"Exporting dashboard to {out}")

    manifest = build_manifest(experiment)
    roots = [Path.cwd(), Path(reg.workdir).resolve(), Path(__file__).resolve().parents[3]]
    exp = manifest["experiment"]
    exp["citation"] = citation(exp, find_cache("citation_cache.json", roots))
    exp["fair"] = fair_scores(exp.get("doi"), find_cache("fuji_cache.json", roots))
    manifest["grid"] = export_grid(getattr(experiment, "region", None), out)
    manifest["catalog"] = export_catalog(experiment, manifest, out)
    if forecasts:
        export_forecasts(experiment, manifest, out)
    manifest["results"] = export_results(experiment, manifest, out) if results else None
    manifest["figures"] = export_figures(experiment, manifest, out) if figures else None
    export_static(out)
    _dump(manifest, out / "manifest.json")
    log.info("Dashboard exported")
    return out
