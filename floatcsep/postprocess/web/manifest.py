"""
JSON manifest of an experiment: the single contract between floatCSEP and the static
dashboard. Everything the pages show is either in this file or in a file it points to.
"""

import datetime
import re
from typing import Any, Dict, List, Optional

from floatcsep import __version__ as fc_version
from floatcsep.utils.helpers import timewindow2str


def _str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def _iso(value: Any) -> Optional[str]:
    """ISO 8601 string in UTC. Naive datetimes are UTC in floatCSEP."""
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.timezone.utc)
        return value.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    return _str(value)


def _duration(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*-?\s*([a-zA-Z]+?)s?\s*", value)
        if not m:
            return value
        n, unit = m.groups()
        return f"{n} {unit}{'' if n in ('1', '1.0') else 's'}"
    days = getattr(value, "days", None)
    secs = getattr(value, "seconds", 0)
    if days is None:
        return str(value)
    if days and not secs:
        if days % 365 == 0:
            n = days // 365
            return f"{n} year{'s' if n != 1 else ''}"
        return f"{days} day{'s' if days != 1 else ''}"
    if not days and secs:
        h = secs // 3600
        return f"{h} hour{'s' if h != 1 else ''}" if h else f"{secs} s"
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _refs(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def window_entries(experiment: Any) -> List[Dict[str, Any]]:
    out = []
    for i, tw in enumerate(experiment.time_windows):
        out.append(
            {
                "id": f"w{i}",
                "key": timewindow2str(tw),
                "label": timewindow2str(tw).replace("_", " – "),
                "start": _iso(tw[0]),
                "end": _iso(tw[1]),
            }
        )
    return out


def build_manifest(experiment: Any) -> Dict[str, Any]:
    """
    Serializable description of an experiment for the dashboard.

    Parameters
    ----------
    experiment : floatcsep.experiment.Experiment
        A staged experiment (``stage_models`` and ``set_tree`` already called).

    Returns
    -------
    dict
        Keys ``experiment``, ``region``, ``magnitudes``, ``time_windows``, ``models``,
        ``tests``. File references (``catalog``, ``grid``, ``results``, ``figures``) are
        added by the exporter.
    """
    tcfg = experiment.time_config or {}
    rcfg = experiment.region_config or {}
    region = getattr(experiment, "region", None)
    exp_class = tcfg.get("exp_class", "ti")

    info = {
        "name": experiment.name,
        "authors": _str(experiment.authors),
        "doi": experiment.doi,
        "catalog_doi": experiment.catalog_doi,
        "license": experiment.LICENSE,
        "journal": experiment.journal,
        "manuscript_doi": experiment.manuscript_doi,
        "floatcsep_version": experiment.floatcsep_version or fc_version,
        "pycsep_version": experiment.pycsep_version,
        "last_run": _str(experiment.last_run),
        "exp_time": _str(experiment.exp_time),
        "exp_class": "Time-Dependent" if exp_class in ("td", "time-dependent") else "Time-Independent",
        "start_date": _iso(experiment.start_date),
        "end_date": _iso(experiment.end_date),
        "horizon": _duration(tcfg.get("horizon")),
        "offset": _duration(tcfg.get("offset")),
        "growth": tcfg.get("growth"),
        "n_intervals": len(experiment.time_windows),
        "run_mode": experiment.run_mode,
        "run_dir": _str(experiment.run_dir),
        "config_file": _str(experiment.config_file),
        "model_config": _str(experiment.model_config),
        "test_config": _str(experiment.test_config),
    }

    reg = None
    if region is not None:
        x0, x1, y0, y1 = region.get_bbox()
        reg = {
            "name": getattr(region, "name", None) or rcfg.get("region"),
            "dh": float(region.dh),
            "n_cells": int(len(region.origins())),
            "bbox": [float(x0), float(y0), float(x1), float(y1)],
        }
    reg = reg or {}
    reg.update(
        {
            "mag_min": rcfg.get("mag_min"),
            "mag_max": rcfg.get("mag_max"),
            "mag_bin": rcfg.get("mag_bin"),
            "depth_min": rcfg.get("depth_min"),
            "depth_max": rcfg.get("depth_max"),
        }
    )

    models = []
    for i, m in enumerate(experiment.models):
        models.append(
            {
                "id": f"m{i}",
                "name": m.name,
                "authors": _str(getattr(m, "authors", None)),
                "doi": getattr(m, "doi", None),
                "giturl": getattr(m, "giturl", None),
                "git_hash": getattr(m, "repo_hash", None),
                "zenodo_id": getattr(m, "zenodo_id", None),
                "path": _str(m.registry.rel(m.registry.path)),
                "func": _str(getattr(m, "func", None)),
                "func_kwargs": _jsonable(getattr(m, "func_kwargs", None)),
                "fmt": getattr(m, "fmt", None),
                "forecast_unit": getattr(m, "forecast_unit", None),
                "forecast_class": m.repository.__class__.__name__,
                "forecasts": {},
            }
        )

    tests = []
    for i, t in enumerate(experiment.tests):
        func = getattr(t, "func", None)
        tests.append(
            {
                "id": f"t{i}",
                "name": t.name,
                "func": f"{func.__module__}.{func.__name__}" if func else None,
                "type": getattr(t, "type", None),
                "func_kwargs": _jsonable(getattr(t, "func_kwargs", None)),
                "ref_model": _str(getattr(t, "ref_model", None)),
                "ref_models": _refs(getattr(t, "ref_model", None)),
                "plot_func": [f"{p.__module__}.{p.__name__}" for p in (t.plot_func or [])],
            }
        )

    return {
        "schema": 1,
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "experiment": info,
        "region": reg,
        "magnitudes": [float(m) for m in experiment.magnitudes],
        "time_windows": window_entries(experiment),
        "models": models,
        "tests": tests,
    }
