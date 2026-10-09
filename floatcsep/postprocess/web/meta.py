"""
Citation and FAIR metadata for the dashboard.

Reads the caches written by ``fetch_citations.py`` (``citation_cache.json``) and
``fetch_fuji.py`` (``fuji_cache.json``) when they exist, and falls back to a citation built
from the experiment fields otherwise. Both caches are keyed by ``https://doi.org/<doi>``.
"""

import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

PUBLISHER = "Collaboratory for the Study of Earthquake Predictability (CSEP)"


def _doi_url(doi: Optional[str]) -> str:
    doi = (doi or "").strip()
    if not doi:
        return ""
    return doi if doi.startswith("http") else f"https://doi.org/{doi}"


def find_cache(name: str, roots: List[Path]) -> Dict[str, Any]:
    for root in roots:
        for d in [root, *root.parents][:4]:
            p = d / name
            if p.exists():
                try:
                    return json.loads(p.read_text())
                except Exception:
                    return {}
    return {}


def citation(exp: Dict[str, Any], cache: Dict[str, Any]) -> Dict[str, str]:
    """APA, BibTeX and RIS strings for the experiment."""
    url = _doi_url(exp.get("doi"))
    hit = cache.get(url) if url else None
    if hit and any(hit.get(k) for k in ("apa", "bibtex", "ris")):
        return {k: hit.get(k, "") for k in ("apa", "bibtex", "ris")} | {"source": "DataCite"}

    name = exp.get("name") or "Experiment"
    authors = exp.get("authors") or "CSEP Working Group"
    m = re.search(r"(\d{4})", str(exp.get("last_run") or ""))
    year = m.group(1) if m else str(datetime.date.today().year)
    doi = (exp.get("doi") or "").strip()

    apa = f"{authors} ({year}). {name}. CSEP." + (f" {url}" if url else "")
    slug = (re.sub(r"[^A-Za-z0-9]+", "", name)[:20] or "experiment") + year
    bib = [f"  author = {{{authors}}}", f"  title = {{{name}}}", f"  year = {{{year}}}", f"  publisher = {{{PUBLISHER}}}"]
    if doi:
        bib += [f"  doi = {{{doi}}}", f"  url = {{{url}}}"]
    bibtex = f"@misc{{{slug},\n" + ",\n".join(bib) + "\n}"
    ris = ["TY  - DATA", f"TI  - {name}", f"PY  - {year}", f"PB  - {PUBLISHER}"]
    ris += [f"AU  - {a.strip()}" for a in authors.split(",") if a.strip()]
    if doi:
        ris += [f"DO  - {doi}", f"UR  - {url}"]
    ris.append("ER  - ")
    return {"apa": apa, "bibtex": bibtex, "ris": "\n".join(ris), "source": "experiment fields"}


def fair_scores(doi: Optional[str], cache: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """F, A, I, R and FAIR percentages from an F-UJI assessment, if cached."""
    url = _doi_url(doi)
    res = cache.get(url) if url else None
    if not isinstance(res, dict):
        return None
    pct = (res.get("summary") or {}).get("score_percent")
    if not isinstance(pct, dict):
        return None
    out = {k: float(pct[k]) for k in ("F", "A", "I", "R", "FAIR") if pct.get(k) is not None}
    return out or None
