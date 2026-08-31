"""Pydantic schemas for floatCSEP Next.js dashboard."""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel, ConfigDict, field_validator, Field


def serialize_value_recursive(value: Any) -> Any:
    """Recursively convert values to JSON-serializable types."""
    if isinstance(value, Path):
        return str(value)
    elif isinstance(value, dict):
        return {k: serialize_value_recursive(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [serialize_value_recursive(v) for v in value]
    elif hasattr(value, "__dict__"):
        return serialize_value_recursive(vars(value))
    else:
        return value


class ManifestModel(BaseModel):
    """
    Pydantic model for the Experiment Manifest.
    Validates and serializes the Manifest dataclass from floatcsep.
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    # --- Existing fields ---
    name: str
    start_date: str
    end_date: str
    authors: Optional[str] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    manuscript_doi: Optional[str] = None
    exp_time: Optional[str] = None
    floatcsep_version: Optional[str] = None
    pycsep_version: Optional[str] = None
    last_run: Optional[str] = None
    catalog_doi: Optional[str] = None
    license: Optional[str] = None
    date_range: str
    magnitudes: List[float]
    
    # Region is typically an object in the dataclass
    region: Optional[Dict[str, Any]] = None

    models: List[Dict[str, Any]]
    tests: List[Dict[str, Any]]
    time_windows: List[str]

    catalog: Dict[str, Any]
    results_main: Dict[str, str]  # Key will be converted to string pipe-delimited
    results_model: Dict[str, str]

    app_root: Optional[str] = None

    # --- Metadata fields ---
    exp_class: str
    n_intervals: int
    horizon: Optional[str] = None
    offset: Optional[str] = None
    growth: Optional[str] = None

    mag_min: Optional[float] = None
    mag_max: Optional[float] = None
    mag_bin: Optional[float] = None
    depth_min: Optional[float] = None
    depth_max: Optional[float] = None

    run_mode: Optional[str] = None
    run_dir: Optional[str] = None
    config_file: Optional[str] = None
    # Rename to avoid conflict with Pydantic's model_config
    model_config_path: Optional[str] = Field(None, validation_alias="model_config", serialization_alias="model_config")
    test_config: Optional[str] = None

    @field_validator("region", mode="before")
    def serialize_region(cls, v: Any) -> Optional[Dict[str, Any]]:
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        # Attempt to extract attributes from Region object
        return {
            "name": getattr(v, "name", None),
            "bbox": list(v.get_bbox()) if hasattr(v, "get_bbox") else None,
            "dh": float(v.dh) if hasattr(v, "dh") else None,
            "origins": v.origins().tolist() if hasattr(v, "origins") else None,
        }

    @field_validator("models", "tests", "catalog", mode="before")
    def serialize_generic_structures(cls, v: Any) -> Any:
        return serialize_value_recursive(v)
    
    @field_validator("results_main", mode="before")
    def serialize_results_main(cls, v: Any) -> Dict[str, str]:
        # transform Dict[Tuple[str, str], str] -> Dict[str, str]
        if isinstance(v, dict):
            new_dict = {}
            for key, val in v.items():
                if isinstance(key, tuple):
                    new_key = f"{key[0]}|{key[1]}"
                else:
                    new_key = str(key)
                new_dict[new_key] = serialize_value_recursive(val)
            return new_dict
        return v

    @field_validator("results_model", mode="before")
    def serialize_results_model(cls, v: Any) -> Dict[str, str]:
        # transform Dict[Tuple[str, str, str], str] -> Dict[str, str]
        if isinstance(v, dict):
            new_dict = {}
            for key, val in v.items():
                if isinstance(key, tuple):
                    new_key = f"{key[0]}|{key[1]}|{key[2]}"
                else:
                    new_key = str(key)
                new_dict[new_key] = serialize_value_recursive(val)
            return new_dict
        return v

    @field_validator("app_root", "run_dir", "config_file", "model_config_path", "test_config", mode="before")
    def serialize_paths(cls, v: Any) -> Optional[str]:
        if isinstance(v, Path):
            return str(v)
        return v

