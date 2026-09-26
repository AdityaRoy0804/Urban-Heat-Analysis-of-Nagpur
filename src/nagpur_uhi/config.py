"""
Central configuration loader.

Reads config/config.yaml into typed, dot-accessible dataclasses so every module
gets validated, IDE-completable config instead of passing raw dicts around.
Secrets (API credentials) are never stored in the YAML - only environment
variable *names* are stored there, and the actual values are read from the
environment at call time (see `Config.get_secret`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@dataclass
class StudyYear:
    year: int
    start: str
    end: str


@dataclass
class AOIConfig:
    name: str
    bbox: list[float]
    boundary_geojson: str | None = None


@dataclass
class CRSConfig:
    processing_epsg: int
    storage_epsg: int


@dataclass
class GEEConfig:
    project_id: str
    landsat_collections: dict[str, str]
    sentinel2_collection: str
    sentinel1_collection: str
    max_cloud_cover_pct: int


@dataclass
class CopernicusConfig:
    token_url: str
    odata_base_url: str
    process_api_base_url: str
    dem_collection_name: str
    client_id_env: str
    client_secret_env: str


@dataclass
class PathsConfig:
    raw_dir: str
    processed_dir: str
    models_dir: str

    def resolve(self, root: Path = REPO_ROOT) -> "PathsConfig":
        return PathsConfig(
            raw_dir=str(root / self.raw_dir),
            processed_dir=str(root / self.processed_dir),
            models_dir=str(root / self.models_dir),
        )


@dataclass
class LULCConfig:
    classes: list[str]
    ndvi_vegetation_min: float
    ndwi_water_min: float
    ndbi_built_up_min: float


@dataclass
class HotspotConfig:
    window_size: int
    confidence_levels: list[float]
    persistent_threshold: float

    def __post_init__(self) -> None:
        if self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError("hotspot.window_size must be an odd integer >= 3")


@dataclass
class ModelConfig:
    type: str
    n_estimators: int
    max_depth: int
    min_samples_leaf: int
    random_state: int
    test_size: float
    spatial_block_size_m: float


@dataclass
class ScenarioConfig:
    n_analog_zones: int
    feature_columns: list[str]


@dataclass
class Config:
    aoi: AOIConfig
    crs: CRSConfig
    study_years: list[StudyYear]
    gee: GEEConfig
    copernicus_dataspace: CopernicusConfig
    paths: PathsConfig
    lulc: LULCConfig
    hotspot: HotspotConfig
    model: ModelConfig
    scenario: ScenarioConfig

    @staticmethod
    def get_secret(env_var_name: str) -> str:
        """Read a secret from the environment, raising a clear error if missing."""
        value = os.environ.get(env_var_name)
        if not value:
            raise EnvironmentError(
                f"Required environment variable '{env_var_name}' is not set. "
                "Copy .env.example to .env and fill in credentials, or export it directly."
            )
        return value


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate config/config.yaml into a `Config` object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    try:
        gee_project = os.environ.get("GEE_PROJECT_ID", raw["gee"]["project_id"])
        cfg = Config(
            aoi=AOIConfig(**raw["aoi"]),
            crs=CRSConfig(**raw["crs"]),
            study_years=[StudyYear(**sy) for sy in raw["study_years"]],
            gee=GEEConfig(
                project_id=gee_project,
                landsat_collections=raw["gee"]["landsat_collections"],
                sentinel2_collection=raw["gee"]["sentinel2_collection"],
                sentinel1_collection=raw["gee"]["sentinel1_collection"],
                max_cloud_cover_pct=raw["gee"]["max_cloud_cover_pct"],
            ),
            copernicus_dataspace=CopernicusConfig(**raw["copernicus_dataspace"]),
            paths=PathsConfig(**raw["paths"]).resolve(),
            lulc=LULCConfig(**raw["lulc"]),
            hotspot=HotspotConfig(**raw["hotspot"]),
            model=ModelConfig(**raw["model"]),
            scenario=ScenarioConfig(**raw["scenario"]),
        )
    except KeyError as e:
        raise KeyError(f"Missing required config key: {e}") from e

    if len(cfg.study_years) < 3:
        raise ValueError(
            "The problem statement requires a multi-temporal comparison across "
            "approximately 3-5+ years; config.study_years has fewer than 3 entries."
        )

    return cfg
