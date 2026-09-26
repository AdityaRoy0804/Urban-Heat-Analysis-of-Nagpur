"""Pydantic request/response models for the FastAPI service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ZoneHistoryResponse(BaseModel):
    zone_id: str
    years: list[int]
    lst_celsius: list[float]
    ndvi: list[float]
    ndbi: list[float]
    hotspot_frequency: float = Field(..., description="Fraction of study years classified as a significant heat hotspot")
    is_persistent_hotspot: bool


class ScenarioRequest(BaseModel):
    zone_id: str
    scenario_land_cover_class: str = Field(
        ..., description="Target LULC class for the scenario, e.g. 'built_up' for 'a mall gets built here'"
    )
    k_analogs: int = Field(5, ge=1, le=20)


class FeatureContribution(BaseModel):
    feature: str
    contribution_celsius: float


class ScenarioResponse(BaseModel):
    zone_id: str
    baseline_lst_celsius: float
    scenario_lst_celsius: float
    estimated_shift_celsius: float
    uncertainty_band_celsius: tuple[float, float]
    explanation_method: str
    top_contributions: list[FeatureContribution]
    explanation_summary: str
    historical_hotspot_context: str
    analog_zone_ids: list[str]
    caveat_text: str


class ZoneSummary(BaseModel):
    zone_id: str
    row: int | None = None
    col: int | None = None
    latest_lst_celsius: float | None = None
    is_persistent_hotspot: bool = False
    latest_lulc_class: str | None = None


class ZonesListResponse(BaseModel):
    total_zones: int
    zones: list[ZoneSummary]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
