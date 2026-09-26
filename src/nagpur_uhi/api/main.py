"""
FastAPI backend for the Nagpur UHI platform.

Serves two things the interactive frontend needs:
  1. Per-zone multi-temporal history (LST/NDVI/NDBI trend + hotspot context)
  2. Scenario simulation (the explainable "what if" engine)

Raster tile serving (for the map's LST/NDVI/hotspot layer overlays) is
intentionally NOT reimplemented here - use `titiler` (https://developmentseed.org/titiler/)
pointed at the Cloud-Optimized GeoTIFFs produced by the processing scripts;
wiring a bespoke tile server would duplicate a well-maintained, purpose-built
tool. See scripts/05_run_api.sh for how the two are run side by side.

This module requires `fastapi`, `pydantic`, and `uvicorn` (see requirements.txt).
It is not exercised by the numpy/pandas-only unit tests in this environment;
tests/test_api.py skips itself automatically if fastapi is not importable, and
runs for real once you `pip install -r requirements.txt`.
"""

from __future__ import annotations

import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from nagpur_uhi.api.schemas import (
    FeatureContribution,
    HealthResponse,
    ScenarioRequest,
    ScenarioResponse,
    ZoneHistoryResponse,
    ZoneSummary,
    ZonesListResponse,
)
from nagpur_uhi.config import Config, load_config
from nagpur_uhi.ml.explain import ScenarioExplainer
from nagpur_uhi.ml.scenario import simulate_scenario


class AppState:
    """Holds the loaded config, zone dataset, and trained model/explainer."""

    def __init__(self) -> None:
        self.config: Config | None = None
        self.zones_df: pd.DataFrame | None = None
        self.model = None
        self.explainer: ScenarioExplainer | None = None

    def load(self) -> None:
        self.config = load_config()
        processed_dir = Path(self.config.paths.processed_dir)
        models_dir = Path(self.config.paths.models_dir)

        zones_path = processed_dir / "zones_dataset.csv"
        model_path = models_dir / "lst_model.pkl"

        if zones_path.exists():
            self.zones_df = pd.read_csv(zones_path)
        else:
            self.zones_df = pd.DataFrame()

        if model_path.exists():
            with open(model_path, "rb") as f:
                artifact = pickle.load(f)
            self.model = artifact["model"]
            self.explainer = ScenarioExplainer(
                model=self.model,
                feature_names=artifact["feature_columns"],
                background_data=artifact["background_data"],
            )

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.zones_df is not None and not self.zones_df.empty


state = AppState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.load()
    yield


app = FastAPI(
    title="Nagpur Urban Heat Island Platform API",
    description="Multi-temporal UHI analysis and explainable scenario modelling for Nagpur.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=state.model is not None)


@app.get("/zones", response_model=ZonesListResponse)
def list_zones() -> ZonesListResponse:
    """Return all available zones with their latest summary attributes."""
    if state.zones_df is None or state.zones_df.empty:
        return ZonesListResponse(total_zones=0, zones=[])

    latest_rows = state.zones_df.sort_values("year").groupby("zone_id").last().reset_index()
    zones = [
        ZoneSummary(
            zone_id=str(row["zone_id"]),
            row=int(row["row"]) if "row" in row and pd.notna(row["row"]) else None,
            col=int(row["col"]) if "col" in row and pd.notna(row["col"]) else None,
            latest_lst_celsius=round(float(row["target"]), 2) if "target" in row and pd.notna(row["target"]) else None,
            is_persistent_hotspot=bool(row.get("is_persistent_hotspot", False)),
            latest_lulc_class=str(row["lulc_class"]) if "lulc_class" in row and pd.notna(row["lulc_class"]) else None,
        )
        for _, row in latest_rows.iterrows()
    ]
    return ZonesListResponse(total_zones=len(zones), zones=zones)


@app.get("/zones/{zone_id}/history", response_model=ZoneHistoryResponse)
def zone_history(zone_id: str) -> ZoneHistoryResponse:
    if state.zones_df is None or state.zones_df.empty:
        raise HTTPException(status_code=503, detail="Zone dataset not loaded - run scripts/04_train_model.py first")

    zone_rows = state.zones_df[state.zones_df["zone_id"] == zone_id].sort_values("year")
    if zone_rows.empty:
        raise HTTPException(status_code=404, detail=f"Unknown zone_id '{zone_id}'")

    latest = zone_rows.iloc[-1]
    return ZoneHistoryResponse(
        zone_id=zone_id,
        years=zone_rows["year"].tolist(),
        lst_celsius=zone_rows["target"].tolist(),
        ndvi=zone_rows["ndvi"].tolist(),
        ndbi=zone_rows["ndbi"].tolist(),
        hotspot_frequency=float(latest.get("hotspot_frequency", 0.0)),
        is_persistent_hotspot=bool(latest.get("is_persistent_hotspot", False)),
    )


@app.post("/scenario/simulate", response_model=ScenarioResponse)
def scenario_simulate(request: ScenarioRequest) -> ScenarioResponse:
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded - run scripts/04_train_model.py first")

    zone_rows = state.zones_df[state.zones_df["zone_id"] == request.zone_id]
    if zone_rows.empty:
        raise HTTPException(status_code=404, detail=f"Unknown zone_id '{request.zone_id}'")

    baseline_row = zone_rows.sort_values("year").iloc[-1]
    feature_columns = state.config.scenario.feature_columns
    static_context_columns = ["elevation_m", "slope_deg"]

    try:
        result = simulate_scenario(
            zone_id=request.zone_id,
            baseline_features=baseline_row,
            scenario_land_cover_class=request.scenario_land_cover_class,
            candidate_zones=state.zones_df,
            model=state.model,
            explainer=state.explainer,
            feature_columns=feature_columns,
            static_context_columns=static_context_columns,
            hotspot_frequency=float(baseline_row.get("hotspot_frequency", 0.0)),
            k_analogs=request.k_analogs,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    top_contributions = [
        FeatureContribution(feature=name, contribution_celsius=value)
        for name, value in result.explanation.as_sorted_list()
    ]

    return ScenarioResponse(
        zone_id=result.zone_id,
        baseline_lst_celsius=result.baseline_lst_estimate,
        scenario_lst_celsius=result.scenario_lst_estimate,
        estimated_shift_celsius=result.estimated_shift_celsius,
        uncertainty_band_celsius=result.uncertainty_band_celsius,
        explanation_method=result.explanation.method,
        top_contributions=top_contributions,
        explanation_summary=result.explanation.summary_text(),
        historical_hotspot_context=result.historical_hotspot_context,
        analog_zone_ids=result.analog_zone_ids,
        caveat_text=result.caveat_text,
    )


# Serve interactive frontend if available
_frontend_dir = Path(__file__).resolve().parent.parent.parent.parent / "frontend"
if _frontend_dir.exists() and (_frontend_dir / "index.html").exists():
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(_frontend_dir / "index.html")

    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")
