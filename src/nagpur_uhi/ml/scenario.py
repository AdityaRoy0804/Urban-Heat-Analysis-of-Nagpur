"""
Scenario simulation engine - the "what if a mall is built in Hingna Gramin"
workflow.

Design principle (per project scope): don't let a user drag an abstract
"+30% built-up" slider with no physical grounding. Instead, ground the
scenario in a real analog - an existing zone elsewhere in the study area
that already underwent a similar land-cover transition - and use *its*
observed feature signature as the post-scenario input, rather than an
arbitrary hand-picked number. This turns "guess a number" into "here's what
happened when this kind of change happened elsewhere in this city."

The output (`ScenarioResult`) is a structured explanation, not a bare number:
estimated shift, SHAP/occlusion breakdown, the analog zones used, the
target zone's own historical hotspot context, and an uncertainty band derived
from spread across analogs - all surfaced together because a single point
estimate without this context would overstate the model's confidence for
what is, by design, a scenario estimate rather than a prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from nagpur_uhi.ml.explain import ExplanationResult, ScenarioExplainer


@dataclass
class ScenarioResult:
    zone_id: str
    baseline_lst_estimate: float
    scenario_lst_estimate: float
    estimated_shift_celsius: float
    uncertainty_band_celsius: tuple[float, float]
    explanation: ExplanationResult
    historical_hotspot_context: str
    analog_zone_ids: list[str]
    caveat_text: str = field(
        default=(
            "This is a scenario-based estimate grounded in historical spatial "
            "patterns, not a prediction of future climate. It reflects how LST "
            "has responded to similar land-cover changes elsewhere in the study "
            "area, adjusted for this location's own terrain context."
        )
    )


def find_analog_zones(
    target_land_cover_class: str,
    scenario_land_cover_class: str,
    candidate_zones: pd.DataFrame,
    feature_columns: list[str],
    reference_features: pd.Series,
    k: int = 5,
) -> pd.DataFrame:
    """
    Find the `k` nearest real zones in `candidate_zones` that (a) already have
    LULC class `scenario_land_cover_class` (i.e. they represent the "after"
    state we're simulating) and (b) are the closest analogs to
    `reference_features` in the feature space provided - so the analog is
    both the right *kind* of change and the right *scale/context* of change.

    `candidate_zones` must contain `feature_columns` plus a 'lulc_class' and
    a 'zone_id' column. Raises ValueError if fewer than `k` candidates exist.
    """
    pool = candidate_zones[candidate_zones["lulc_class"] == scenario_land_cover_class]
    if len(pool) < k:
        raise ValueError(
            f"Only {len(pool)} candidate zones found with lulc_class='{scenario_land_cover_class}'; "
            f"need at least k={k}. Lower k or expand the candidate zone dataset."
        )

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(pool[feature_columns].to_numpy())
    _distances, indices = nn.kneighbors(reference_features[feature_columns].to_numpy().reshape(1, -1))
    return pool.iloc[indices[0]].reset_index(drop=True)


def simulate_scenario(
    zone_id: str,
    baseline_features: pd.Series,
    scenario_land_cover_class: str,
    candidate_zones: pd.DataFrame,
    model,
    explainer: ScenarioExplainer,
    feature_columns: list[str],
    static_context_columns: list[str],
    hotspot_frequency: float,
    k_analogs: int = 5,
) -> ScenarioResult:
    """
    Run one full scenario simulation for a target zone.

    baseline_features: the zone's current (observed) feature row.
    scenario_land_cover_class: the LULC class the scenario represents
        transitioning *to* (e.g. "built_up" for "a mall gets built here").
    candidate_zones: dataframe of other real zones with features + lulc_class,
        used as the analog pool (see find_analog_zones).
    static_context_columns: feature columns that do NOT change under the
        scenario (e.g. elevation_m, slope_deg from the Copernicus DEM branch) -
        these are held fixed at the target zone's own values in the analog match.
    hotspot_frequency: this zone's fraction-of-years classified as a hotspot
        (from analysis.hotspots.persistent_hotspots), surfaced as context.
    """
    analogs = find_analog_zones(
        target_land_cover_class=baseline_features.get("lulc_class", "unknown"),
        scenario_land_cover_class=scenario_land_cover_class,
        candidate_zones=candidate_zones,
        feature_columns=feature_columns,
        reference_features=baseline_features,
        k=k_analogs,
    )

    # Build the "after" feature row: changeable features come from the analog
    # average; static terrain context is held at this zone's own value.
    scenario_features = baseline_features.copy()
    changeable_columns = [c for c in feature_columns if c not in static_context_columns]
    scenario_features[changeable_columns] = analogs[changeable_columns].mean()

    baseline_pred = float(model.predict(baseline_features[feature_columns].to_numpy().reshape(1, -1))[0])
    scenario_pred = float(model.predict(scenario_features[feature_columns].to_numpy().reshape(1, -1))[0])

    # Uncertainty band: spread of LST actually observed across the analog zones
    # themselves, rather than a statistical CI on the regression alone - this
    # reflects real-world variability in "what happens when this change occurs".
    analog_lst_values = analogs["target"].to_numpy() if "target" in analogs.columns else np.array([scenario_pred])
    lower, upper = float(np.percentile(analog_lst_values, 10)), float(np.percentile(analog_lst_values, 90))

    explanation = explainer.explain_instance(scenario_features)

    hotspot_context = (
        f"This zone was classified as a statistically significant heat hotspot in "
        f"{hotspot_frequency:.0%} of the study years analysed."
        if hotspot_frequency > 0
        else "This zone was not classified as a significant heat hotspot in the study years analysed."
    )

    return ScenarioResult(
        zone_id=zone_id,
        baseline_lst_estimate=baseline_pred,
        scenario_lst_estimate=scenario_pred,
        estimated_shift_celsius=scenario_pred - baseline_pred,
        uncertainty_band_celsius=(lower, upper),
        explanation=explanation,
        historical_hotspot_context=hotspot_context,
        analog_zone_ids=analogs["zone_id"].tolist() if "zone_id" in analogs.columns else [],
    )
