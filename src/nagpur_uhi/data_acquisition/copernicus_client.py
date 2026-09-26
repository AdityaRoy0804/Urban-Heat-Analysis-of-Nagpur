"""
Copernicus Data Space Ecosystem (CDSE) client.

This is deliberately a *separate* data path from `gee_client.py`. It supplies
static, scenario-invariant terrain context - primarily the Copernicus DEM
(GLO-30) - directly to the explainable scenario model (see `ml/scenario.py`),
rather than being merged into the temporal feature-extraction stage.

Rationale (see project discussion): elevation, slope and aspect condition how
a given change in vegetation/built-up cover translates into a temperature
outcome at a *specific* location (drainage, airflow, natural shading). Feeding
this in as its own labeled branch keeps that distinction visible in the
model's explanation output (e.g. SHAP contribution attributed to "terrain
context" vs. "land-cover change").

Endpoints below are the documented CDSE OAuth2 + OData endpoints as of this
writing (https://documentation.dataspace.copernicus.eu/APIs). Verify the DEM
collection/product-type name in `config.yaml` against current CDSE docs
before running, since catalogue naming has changed over time.

`requests` is a lightweight, near-universal dependency, so it is imported at
module level (unlike the heavier `ee`/`geemap`/`rasterio` stack), but all
network calls are isolated in functions so the module still imports cleanly
offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests

from nagpur_uhi.config import Config


@dataclass
class CDSEToken:
    access_token: str
    expires_in: int


def get_access_token(cfg: Config) -> CDSEToken:
    """OAuth2 client-credentials flow against the CDSE identity provider."""
    client_id = os.environ.get(cfg.copernicus_dataspace.client_id_env)
    client_secret = Config.get_secret(cfg.copernicus_dataspace.client_secret_env)
    if not client_id:
        raise EnvironmentError(
            f"Set {cfg.copernicus_dataspace.client_id_env} in the environment (see .env.example)."
        )

    response = requests.post(
        cfg.copernicus_dataspace.token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return CDSEToken(access_token=payload["access_token"], expires_in=payload["expires_in"])


def search_dem_products(cfg: Config, token: CDSEToken) -> list[dict[str, Any]]:
    """
    Query the CDSE OData catalogue for Copernicus DEM tiles intersecting the AOI.
    Returns the raw product metadata list; download separately via `download_product`.
    """
    lon_min, lat_min, lon_max, lat_max = cfg.aoi.bbox
    wkt_polygon = (
        f"POLYGON(({lon_min} {lat_min}, {lon_max} {lat_min}, "
        f"{lon_max} {lat_max}, {lon_min} {lat_max}, {lon_min} {lat_min}))"
    )
    filter_str = (
        f"Collection/Name eq '{cfg.copernicus_dataspace.dem_collection_name}' "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{wkt_polygon}')"
    )
    resp = requests.get(
        f"{cfg.copernicus_dataspace.odata_base_url}/Products",
        params={"$filter": filter_str, "$top": 20},
        headers={"Authorization": f"Bearer {token.access_token}"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("value", [])


def download_product(cfg: Config, token: CDSEToken, product_id: str, out_path: str) -> str:
    """Stream-download a single CDSE product (e.g. a Copernicus DEM tile) to disk."""
    url = f"{cfg.copernicus_dataspace.odata_base_url}/Products({product_id})/$value"
    with requests.get(
        url, headers={"Authorization": f"Bearer {token.access_token}"}, stream=True, timeout=300
    ) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return out_path


def get_terrain_context(cfg: Config, dem_path: str) -> dict[str, Any]:
    """
    Derive elevation, slope and aspect rasters from a downloaded Copernicus DEM
    tile. Requires `rasterio` (heavy dependency, lazily imported).

    Returns a dict of numpy arrays: {"elevation_m", "slope_deg", "aspect_deg"}.
    """
    try:
        import numpy as np
        import rasterio
    except ImportError as e:
        raise ImportError("rasterio is required for terrain derivation: pip install rasterio") from e

    with rasterio.open(dem_path) as src:
        elevation = src.read(1).astype("float64")
        pixel_size_x, pixel_size_y = src.res

    dz_dy, dz_dx = np.gradient(elevation, pixel_size_y, pixel_size_x)
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    aspect_deg = (np.degrees(np.arctan2(-dz_dx, dz_dy)) + 360.0) % 360.0

    return {"elevation_m": elevation, "slope_deg": slope_deg, "aspect_deg": aspect_deg}
