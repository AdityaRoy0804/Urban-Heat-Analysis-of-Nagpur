"""
Google Earth Engine data acquisition.

Pulls the multi-temporal signals: Landsat (thermal + optical),
Sentinel-2 (optical indices), and Sentinel-1 (SAR).

This module requires network access and Earth Engine credentials, so `ee`
and `geemap` are imported lazily inside functions rather than at module load
time - this lets the rest of the package (and its tests) import cleanly in
environments without GEE access configured.

Reference product used for LST: Landsat Collection 2 Level-2 `ST_B10`,
which is already atmospherically and emissivity corrected by USGS.
"""

from __future__ import annotations

from typing import Any

from nagpur_uhi.config import Config, StudyYear


def _lazy_import_ee():
    try:
        import ee  # type: ignore
    except ImportError as e:
        raise ImportError(
            "earthengine-api is required for GEE data acquisition. "
            "Install with: pip install earthengine-api geemap"
        ) from e
    return ee


def initialize(project_id: str) -> None:
    """Authenticate and initialize the Earth Engine session for this project."""
    ee = _lazy_import_ee()

    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def _aoi_geometry(cfg: Config):
    """Load the configured AOI geometry."""
    ee = _lazy_import_ee()

    if cfg.aoi.boundary_geojson:
        import json

        with open(cfg.aoi.boundary_geojson, "r", encoding="utf-8") as f:
            geojson = json.load(f)

        return ee.Geometry(geojson["features"][0]["geometry"])

    return ee.Geometry.Rectangle(cfg.aoi.bbox)


# ---------------------------------------------------------------------------
# LANDSAT
# ---------------------------------------------------------------------------

def _mask_landsat_c2_clouds(image: "Any"):
    """Cloud/shadow mask using the Landsat Collection 2 QA_PIXEL band."""
    ee = _lazy_import_ee()

    qa = image.select("QA_PIXEL")

    cloud_bit = 3
    shadow_bit = 4
    cirrus_bit = 2

    mask = (
        qa.bitwiseAnd(1 << cloud_bit).eq(0)
        .And(qa.bitwiseAnd(1 << shadow_bit).eq(0))
        .And(qa.bitwiseAnd(1 << cirrus_bit).eq(0))
    )

    return image.updateMask(mask)


def _scale_landsat_c2(image: "Any"):
    """Apply USGS Collection-2 scale/offset to optical and thermal bands."""

    optical = (
        image
        .select("SR_B.")
        .multiply(0.0000275)
        .add(-0.2)
    )

    thermal = (
        image
        .select("ST_B10")
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
        .rename("LST_Celsius")
    )

    return (
        image
        .addBands(optical, None, True)
        .addBands(thermal, None, True)
    )


def get_landsat_composite(cfg: Config, study_year: StudyYear):
    """
    Median, cloud-masked Landsat 8/9 Collection-2 Level-2 composite
    for one study-year window.

    Provides:
        - Landsat optical surface reflectance bands
        - LST in Celsius
    """

    ee = _lazy_import_ee()
    aoi = _aoi_geometry(cfg)

    collections = []

    for coll_id in [
        cfg.gee.landsat_collections["l8_l9_sr"],
        "LANDSAT/LC09/C02/T1_L2",
    ]:

        coll = (
            ee.ImageCollection(coll_id)
            .filterBounds(aoi)
            .filterDate(study_year.start, study_year.end)
            .filter(
                ee.Filter.lt(
                    "CLOUD_COVER",
                    cfg.gee.max_cloud_cover_pct,
                )
            )
            .map(_mask_landsat_c2_clouds)
            .map(_scale_landsat_c2)
        )

        collections.append(coll)

    merged = collections[0].merge(collections[1])

    composite = (
        merged
        .median()
        .clip(aoi)
    )

    return composite.set("year", study_year.year)


# ---------------------------------------------------------------------------
# SENTINEL-2
# ---------------------------------------------------------------------------

def _mask_s2_clouds(image: "Any"):
    """Cloud mask using the Sentinel-2 Scene Classification Layer (SCL)."""

    scl = image.select("SCL")

    # Exclude:
    # 3  = cloud shadow
    # 8  = medium probability cloud
    # 9  = high probability cloud
    # 10 = thin cirrus
    bad = (
        scl.eq(3)
        .Or(scl.eq(8))
        .Or(scl.eq(9))
        .Or(scl.eq(10))
    )

    return (
        image
        .updateMask(bad.Not())
        .divide(10000)
        .copyProperties(image, image.propertyNames())
    )


def get_sentinel2_composite(cfg: Config, study_year: StudyYear):
    """
    Median, cloud-masked Sentinel-2 L2A surface-reflectance composite.
    """

    ee = _lazy_import_ee()
    aoi = _aoi_geometry(cfg)

    coll = (
        ee.ImageCollection(cfg.gee.sentinel2_collection)
        .filterBounds(aoi)
        .filterDate(
            study_year.start,
            study_year.end,
        )
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                cfg.gee.max_cloud_cover_pct,
            )
        )
        .map(_mask_s2_clouds)
    )

    return (
        coll
        .median()
        .clip(aoi)
        .set("year", study_year.year)
    )


# ---------------------------------------------------------------------------
# SENTINEL-1
# ---------------------------------------------------------------------------

def get_sentinel1_composite(cfg: Config, study_year: StudyYear):
    """
    Median Sentinel-1 GRD VV/VH composite.

    Uses both ascending and descending passes rather than restricting
    observations to a single orbit direction.

    Requires:
        - IW acquisition mode
        - VV polarization
        - VH polarization

    Produces:
        - VV
        - VH
        - sar_vv_vh_ratio
    """

    ee = _lazy_import_ee()
    aoi = _aoi_geometry(cfg)

    coll = (
        ee.ImageCollection(cfg.gee.sentinel1_collection)
        .filterBounds(aoi)
        .filterDate(
            study_year.start,
            study_year.end,
        )
        .filter(
            ee.Filter.eq(
                "instrumentMode",
                "IW",
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VV",
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VH",
            )
        )
        .select(
            ["VV", "VH"]
        )
        .map(
            lambda img: img.focal_median(
                30,
                "circle",
                "meters",
            )
        )
    )

    composite = (
        coll
        .median()
        .clip(aoi)
    )

    vv_vh_ratio = (
        composite
        .select("VV")
        .subtract(
            composite.select("VH")
        )
        .rename("sar_vv_vh_ratio")
    )

    return (
        composite
        .addBands(vv_vh_ratio)
        .set("year", study_year.year)
    )


# ---------------------------------------------------------------------------
# DEM / TERRAIN
# ---------------------------------------------------------------------------

def get_terrain_context_composite(cfg: Config):
    """
    Copernicus DEM GLO-30 terrain context via Earth Engine.

    Returns:
        elevation_m
        slope_deg
        aspect_deg
    """

    ee = _lazy_import_ee()
    aoi = _aoi_geometry(cfg)

    # Updated Copernicus DEM collection.
    dem = (
        ee.ImageCollection(
            "COPERNICUS/DEM/GLO30_2024_1"
        )
        .filterBounds(aoi)
        .mosaic()
        .select("DEM")
        .rename("elevation_m")
    )

    slope = (
        ee.Terrain
        .slope(dem)
        .rename("slope_deg")
    )

    aspect = (
        ee.Terrain
        .aspect(dem)
        .rename("aspect_deg")
    )

    return (
        dem
        .addBands(slope)
        .addBands(aspect)
        .clip(aoi)
    )


# ---------------------------------------------------------------------------
# LOCAL EXPORT
# ---------------------------------------------------------------------------

def export_composite_local(
    image,
    out_path: str,
    cfg: Config,
    scale: int = 30,
    bands: list[str] | None = None,
):
    """Export an ee.Image to a local GeoTIFF via geemap."""

    try:
        import geemap  # type: ignore
    except ImportError as e:
        raise ImportError(
            "geemap is required for local export: "
            "pip install geemap"
        ) from e

    aoi = _aoi_geometry(cfg)

    if bands:
        image = image.select(bands)

    geemap.ee_export_image(
        image,
        filename=out_path,
        scale=scale,
        region=aoi,
        file_per_band=False,
    )

    return out_path