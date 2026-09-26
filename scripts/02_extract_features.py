#!/usr/bin/env python3
"""
Stage 2 - Preprocessing + feature extraction.

For each study year: reproject/align the Landsat, Sentinel-2 and Sentinel-1
rasters to a common grid (config.crs.processing_epsg), validate alignment,
then compute NDVI, NDBI, NDWI, the SAR VV/VH texture ratio, and a rule-based
LULC map. Terrain context (elevation/slope) is derived once from the
Copernicus DEM and reused across all years, since terrain doesn't change
between study years.

Requires `rasterio` (see requirements.txt). Run after scripts/01_acquire_data.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from nagpur_uhi.config import load_config
from nagpur_uhi.data_acquisition.copernicus_client import get_terrain_context
from nagpur_uhi.features import indices, lulc
from nagpur_uhi.preprocessing import raster_ops
from nagpur_uhi.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def process_one_year(cfg, year: int, raw_dir: Path, processed_dir: Path) -> None:
    logger.info("Processing features for %s", year)
    epsg = cfg.crs.processing_epsg

    aligned = {}
    for source in ["landsat", "sentinel2", "sentinel1"]:
        src_path = raw_dir / f"{source}_{year}.tif"
        aligned_path = processed_dir / f"{source}_{year}_aligned.tif"
        raster_ops.reproject_and_resample(str(src_path), str(aligned_path), dst_crs_epsg=epsg, dst_resolution_m=30.0)
        aligned[source] = aligned_path

    landsat_arr, landsat_transform, landsat_crs = raster_ops.read_as_array(str(aligned["landsat"]), band=4)  # LST
    red_arr, _, _ = raster_ops.read_as_array(str(aligned["landsat"]), band=1)
    nir_arr, _, _ = raster_ops.read_as_array(str(aligned["landsat"]), band=2)
    swir_arr, _, _ = raster_ops.read_as_array(str(aligned["landsat"]), band=3)

    green_arr, s2_transform, s2_crs = raster_ops.read_as_array(str(aligned["sentinel2"]), band=1)
    s2_nir_arr, _, _ = raster_ops.read_as_array(str(aligned["sentinel2"]), band=3)

    vv_arr, s1_transform, s1_crs = raster_ops.read_as_array(str(aligned["sentinel1"]), band=1)
    vh_arr, _, _ = raster_ops.read_as_array(str(aligned["sentinel1"]), band=2)

    raster_ops.validate_alignment(
        {
            "landsat": (landsat_arr, landsat_transform, landsat_crs),
            "sentinel2": (green_arr, s2_transform, s2_crs),
            "sentinel1": (vv_arr, s1_transform, s1_crs),
        }
    )

    ndvi_arr = indices.ndvi(nir_arr, red_arr)
    ndbi_arr = indices.ndbi(swir_arr, nir_arr)
    ndwi_arr = indices.ndwi(green_arr, s2_nir_arr)
    sar_ratio_arr = indices.sar_vv_vh_ratio_db(vv_arr, vh_arr)

    lulc_arr = lulc.rule_based_lulc(
        ndvi_arr,
        ndbi_arr,
        ndwi_arr,
        ndvi_vegetation_min=cfg.lulc.ndvi_vegetation_min,
        ndwi_water_min=cfg.lulc.ndwi_water_min,
        ndbi_built_up_min=cfg.lulc.ndbi_built_up_min,
    )

    out_path = processed_dir / f"features_{year}.npz"
    np.savez_compressed(
        out_path,
        lst=landsat_arr,
        ndvi=ndvi_arr,
        ndbi=ndbi_arr,
        ndwi=ndwi_arr,
        sar_vv_vh_ratio=sar_ratio_arr,
        lulc_class=lulc_arr,
    )
    logger.info("Wrote %s", out_path)


def process_terrain_context(cfg, raw_dir: Path, processed_dir: Path) -> None:
    gee_dem_path = raw_dir / "dem_gee.tif"
    if gee_dem_path.exists():
        # GEE path: elevation_m, slope_deg, aspect_deg already computed server-side
        # (see gee_client.get_terrain_context_composite) - just read the 3 bands.
        elevation, _, _ = raster_ops.read_as_array(str(gee_dem_path), band=1)
        slope, _, _ = raster_ops.read_as_array(str(gee_dem_path), band=2)
        aspect, _, _ = raster_ops.read_as_array(str(gee_dem_path), band=3)
        terrain = {"elevation_m": elevation, "slope_deg": slope, "aspect_deg": aspect}
        out_path = processed_dir / "terrain_context.npz"
        np.savez_compressed(out_path, **terrain)
        logger.info("Wrote terrain context (from GEE Copernicus DEM): %s", out_path)
        return

    dem_files = sorted(raw_dir.glob("dem_*.tif"))
    if not dem_files:
        logger.warning(
            "No DEM data found in %s (neither dem_gee.tif nor CDSE dem_*.tif tiles) - "
            "skipping terrain context (run scripts/01 first)", raw_dir
        )
        return
    # CDSE path: a raw single-band DEM tile - compute slope/aspect locally.
    terrain = get_terrain_context(cfg, str(dem_files[0]))
    out_path = processed_dir / "terrain_context.npz"
    np.savez_compressed(out_path, **terrain)
    logger.info("Wrote terrain context (from CDSE DEM tile): %s", out_path)


def main() -> None:
    configure_logging()
    cfg = load_config()
    raw_dir = Path(cfg.paths.raw_dir)
    processed_dir = Path(cfg.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for study_year in cfg.study_years:
        process_one_year(cfg, study_year.year, raw_dir, processed_dir)

    process_terrain_context(cfg, raw_dir, processed_dir)
    logger.info("Stage 2 complete. Processed features written to %s", processed_dir)


if __name__ == "__main__":
    main()
