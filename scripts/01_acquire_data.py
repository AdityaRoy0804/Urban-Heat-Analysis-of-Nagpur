#!/usr/bin/env python3
"""
Stage 1 - Data acquisition.

Pulls, per study year: a Landsat composite (LST + optical bands), a
Sentinel-2 composite (optical), and a Sentinel-1 composite (SAR); plus one
Copernicus DEM tile (terrain context) via the Copernicus Data Space
Ecosystem (CDSE) - always free, no billing account.

Two interchangeable sources for the Landsat/Sentinel-1/Sentinel-2 composites:

  --source stac (default) - free, public STAC catalogs (AWS Earth Search by
      default; no signup, no billing account at all). See
      data_acquisition/stac_client.py. This is the recommended default if
      you don't already have Earth Engine access, since it needs nothing
      beyond `pip install -r requirements.txt`.

  --source gee - Google Earth Engine (server-side compositing, generally
      faster/more robust cloud masking, but requires a registered - free for
      genuine noncommercial use - Google Cloud project; see README for the
      noncommercial registration flow).

Either way, this script requires network access and cannot run in a
network-isolated sandbox - run it in your own environment. If you have
neither GEE nor a fast connection available yet, use
scripts/00_generate_synthetic_demo_data.py to exercise the rest of the
pipeline immediately with synthetic data instead.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nagpur_uhi.config import load_config
from nagpur_uhi.data_acquisition import copernicus_client, stac_client
from nagpur_uhi.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)

# STAC asset keys differ by provider/collection; these are Earth Search's
# landsat-c2-l2 / sentinel-2-l2a / sentinel-1-grd asset keys.
STAC_BANDS = {
    "landsat": {"red": "red", "nir": "nir08", "swir1": "swir16", "lst_dn": "lwir11"},
    "sentinel2": {"green": "green", "red": "red", "nir": "nir", "swir1": "swir16"},
    "sentinel1": {"vv": "vv", "vh": "vh"},
}


def acquire_via_stac(cfg, raw_dir: Path, provider: str) -> None:
    for study_year in cfg.study_years:
        logger.info("Acquiring STAC (%s) composites for %s", provider, study_year.year)
        for dataset in ["landsat", "sentinel2", "sentinel1"]:
            out_path = raw_dir / f"{dataset}_{study_year.year}.tif"
            stac_client.build_composite(
                cfg, dataset, study_year, bands=STAC_BANDS[dataset], out_path=str(out_path), provider=provider
            )
            if dataset == "landsat":
                # STAC's 'lwir11' asset is the raw digital number (same convention as a
                # direct USGS download) - convert to Celsius so downstream stages see the
                # same units regardless of whether GEE or STAC produced this file.
                stac_client.convert_landsat_lst_band_to_celsius(str(out_path), lst_band_index=len(STAC_BANDS["landsat"]))


def acquire_via_gee(cfg, raw_dir: Path) -> None:
    from nagpur_uhi.data_acquisition import gee_client  # lazy: only needed for this path

    logger.info("Initializing Earth Engine (project=%s)", cfg.gee.project_id)
    gee_client.initialize(cfg.gee.project_id)

    for study_year in cfg.study_years:
        logger.info("Acquiring GEE composites for %s", study_year.year)

        landsat = gee_client.get_landsat_composite(cfg, study_year)
        gee_client.export_composite_local(
            landsat, str(raw_dir / f"landsat_{study_year.year}.tif"), cfg, scale=30,
            bands=["SR_B4", "SR_B5", "SR_B6", "LST_Celsius"],
        )
        sentinel2 = gee_client.get_sentinel2_composite(cfg, study_year)
        gee_client.export_composite_local(
            sentinel2, str(raw_dir / f"sentinel2_{study_year.year}.tif"), cfg, scale=30, bands=["B3", "B4", "B8", "B11"],
        )
        sentinel1 = gee_client.get_sentinel1_composite(cfg, study_year)
        gee_client.export_composite_local(
            sentinel1, str(raw_dir / f"sentinel1_{study_year.year}.tif"), cfg, scale=30,
            bands=["VV", "VH", "sar_vv_vh_ratio"],
        )


def acquire_dem_via_gee(cfg, raw_dir: Path) -> None:
    """
    Copernicus DEM via Earth Engine - the recommended default if you already
    have GEE working, since it avoids needing a separate CDSE account.
    """
    from nagpur_uhi.data_acquisition import gee_client  # lazy: only needed for this path

    logger.info("Fetching Copernicus DEM terrain context via Earth Engine")
    terrain = gee_client.get_terrain_context_composite(cfg)
    out_path = raw_dir / "dem_gee.tif"
    gee_client.export_composite_local(terrain, str(out_path), cfg, scale=30)
    logger.info("Wrote %s (elevation_m, slope_deg, aspect_deg bands)", out_path)


def acquire_dem_via_cdse(cfg, raw_dir: Path) -> None:
    logger.info("Fetching Copernicus DEM tiles from CDSE (terrain context, acquired once - not per year)")
    token = copernicus_client.get_access_token(cfg)
    products = copernicus_client.search_dem_products(cfg, token)
    if not products:
        logger.warning(
            "No Copernicus DEM products found for the configured AOI/collection name - "
            "verify config.copernicus_dataspace.dem_collection_name against current CDSE docs."
        )
    for product in products:
        out_path = raw_dir / f"dem_{product['Id']}.tif"
        logger.info("Downloading DEM tile %s -> %s", product["Id"], out_path)
        copernicus_client.download_product(cfg, token, product["Id"], str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", choices=["stac", "gee"], default="stac",
        help="Where to pull Landsat/Sentinel-1/Sentinel-2 from (default: stac - free, no billing account)",
    )
    parser.add_argument(
        "--stac-provider", choices=["earthsearch", "planetary_computer"], default="earthsearch",
        help="Which public STAC catalog to use when --source=stac (default: earthsearch - zero auth)",
    )
    parser.add_argument(
        "--dem-source", choices=["gee", "cdse"], default="gee",
        help="Where to pull the Copernicus DEM terrain context from (default: gee - reuses your existing "
             "Earth Engine registration; use cdse only if you specifically need the independent CDSE data path)",
    )
    args = parser.parse_args()

    configure_logging()
    cfg = load_config()
    raw_dir = Path(cfg.paths.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "stac":
        acquire_via_stac(cfg, raw_dir, provider=args.stac_provider)
    else:
        acquire_via_gee(cfg, raw_dir)

    if args.dem_source == "gee":
        acquire_dem_via_gee(cfg, raw_dir)
    else:
        acquire_dem_via_cdse(cfg, raw_dir)

    logger.info("Stage 1 complete. Raw data written to %s", raw_dir)


if __name__ == "__main__":
    main()
