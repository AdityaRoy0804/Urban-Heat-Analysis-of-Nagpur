"""
STAC-based data acquisition - a free, no-billing-account alternative to
Google Earth Engine for Landsat, Sentinel-2, and Sentinel-1.

Two public STAC (SpatioTemporal Asset Catalog) providers are supported:

  "earthsearch" (default) - AWS Earth Search by Element84
      https://earth-search.aws.element84.com/v1
      Fully public, zero authentication, zero signup. Hosts Sentinel-2 L2A
      and Landsat Collection-2 L2 as Cloud-Optimized GeoTIFFs on S3.

  "planetary_computer" - Microsoft Planetary Computer
      https://planetarycomputer.microsoft.com/api/stac/v1
      Free tier, no billing account. Anonymous access works for light use;
      set PC_SDK_SUBSCRIPTION_KEY (also free to obtain) for higher rate
      limits. Hosts Sentinel-1, Sentinel-2, and Landsat.

Both are genuinely free (no credit card, no institutional verification) and
neither needs the Google Cloud project registration that blocks GEE for some
users. This module produces the exact same local GeoTIFF outputs as
`gee_client.py` (data/raw/{landsat,sentinel2,sentinel1}_{year}.tif), so
scripts/02_extract_features.py onward doesn't need to know which acquisition
path produced them.

Trade-off vs. GEE: GEE composites entirely server-side (you never move
pixels you don't need). Here, each function performs a genuine partial read
over HTTP (via GDAL's /vsicurl/, through rasterio) of just the AOI window
from each matching scene, then composites locally with numpy - efficient for
a city-sized AOI, but you are downloading real (if small) data rather than
computing remotely.

`pystac_client`, `planetary_computer`, and `rasterio` are all imported
lazily so the rest of the package stays importable without them.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from nagpur_uhi.config import Config, StudyYear

logger = logging.getLogger(__name__)

STAC_ENDPOINTS = {
    "earthsearch": "https://earth-search.aws.element84.com/v1",
    "planetary_computer": "https://planetarycomputer.microsoft.com/api/stac/v1",
}

COLLECTION_NAMES = {
    "earthsearch": {"landsat": "landsat-c2-l2", "sentinel2": "sentinel-2-l2a", "sentinel1": "sentinel-1-grd"},
    "planetary_computer": {"landsat": "landsat-c2-l2", "sentinel2": "sentinel-2-l2a", "sentinel1": "sentinel-1-grd"},
}


def _lazy_import_pystac_client():
    try:
        import pystac_client  # type: ignore
    except ImportError as e:
        raise ImportError("pystac-client is required: pip install pystac-client") from e
    return pystac_client


def _lazy_import_rasterio():
    try:
        import rasterio  # type: ignore
    except ImportError as e:
        raise ImportError("rasterio is required: pip install rasterio") from e
    return rasterio


def _sign_item_if_needed(item: Any, provider: str) -> Any:
    """Planetary Computer assets need a short-lived SAS token appended to their URL; Earth Search doesn't."""
    if provider != "planetary_computer":
        return item
    try:
        import planetary_computer  # type: ignore
    except ImportError as e:
        raise ImportError(
            "planetary-computer is required for provider='planetary_computer': pip install planetary-computer"
        ) from e
    return planetary_computer.sign(item)


def search_items(
    cfg: Config,
    dataset: str,
    study_year: StudyYear,
    provider: str = "earthsearch",
    max_cloud_cover_pct: int | None = None,
) -> list[Any]:
    """
    Search a STAC provider for scenes intersecting the AOI and date range.
    `dataset` is one of "landsat", "sentinel2", "sentinel1".
    """
    pystac_client = _lazy_import_pystac_client()
    endpoint = STAC_ENDPOINTS[provider]
    collection = COLLECTION_NAMES[provider][dataset]
    cloud_cover_limit = max_cloud_cover_pct if max_cloud_cover_pct is not None else cfg.gee.max_cloud_cover_pct

    catalog = pystac_client.Client.open(endpoint)
    query = {}
    if dataset != "sentinel1":  # SAR has no cloud-cover property
        cloud_property = "eo:cloud_cover"
        query[cloud_property] = {"lt": cloud_cover_limit}

    search = catalog.search(
        collections=[collection],
        bbox=cfg.aoi.bbox,
        datetime=f"{study_year.start}/{study_year.end}",
        query=query or None,
    )
    items = list(search.items())
    logger.info("STAC search (%s, %s, %s): %d matching scenes", provider, dataset, study_year.year, len(items))
    return [_sign_item_if_needed(item, provider) for item in items]


def _read_windowed_asset(rasterio_module, href: str, bounds_wgs84: tuple[float, float, float, float]) -> tuple[np.ndarray, Any, Any]:
    """Open a (possibly remote, https) raster asset and read just the AOI window."""
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    with rasterio_module.open(href) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, *bounds_wgs84)
        window = from_bounds(*bounds, transform=src.transform)
        array = src.read(1, window=window, out_dtype="float64", boundless=True, fill_value=np.nan)
        window_transform = src.window_transform(window)
        return array, window_transform, src.crs


def _composite_median(arrays: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(arrays, axis=0)
    with np.errstate(invalid="ignore"):
        return np.nanmedian(stacked, axis=0)


def build_composite(
    cfg: Config,
    dataset: str,
    study_year: StudyYear,
    bands: dict[str, str],
    out_path: str,
    provider: str = "earthsearch",
    max_scenes: int = 10,
) -> str:
    """
    Search, partially read, cloud/backscatter-composite, and write a local
    multi-band GeoTIFF for one dataset/study-year - the STAC equivalent of
    `gee_client.get_*_composite` + `export_composite_local` combined.

    `bands` maps output band name -> the STAC asset key to read for it
    (e.g. {"red": "red", "nir": "nir", "swir1": "swir16"} for Landsat on
    Earth Engine; asset keys differ slightly by provider/collection - check
    the collection's STAC item assets if a key isn't found).
    """
    rasterio = _lazy_import_rasterio()
    items = search_items(cfg, dataset, study_year, provider=provider)
    if not items:
        raise RuntimeError(
            f"No {dataset} scenes found for {study_year.year} via {provider} - "
            "widen the date range or raise max_cloud_cover_pct in config.yaml"
        )
    items = items[:max_scenes]

    band_composites: dict[str, np.ndarray] = {}
    reference_transform = reference_crs = None

    for band_name, asset_key in bands.items():
        arrays = []
        for item in items:
            asset = item.assets.get(asset_key)
            if asset is None:
                logger.warning("Asset '%s' not found on item %s; skipping this scene for this band", asset_key, item.id)
                continue
            array, transform, crs = _read_windowed_asset(rasterio, asset.href, tuple(cfg.aoi.bbox))
            arrays.append(array)
            reference_transform, reference_crs = transform, crs
        if not arrays:
            raise RuntimeError(f"No usable scenes provided asset '{asset_key}' for band '{band_name}'")
        band_composites[band_name] = _composite_median(arrays)

    band_names = list(band_composites.keys())
    height, width = band_composites[band_names[0]].shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(band_names),
        "dtype": "float64",
        "crs": reference_crs,
        "transform": reference_transform,
        "nodata": np.nan,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        for i, band_name in enumerate(band_names, start=1):
            dst.write(band_composites[band_name], i)
            dst.set_band_description(i, band_name)

    logger.info("Wrote STAC composite: %s (%s)", out_path, band_names)
    return out_path


def convert_landsat_lst_band_to_celsius(path: str, lst_band_index: int) -> None:
    """
    Rewrite a raw Landsat Collection-2 thermal band (as read via the STAC
    'lwir11' asset - a raw digital number, same convention as a direct USGS
    download) in-place as Celsius, using the official scale/offset in
    `features/indices.py`. Verify against the STAC item's `raster:bands`
    scale/offset metadata if results look implausible - this assumes the
    standard Collection-2 Level-2 convention.
    """
    rasterio = _lazy_import_rasterio()
    from nagpur_uhi.features.indices import lst_celsius_from_landsat_c2

    with rasterio.open(path) as src:
        profile = src.profile
        bands = [src.read(i) for i in range(1, src.count + 1)]
        descriptions = list(src.descriptions)

    bands[lst_band_index - 1] = lst_celsius_from_landsat_c2(bands[lst_band_index - 1])
    if descriptions[lst_band_index - 1] == "lst_dn":
        descriptions[lst_band_index - 1] = "LST_Celsius"

    with rasterio.open(path, "w", **profile) as dst:
        for i, band in enumerate(bands, start=1):
            dst.write(band, i)
            if descriptions[i - 1]:
                dst.set_band_description(i, descriptions[i - 1])
