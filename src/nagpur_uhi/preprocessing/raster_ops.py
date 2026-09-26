"""
Raster preprocessing: reprojection, clipping, resampling, and alignment
validation. All functions operate on GeoTIFFs written by the data-acquisition
scripts.

`rasterio` is imported lazily so this module - and, transitively, anything
that imports it just to reuse `validate_alignment`'s pure-numpy logic - stays
importable in environments without the GIS stack installed (e.g. this
repo's unit-test environment). In a real deployment, install the
`requirements.txt` GIS extras and every function below runs directly.
"""

from __future__ import annotations

from typing import Any


def _lazy_import_rasterio():
    try:
        import rasterio  # type: ignore
    except ImportError as e:
        raise ImportError("rasterio is required for raster I/O: pip install rasterio") from e
    return rasterio


def reproject_and_resample(
    src_path: str,
    dst_path: str,
    dst_crs_epsg: int,
    dst_resolution_m: float,
    resampling: str = "bilinear",
) -> str:
    """Reproject a raster to `dst_crs_epsg` at `dst_resolution_m` resolution."""
    rasterio = _lazy_import_rasterio()
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    resampling_method = getattr(Resampling, resampling)

    with rasterio.open(src_path) as src:
        dst_crs = f"EPSG:{dst_crs_epsg}"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=dst_resolution_m
        )
        meta = src.meta.copy()
        meta.update({"crs": dst_crs, "transform": transform, "width": width, "height": height})

        with rasterio.open(dst_path, "w", **meta) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling_method,
                )
    return dst_path


def clip_to_aoi(src_path: str, dst_path: str, aoi_geojson_path: str) -> str:
    """Clip a raster to an AOI polygon (GeoJSON), masking pixels outside it."""
    rasterio = _lazy_import_rasterio()
    import json

    from rasterio.mask import mask

    with open(aoi_geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)
    shapes = [feature["geometry"] for feature in geojson["features"]]

    with rasterio.open(src_path) as src:
        out_image, out_transform = mask(src, shapes, crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {"height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform}
        )

    with rasterio.open(dst_path, "w", **out_meta) as dst:
        dst.write(out_image)
    return dst_path


def read_as_array(path: str, band: int = 1):
    """Read a single band as a numpy array plus its affine transform and CRS."""
    rasterio = _lazy_import_rasterio()
    with rasterio.open(path) as src:
        array = src.read(band)
        return array, src.transform, src.crs


def validate_alignment(rasters: dict[str, Any]) -> None:
    """
    Validate that a set of already-loaded rasters (as returned by
    `read_as_array`, i.e. (array, transform, crs) tuples) share the same
    shape, transform and CRS before they are stacked into a feature table.

    This is intentionally pure Python/numpy (no rasterio needed) so it can be
    unit-tested with synthetic transforms without the GIS stack installed.
    Raises ValueError with a specific mismatch description on failure.
    """
    if not rasters:
        raise ValueError("No rasters provided to validate_alignment")

    names = list(rasters.keys())
    ref_name = names[0]
    ref_array, ref_transform, ref_crs = rasters[ref_name]

    for name in names[1:]:
        array, transform, crs = rasters[name]
        if array.shape != ref_array.shape:
            raise ValueError(
                f"Shape mismatch: '{name}' has shape {array.shape}, "
                f"expected {ref_array.shape} (from '{ref_name}')"
            )
        if tuple(transform) != tuple(ref_transform):
            raise ValueError(
                f"Affine transform mismatch between '{name}' and '{ref_name}' - "
                "rasters are not pixel-aligned. Re-run reproject_and_resample with "
                "a shared reference grid."
            )
        if str(crs) != str(ref_crs):
            raise ValueError(f"CRS mismatch: '{name}' is {crs}, expected {ref_crs} (from '{ref_name}')")
