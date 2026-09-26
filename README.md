# Nagpur Urban Heat Island Platform

Multi-temporal urban heat island (UHI) analysis and **explainable, scenario-based**
land-cover impact estimation for Nagpur. Built around the pipeline discussed in
the project design: satellite acquisition → preprocessing → feature extraction
→ multi-temporal spatial analysis → an explainable ML scenario engine → an API
→ an interactive map/scenario frontend, with a Copernicus Data Space Ecosystem
branch feeding terrain context directly into the scenario engine.

## Already have GEE working? You can skip CDSE entirely

If your Earth Engine registration is working (as confirmed by successfully
reading any GEE dataset), you don't need a separate Copernicus Data Space
Ecosystem account at all right now. GEE hosts a copy of Copernicus DEM
(`COPERNICUS/DEM/GLO30`) alongside Landsat and Sentinel-1/2 - same underlying
ESA data, accessed through the connection you already have:

```bash
python scripts/01_acquire_data.py --source gee --dem-source gee
```

`--dem-source gee` (the default) fetches terrain context via
`gee_client.get_terrain_context_composite`, computing slope/aspect
server-side with `ee.Terrain` - no separate account, no CDSE captcha to
fight with. Use `--dem-source cdse` only if you specifically want the
independent Copernicus Data Space Ecosystem data path (e.g. for the
redundancy/cross-validation rationale discussed when this branch was first
designed) once CDSE registration is working for you.

## No-cost data options (no Google Cloud billing needed)

If Google Earth Engine's registration flow is asking for a payment: per
Google's own docs, the default **Community Tier** for individual/noncommercial
use requires **no billing account at all** ("If you register a noncommercial
project, no billing configuration is required"). A mandatory ~$1000 charge is
not part of any standard noncommercial flow - retry registration selecting
**"Individual using Earth Engine for noncommercial purposes"**, which should
land you on the free Community Tier. That said, you don't need GEE at all to
use this repo. Three genuinely free alternatives, in order of how quickly
they get you moving:

1. **Synthetic demo data (zero setup, zero data, works right now):**
   ```bash
   python scripts/00_generate_synthetic_demo_data.py
   python scripts/03_run_analysis.py
   python scripts/04_train_model.py
   # On Linux/macOS:
   bash scripts/05_run_api.sh
   # On Windows (PowerShell):
   powershell scripts/05_run_api.ps1
   ```
   Generates a deterministic, physically-plausible synthetic Nagpur scene
   (urban core + a faster-growing peri-urban cluster, two water bodies, a
   green belt, all evolving across the study years) and runs the entire
   pipeline on it. **This is not real satellite data** - every log line and
   docstring says so - but it exercises every layer (hotspots, relationships,
   the ML model, the explainer, the scenario engine, the API) end to end so
   you can develop and demo the platform immediately, then swap in real data
   later without changing anything downstream.

2. **Free public STAC catalogs (real data, no billing account, no signup for the default provider):**
   ```bash
   python scripts/01_acquire_data.py --source stac                     # AWS Earth Search - zero auth
   python scripts/01_acquire_data.py --source stac --stac-provider planetary_computer  # optional alternative
   python scripts/02_extract_features.py
   python scripts/03_run_analysis.py
   python scripts/04_train_model.py
   ```
   `data_acquisition/stac_client.py` pulls the same Landsat Collection-2 L2,
   Sentinel-2 L2A, and Sentinel-1 GRD data GEE would give you, via public STAC
   APIs instead: AWS Earth Search needs no account at all; Microsoft Planetary
   Computer needs at most a free API key for higher rate limits. Trade-off:
   compositing happens locally (partial HTTP reads via rasterio) rather than
   server-side, so it's a bit slower for a large AOI/many years than GEE, and
   cloud filtering here is scene-level (via STAC's cloud-cover metadata)
   rather than GEE's per-pixel QA mask - fine for a city-sized AOI, worth
   knowing if results look noisier than expected.

3. **Manual GUI download (no coding, no API of any kind):**
   - Landsat Collection 2 Level-2: [USGS EarthExplorer](https://earthexplorer.usgs.gov)
     (free USGS account, no payment) - search Nagpur + your study-year date
     ranges, download the SR + ST bands.
   - Sentinel-1 GRD / Sentinel-2 L2A: [Copernicus Browser](https://browser.dataspace.copernicus.eu)
     (free CDSE account, no payment) - the same free CDSE account the DEM
     branch (`copernicus_client.py`) already uses.
   - Place the downloaded, mosaicked GeoTIFFs at `data/raw/landsat_{year}.tif`,
     `data/raw/sentinel2_{year}.tif`, `data/raw/sentinel1_{year}.tif` (band
     order matching `scripts/02_extract_features.py`'s expectations - red,
     NIR, SWIR1[, LST] for Landsat; green, red, NIR, SWIR1 for Sentinel-2; VV,
     VH for Sentinel-1), then run `scripts/02_extract_features.py` onward as
     normal - it only cares that the files exist with the right name/bands,
     not how they got there.

## What this is (and isn't)

- **LST is Land Surface Temperature**, not air temperature - retrieved from
  Landsat Collection-2 Level-2's official `ST_B10` product, not reimplemented
  from raw brightness temperature.
- **The scenario model is a statistical surrogate**, not a physics-based
  climate simulator or a forecaster. It answers "how has LST responded to
  this kind of land-cover change elsewhere, adjusted for this location's
  terrain" - explicitly scenario-based estimation, not absolute prediction.
- **Explainability is prioritised over raw accuracy.** Every scenario result
  ships with a feature-by-feature explanation (SHAP, or a documented
  occlusion-based fallback), a historical hotspot context, and an uncertainty
  band derived from real analog zones - not a bare number.

## Repository structure

```
config/config.yaml           All tunable parameters (AOI, study years, thresholds, model hyperparams)
src/nagpur_uhi/
  config.py                  Typed config loader
  data_acquisition/
    gee_client.py               Landsat + Sentinel-1 + Sentinel-2 + Copernicus DEM, all via Earth Engine (recommended if GEE is already working for you)
    stac_client.py              Same imagery via free public STAC catalogs (no billing account, alternative if GEE isn't set up yet)
    copernicus_client.py       Copernicus DEM via CDSE directly (optional independent data path)
  preprocessing/
    raster_ops.py              Reprojection, clipping, alignment validation (rasterio)
  features/
    indices.py                  NDVI, NDBI, NDWI, SAR VV/VH ratio, Landsat C2 LST conversion (pure numpy)
    lulc.py                      Rule-based (explainable) + trainable Random Forest LULC classifiers
  analysis/
    hotspots.py                  Getis-Ord Gi* hotspot detection + multi-year persistence overlay
    relationships.py             Global correlation + approximate GWR (local regression map)
  ml/
    dataset.py                    Tabular dataset assembly + spatial-block train/test split
    model.py                      RF/GBM LST model + directional-validity backtest
    explain.py                    SHAP-based (or occlusion-fallback) local explanations
    scenario.py                   Analog-grounded "what if" scenario simulation engine
  api/
    main.py, schemas.py           FastAPI service: zone history + scenario simulation endpoints
scripts/
  00_generate_synthetic_demo_data.py   Optional - synthetic demo data, zero setup, zero cost
  01_acquire_data.py            Stage 1: real data via free STAC (default) or GEE (needs network; see "No-cost data options" above)
  02_extract_features.py       Stage 2 (needs rasterio)
  03_run_analysis.py           Stage 3 (hotspots + relationships)
  04_train_model.py            Stage 4 (zone table + model training + backtest)
  05_run_api.sh                 Stage 5 (FastAPI + optional titiler)
frontend/index.html            Single-file interactive map + time slider + scenario panel prototype
tests/                          Unit tests for every pure-Python module (see below)
```

## What's actually executable right now vs. what needs your own environment

This matters, so being explicit about it:

| Layer | Runs with just `pip install -r requirements.txt`? | Notes |
|---|---|---|
| `features/`, `analysis/`, `ml/` | **Yes** - pure numpy/pandas/scikit-learn | Fully unit-tested (see below) |
| `tests/` (all of them) | **Yes** | 37 tests, all passing; `test_api.py` self-skips if `fastapi` isn't installed |
| `scripts/00_generate_synthetic_demo_data.py` | **Yes** - pure numpy/scipy | Generates synthetic input for stages 3-4; see "No-cost data options" above |
| `data_acquisition/stac_client.py` | Needs `pystac-client`, `rasterio`, network | Free, no billing account - the recommended default for real data |
| `data_acquisition/gee_client.py` | Needs `earthengine-api`, `geemap`, network, and a GEE project + `earthengine authenticate` | Optional alternative; imports are lazy so the rest of the package works without it |
| `data_acquisition/copernicus_client.py` | Needs network + a CDSE account (`CDSE_CLIENT_ID`/`SECRET`) | Verify `dem_collection_name` in `config.yaml` against current CDSE docs before running - catalogue naming has changed over time |
| `preprocessing/raster_ops.py` | Needs `rasterio` | Lazily imported |
| `api/` | Needs `fastapi`, `uvicorn`, `pydantic` | |
| `scripts/01`-`05` | Full pipeline, needs everything above | Meant to be run end-to-end in your own machine/server, not in a network-isolated sandbox |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                 # installs the nagpur_uhi package in editable mode
cp .env.example .env             # fill in GEE_PROJECT_ID, CDSE_CLIENT_ID/SECRET
earthengine authenticate         # one-time GEE auth
```

## Running the tests

```bash
pytest tests/ -v
# or, with zero extra dependencies beyond requirements.txt's core section:
python -m unittest discover -s tests -v
```

All 37 tests pass using only `numpy`/`pandas`/`scipy`/`scikit-learn` - they
validate the actual scientific/ML logic (index formulas, the Gi* statistic
against a synthetic hot cluster, the relationship analysis against a known
synthetic linear relationship, the model's learned direction of effect, the
explainer's additive consistency, and the scenario engine's analog matching)
without needing GEE credentials, rasterio, or real satellite data.

## Running the full pipeline (your own environment, with real data)

```bash
python scripts/01_acquire_data.py               # default: free STAC (Earth Search) - no billing account
python scripts/01_acquire_data.py --source gee  # optional: Earth Engine instead, if you have access
python scripts/02_extract_features.py           # needs rasterio
python scripts/03_run_analysis.py
python scripts/04_train_model.py
bash scripts/05_run_api.sh                      # serves FastAPI on :8000 (+ titiler on :8001 if installed)
# open frontend/index.html in a browser (or serve it statically) - it talks to localhost:8000
```

Or skip stage 1+2 entirely and start from `scripts/00_generate_synthetic_demo_data.py`
(see "No-cost data options" above) if you want to see the platform working
before wiring up any real data source.

## Design notes worth knowing before you extend this

- **Zoning is a placeholder.** `scripts/04_train_model.py` aggregates pixels
  into a regular grid as a stand-in for real administrative/locality
  boundaries (wards, gram panchayat limits like "Hingna Gramin"). Swap in a
  real polygon layer + `rasterstats.zonal_stats`/geopandas spatial join for
  production so `zone_id` becomes a real place name.
- **`local_regression_map`** (approximate GWR) is a simple per-pixel moving-window
  linear fit, O(rows×cols); fine for a prototype AOI, but swap in the `mgwr`
  package for a rigorous, faster kernel-weighted GWR at scale.
- **Directional validity, not R².** `ml/model.py`'s `evaluate_directional_validity`
  is the metric that actually matters for a scenario tool: it backtests
  whether the model gets the *sign and rough magnitude* of an LST change
  right when fed real historical feature changes, rather than asking "how
  close is this to a ground truth we don't have for hypothetical scenarios."
- **Tile serving uses `titiler`**, not a bespoke implementation - see
  `scripts/05_run_api.sh`. Convert your processed GeoTIFFs to Cloud-Optimized
  GeoTIFFs (`rio cogeo create`) before pointing titiler at them.
