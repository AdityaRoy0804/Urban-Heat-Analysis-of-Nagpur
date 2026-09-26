#!/usr/bin/env bash
# Stage 5 - Serve the API (+ optionally titiler for raster tile layers).
#
# The FastAPI app (zone history + scenario simulation) and titiler (raster
# tiles for the LST/NDVI/hotspot overlays, served from the Cloud-Optimized
# GeoTIFFs produced in earlier stages) are run as two separate processes -
# this keeps concerns cleanly separated and lets you scale/restart them
# independently.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Starting FastAPI app on :8000 ..."
uvicorn nagpur_uhi.api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

if command -v titiler >/dev/null 2>&1; then
  echo "Starting titiler on :8001 for raster tile layers (point it at data/processed/*.tif, converted to COG) ..."
  uvicorn titiler.application.main:app --host 0.0.0.0 --port 8001 &
  TILER_PID=$!
else
  echo "titiler not installed (pip install titiler[server]) - raster tile overlays will not be served."
  TILER_PID=""
fi

trap 'kill $API_PID ${TILER_PID:-} 2>/dev/null' EXIT
wait
