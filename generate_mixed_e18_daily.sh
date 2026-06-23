#!/bin/bash
# Generate the E18 Zones 1,3,4,11 daily (9:00 AM local) mixed dataset with intensity action.
# Reuses the existing v27 processed parquets (no vision pipeline needed).

set -euo pipefail

uv run python join_zones.py \
    --root-dir /data/plant-rl/online \
    --experiments 18 \
    --zones 1,3,4,11 \
    --subsample daily \
    --output-name mixed-e18-daily-v27

uv run python create_minari_dataset.py \
    --input /data/plant-rl/offline/v27/mixed-e18-daily-v27.parquet \
    --name visu \
    --action intensity
