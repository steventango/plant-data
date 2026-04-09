#!/bin/bash
set -euo pipefail

# ── E17 metrics plot ──────────────────────────────────────────────────────────
echo "=== Plotting E17 metrics ==="
for max_steps in 2 3 4 5 6 7 14; do
    uv run python visualization/plot_e16_metrics.py \
        --experiment 17 \
        --output results/e17_metrics_${max_steps} \
        --min-return 0.01 \
        --drop-iqr-outliers \
        --iqr-multiplier 1.0 \
        --max-steps ${max_steps}
done

# ── E17 timelapses ────────────────────────────────────────────────────────────
echo "=== Creating E17 timelapses ==="
PARQUET="/data/plant-rl/offline/v24/mixed-v24.parquet"
ZONES=(1 2 3 4 5 6 7 8 9 10 11 12)
mkdir -p results/timelapses/E17

for ZONE in "${ZONES[@]}"; do
    echo "Creating timelapse for E17 Zone $ZONE..."
    uv run python visualization/create_timelapse.py \
        --parquet "$PARQUET" \
        --experiment 17 \
        --zone "$ZONE" \
        --day-cutoff 14 \
        --output "results/timelapses/E17/E17_Z$ZONE.mp4" \
        --framerate 1
done

echo "Creating tarball of E17 timelapses..."
tar -czvf results/E17_timelapses.tar.gz results/timelapses/E17

# ── E17 frame strip ──────────────────────────────────────────────────────────
echo "=== Creating E17 frame strip ==="
uv run python visualization/create_frame_strip.py \
    --experiment 17 \
    --out results/e17_frame_strip.jpg

echo "=== Done ==="
