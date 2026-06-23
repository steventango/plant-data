#!/bin/bash

# Ensure output directory exists
mkdir -p results/timelapses/E16

# Array of E16 zones
ZONES=(1 2 3 4 5 6 7 8 9 10 11 12)

# Parquet file path
PARQUET="/data/plant-rl/offline/v24/mixed-v24.parquet"

# Generate timelapses for each zone
for ZONE in "${ZONES[@]}"; do
    echo "Creating timelapse for E16 Zone $ZONE..."
    uv run python -m visualization.create_timelapse \
        --parquet "$PARQUET" \
        --experiment 16 \
        --zone "$ZONE" \
        --day-cutoff 14 \
        --output "results/timelapses/E16/E16_Z$ZONE.mp4" \
        --framerate 1
done

# Create a tarball of the timelapses
echo "Creating tarball of E16 timelapses..."
tar -czvf results/E16_timelapses.tar.gz results/timelapses/E16
