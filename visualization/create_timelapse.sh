#!/bin/bash

# Create timelapses using the python script
uv run python visualization/create_timelapse.py --experiment 11 --zone 1 --day-cutoff 14 --output E11_Z1.mp4
uv run python visualization/create_timelapse.py --experiment 12 --zone 1 --day-cutoff 14 --output E12_Z1.mp4
uv run python visualization/create_timelapse.py --experiment 13 --zone 1 --day-cutoff 14 --output E13_Z1.mp4
uv run python visualization/create_timelapse.py --experiment 14 --zone 1 --day-cutoff 14 --output E14_Z1.mp4
uv run python visualization/create_timelapse.py --experiment 15 --zone 2 --day-cutoff 14 --output E15_Z2.mp4
uv run python visualization/create_timelapse.py --experiment 15 --zone 3 --day-cutoff 14 --output E15_Z3.mp4
uv run python visualization/create_timelapse.py --experiment 15 --zone 4 --day-cutoff 14 --output E15_Z4.mp4

# ffmpeg -y -framerate 10 -pattern_type glob -i  results/E15_Z4_930AM_raw_v21/*.jpg -c:v mpeg4 -q:v 5 E15_Z4_930AM.mp4