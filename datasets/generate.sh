#!/bin/bash
uv run python datasets/process_zone.py --data-path /data/plant-rl/online/E14/P1/Constant1/alliance-zone01/
uv run python datasets/process_zone.py --data-path /data/plant-rl/online/E14/P1/InAC2/alliance-zone02/
uv run python datasets/join_zones.py --root-dir /data/plant-rl/online --output-dir /data/plant-rl/offline/v16/
uv run python datasets/create_minari_dataset.py --input-dir /data/plant-rl/offline/v16/