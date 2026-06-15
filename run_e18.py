#!/usr/bin/env python3
"""
Run analysis, metrics plotting, timelapses, and frame strip generation for Experiment 18.
"""

import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], description: str) -> None:
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}", file=sys.stderr)
        sys.exit(e.returncode)


def main():
    root_dir = Path(__file__).resolve().parent
    parquet_path = "/data/plant-rl/offline/v27/mixed-v27.parquet"
    zones = [1, 2, 3, 4, 5, 11]
    max_steps_list = [2, 3, 4, 5, 6, 7, 8]
    max_day = 8

    # 1. Plot E18 Metrics
    print("=== Plotting E18 Metrics ===")
    for max_steps in max_steps_list:
        run_cmd(
            [
                "uv",
                "run",
                "python",
                "visualization/plot_e16_metrics.py",
                "--experiment",
                "18",
                "--output",
                f"results/e18_metrics_{max_steps}",
                "--min-return",
                "0.01",
                "--drop-iqr-outliers",
                "--iqr-multiplier",
                "1.0",
                "--max-steps",
                str(max_steps),
            ],
            f"Plotting metrics for max_steps={max_steps}",
        )

    # 2. Plot E18 Metrics for specific zones
    run_cmd(
        [
            "uv",
            "run",
            "python",
            "visualization/plot_e16_metrics.py",
            "--experiment",
            "18",
            "--output",
            f"results/e18_metrics_{max_day}",
            "--min-return",
            "0.01",
            "--drop-iqr-outliers",
            "--iqr-multiplier",
            "1.0",
            "--max-steps",
            str(max_day),
            "--zone",
        ]
        + [str(z) for z in zones],
        "Plotting metrics for max_day and specific zones",
    )

    # 3. Create E18 Timelapses
    timelapse_dir = root_dir / "results" / "timelapses" / "E18"
    timelapse_dir.mkdir(parents=True, exist_ok=True)

    for zone in zones:
        run_cmd(
            [
                "uv",
                "run",
                "python",
                "visualization/create_timelapse.py",
                "--parquet",
                parquet_path,
                "--experiment",
                "18",
                "--zone",
                str(zone),
                "--day-cutoff",
                str(max_day),
                "--output",
                f"results/timelapses/E18/E18_Z{zone}.mp4",
                "--framerate",
                "1",
            ],
            f"Creating timelapse for E18 Zone {zone}",
        )

    # Tar the timelapses
    run_cmd(
        [
            "tar",
            "-czvf",
            "results/E18_timelapses.tar.gz",
            "results/timelapses/E18",
        ],
        "Creating tarball of E18 timelapses",
    )

    # 4. Create E18 Frame Strip
    run_cmd(
        [
            "uv",
            "run",
            "python",
            "visualization/create_frame_strip.py",
            "--experiment",
            "18",
            "--out",
            "results/e18_frame_strip.jpg",
            "--max-day",
            str(float(max_day)),
        ],
        "Creating E18 frame strip",
    )

    # 5. Plot reward over time
    run_cmd(
        [
            "uv",
            "run",
            "python",
            "visualization/plot_reward_over_time.py",
            "--experiment",
            "18",
            "--max-steps",
            str(max_day),
            "--output",
            "results/reward_over_time",
        ],
        "Plotting reward over time for E18",
    )

    # 6. Plot return and energy trade-off grouped by agent
    run_cmd(
        [
            "uv",
            "run",
            "python",
            "visualization/plot_e18_return_energy.py",
            "--parquet",
            parquet_path,
            "--out",
            "results/e18_return_energy.png",
        ],
        "Plotting Return and Energy trade-off grouped by Agent for E18",
    )

    print("\n=== All E18 steps finished successfully ===")


if __name__ == "__main__":
    main()
