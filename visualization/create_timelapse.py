import argparse
import logging
import math
import os
import subprocess
from pathlib import Path

import polars as pl
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Create a timelapse from plant dataset."
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default="/data/plant-rl/offline/v21/mixed-v21.parquet",
        help="Path to parquet file",
    )
    parser.add_argument("--experiment", type=int, required=True, help="Experiment ID")
    parser.add_argument("--zone", type=int, required=True, help="Zone ID")
    parser.add_argument(
        "--day-cutoff", type=float, default=14.0, help="Day cutoff (wall_time)"
    )
    parser.add_argument("--output", type=str, help="Output video path")
    parser.add_argument("--framerate", type=int, default=10, help="Video framerate")

    args = parser.parse_args()

    if args.output is None:
        args.output = f"E{args.experiment}_Z{args.zone}.mp4"

    logging.info(f"Loading parquet: {args.parquet}")
    if not os.path.exists(args.parquet):
        logging.error(f"Parquet file not found: {args.parquet}")
        return

    df = pl.read_parquet(args.parquet)

    # Filter by experiment, zone, and day cutoff
    df = df.filter(
        (pl.col("experiment") == args.experiment)
        & (pl.col("zone") == args.zone)
        & (pl.col("wall_time") <= args.day_cutoff)
    )

    if df.is_empty():
        logging.error(
            f"No data found for E{args.experiment} Z{args.zone} with day <= {args.day_cutoff}"
        )
        return

    # Group by wall_time to aggregate plants in the same tray/timestamp
    agg_df = (
        df.group_by("wall_time")
        .agg(
            [
                pl.col("image_path").first(),
                pl.col("agent").first(),
                pl.col("area").mean().alias("mean_area"),
                pl.col("clean_area").mean().alias("mean_clean_area"),
                pl.col("reward").mean().alias("mean_reward"),
                pl.col("red_coef").mean().alias("red_coef"),
                pl.col("white_coef").mean().alias("white_coef"),
                pl.col("blue_coef").mean().alias("blue_coef"),
            ]
        )
        .sort("wall_time")
        .with_columns(pl.col("mean_reward").cum_sum().alias("mean_return"))
    )

    tmp_dir = Path(f"/tmp/timelapse_E{args.experiment}_Z{args.zone}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Clean up old frames
    logging.info(f"Cleaning up old frames in {tmp_dir}")
    for f in tmp_dir.glob("*.jpg"):
        f.unlink()

    logging.info(f"Generating {len(agg_df)} frames in {tmp_dir}")

    # Try to load a font, fallback to default
    try:
        # Common locations for fonts on Linux
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, 40)
                break
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    for i, row in enumerate(tqdm(agg_df.iter_rows(named=True), total=len(agg_df))):
        orig_img_path = row["image_path"]
        agent_name = row["agent"]
        wall_time = row["wall_time"]
        mean_area = row["mean_area"]
        mean_clean_area = row["mean_clean_area"]
        mean_return = row["mean_return"]
        r, w, b = row["red_coef"], row["white_coef"], row["blue_coef"]

        # Convert individual plant image path to tray visualization path
        # From: .../processed/v21/images/0/2025-11-12T093000.jpg
        # To:   .../processed/v21/visualizations/E14_Z1_2025-11-12T093000_viz.jpg
        p = Path(orig_img_path)
        timestamp = p.stem  # 2025-11-12T093000
        # /images/0/2025-11-12T093000.jpg -> parent.parent.parent is /processed/v21/
        viz_dir = p.parent.parent.parent / "visualizations"
        viz_filename = f"E{args.experiment}_Z{args.zone}_{timestamp}_viz.jpg"
        img_path = viz_dir / viz_filename

        if not img_path.exists():
            logging.warning(f"Image not found: {img_path}")
            continue

        try:
            orig_img = Image.open(img_path)
            header_h = 160
            # Create a new image with a black header at the top
            img = Image.new(
                "RGB", (orig_img.width, orig_img.height + header_h), (0, 0, 0)
            )
            img.paste(orig_img, (0, header_h))
            draw = ImageDraw.Draw(img)

            # Metrics line - rounded to nearest integer with fixed width alignment
            d = int(round(wall_time)) if wall_time is not None else 0
            a = int(round(mean_area)) if mean_area is not None else 0
            ca = int(round(mean_clean_area)) if mean_clean_area is not None else 0
            ret = mean_return if mean_return is not None else 0.0

            text_lines = [
                f"E{args.experiment}Z{args.zone} | {agent_name}",
                f"Day:{d:>2} | Area:{a:>4} | Clean:{ca:>4} | Return:{ret:>8.2f}",
            ]

            for j, line in enumerate(text_lines):
                x, y = 50, 30 + j * 60
                draw.text((x, y), line, font=font, fill=(255, 255, 255))

            # Draw horizontal stacked bar for actions (R, W, B coefficients)
            bar_h = 40
            bar_max_w = 400
            padding = 50

            start_x = img.width - bar_max_w - padding
            start_y = 30

            # Title above bar
            draw.text(
                (start_x, start_y), "Actions (R/W/B)", font=font, fill=(255, 255, 255)
            )

            bar_y = start_y + 50
            # Background for the bar
            draw.rectangle(
                [start_x - 2, bar_y - 2, start_x + bar_max_w + 2, bar_y + bar_h + 2],
                fill=(60, 60, 60),
            )

            val_r = r if r is not None and not math.isnan(r) else 0.0
            val_w = w if w is not None and not math.isnan(w) else 0.0
            val_b = b if b is not None and not math.isnan(b) else 0.0

            # Scaling relative to a sum of 1.0 filling the entire bar
            wr = int(val_r * bar_max_w)
            ww = int(val_w * bar_max_w)
            wb = int(val_b * bar_max_w)

            # Ensure we don't overflow the bar visually if the sum > 1.0
            total_px = wr + ww + wb
            if total_px > bar_max_w:
                scale = bar_max_w / total_px
                wr = int(wr * scale)
                ww = int(ww * scale)
                wb = int(wb * scale)

            # Draw stacked segments
            curr_x = start_x
            if wr > 0:
                draw.rectangle(
                    [curr_x, bar_y, curr_x + wr, bar_y + bar_h], fill=(255, 50, 50)
                )
                curr_x += wr
            if ww > 0:
                draw.rectangle(
                    [curr_x, bar_y, curr_x + ww, bar_y + bar_h], fill=(240, 240, 240)
                )
                curr_x += ww
            if wb > 0:
                draw.rectangle(
                    [curr_x, bar_y, curr_x + wb, bar_y + bar_h], fill=(50, 50, 255)
                )

            frame_path = tmp_dir / f"frame_{i:05d}.jpg"
            img.save(frame_path)
        except Exception as e:
            logging.warning(f"Failed to process image {img_path}: {e}")
            continue

    # Run FFmpeg
    logging.info(f"Creating video: {args.output}")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.framerate),
        "-i",
        str(tmp_dir / "frame_%05d.jpg"),
        "-c:v",
        "mpeg4",
        "-q:v",
        "5",
        args.output,
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        logging.info(f"Timelapse saved to {args.output}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg failed: {e}")


if __name__ == "__main__":
    main()
