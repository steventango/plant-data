import argparse
import datetime
import io
import logging
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")

from config import VERSION

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_font(size: int = 20):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def get_viz_path(image_path: str, experiment: int, zone: int) -> Path:
    """Convert an individual plant image path to the tray visualization path."""
    p = Path(image_path)
    timestamp = p.stem
    viz_dir = p.parent.parent.parent / "visualizations"
    viz_filename = f"E{experiment}_Z{zone}_{timestamp}_viz.jpg"
    return viz_dir / viz_filename


def get_raw_path(image_path: str, time_val) -> Path:
    """Convert a processed image path + time to the raw tray image path (UTC-named)."""
    p = Path(image_path)
    # Go from .../processed/vXX/images/N/timestamp.jpg to .../images/
    if "/processed/" in str(p):
        zone_root = Path(str(p).split("/processed/")[0])
    else:
        zone_root = p.parent.parent
    raw_dir = zone_root / "images"
    utc_time = time_val.astimezone(datetime.timezone.utc)
    timestamp_utc = utc_time.strftime("%Y-%m-%dT%H%M%S")
    return raw_dir / f"{timestamp_utc}+0000_left.jpg"


def resolve_image(image_path: str, experiment: int, zone: int, time_val) -> Path | None:
    """Try to find the best available image: viz → processed → raw."""
    if image_path is None:
        return None

    # 1. Try tray visualization
    viz_path = get_viz_path(image_path, experiment, zone)
    if viz_path.exists():
        return viz_path

    # 2. Try processed image as-is
    processed = Path(image_path)
    if processed.exists():
        return processed

    # 3. Try raw tray image (UTC-named)
    if time_val is not None:
        raw = get_raw_path(image_path, time_val)
        if raw.exists():
            return raw

    return None


def render_actions_plot(
    wall_times: list[float],
    red_coefs: list[float],
    white_coefs: list[float],
    blue_coefs: list[float],
    width_px: int,
    height_px: int,
    initial_day: float,
    final_day: float,
    max_day: float | None = None,
    dpi: int = 100,
) -> Image.Image:
    """Render a stacked area chart of R/W/B light coefficients to a PIL Image."""
    fig_w = width_px / dpi
    fig_h = height_px / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    wt = np.array(wall_times)
    r = np.array(red_coefs, dtype=float)
    w = np.array(white_coefs, dtype=float)
    b = np.array(blue_coefs, dtype=float)

    # Crop to max_day
    if max_day is not None:
        mask = wt <= max_day
        wt, r, w, b = wt[mask], r[mask], w[mask], b[mask]

    # Replace NaN with 0
    r = np.nan_to_num(r)
    w = np.nan_to_num(w)
    b = np.nan_to_num(b)

    ax.stackplot(
        wt,
        r, w, b,
        labels=["Red", "White", "Blue"],
        colors=["#e05050", "#aaaaaa", "#5070e0"],
        alpha=0.85,
    )

    # Mark initial and final day
    for day, ls in ((initial_day, "--"), (final_day, ":")):
        ax.axvline(day, color="black", linewidth=1.0, linestyle=ls, alpha=0.6)

    x_max = max_day if max_day is not None else (wt.max() if len(wt) > 1 else wt.min() + 1)
    ax.set_xlim(wt.min() if len(wt) > 0 else 0, x_max)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Day", fontsize=8)
    ax.set_ylabel("Coefficient", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    parser = argparse.ArgumentParser(
        description="Create a mosaic of initial and final frames grouped by agent and zone."
    )
    parser.add_argument(
        "--parquet",
        "-p",
        default=f"/data/plant-rl/offline/{VERSION}/mixed-{VERSION}.parquet",
        help="Path to parquet file",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=int,
        default=None,
        help="Filter to a single experiment (e.g. 16)",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="results/mosaic.jpg",
        help="Output filename for the mosaic",
    )
    parser.add_argument(
        "--thumb-width",
        type=int,
        default=400,
        help="Width of each thumbnail image (default: 400)",
    )
    parser.add_argument(
        "--thumb-height",
        type=int,
        default=400,
        help="Height of each thumbnail image (default: 400)",
    )
    parser.add_argument(
        "--label-width",
        type=int,
        default=260,
        help="Width of the label column on the left (default: 260)",
    )
    parser.add_argument(
        "--initial-day",
        type=float,
        default=0.0,
        help="Wall time (day) for the initial frame (default: 0)",
    )
    parser.add_argument(
        "--final-day",
        type=float,
        default=14.0,
        help="Wall time (day) for the final frame (default: 14)",
    )
    parser.add_argument(
        "--plot-width",
        type=int,
        default=400,
        help="Width of the actions area plot column (default: 400)",
    )

    args = parser.parse_args()

    # 1. Read Parquet
    logging.info(f"Reading parquet: {args.parquet}")
    df = pl.read_parquet(args.parquet)

    if args.experiment is not None:
        df = df.filter(pl.col("experiment") == args.experiment)
        logging.info(f"Filtered to experiment {args.experiment}: {len(df)} rows")

    # 2. For each (experiment, zone), get frames at the specified days
    initial_day = args.initial_day
    final_day = args.final_day

    base = df.filter(pl.col("image_path").is_not_null())

    # Get the closest wall_time to the requested days per zone
    # For initial: pick the row with wall_time closest to initial_day
    # For final: pick the row with wall_time closest to final_day
    initial_df = (
        base.with_columns((pl.col("wall_time") - initial_day).abs().alias("_dist"))
        .sort("_dist")
        .group_by(["experiment", "zone"])
        .first()
        .select(
            "experiment",
            "zone",
            pl.col("agent"),
            pl.col("image_path").alias("first_image_path"),
            pl.col("time").alias("first_time"),
            pl.col("wall_time").alias("first_wall_time"),
        )
    )

    final_df = (
        base.with_columns((pl.col("wall_time") - final_day).abs().alias("_dist"))
        .sort("_dist")
        .group_by(["experiment", "zone"])
        .first()
        .select(
            "experiment",
            "zone",
            pl.col("image_path").alias("last_image_path"),
            pl.col("time").alias("last_time"),
            pl.col("wall_time").alias("last_wall_time"),
        )
    )

    zone_summary = (
        initial_df.join(final_df, on=["experiment", "zone"])
        .sort(["agent", "experiment", "zone"])
    )

    # 2b. Collect per-zone action time series (all days, sorted by wall_time)
    actions_by_zone: dict[tuple[int, int], dict] = {}
    for (exp, zone), grp in (
        df.filter(pl.col("wall_time").is_not_null())
        .sort("wall_time")
        .group_by(["experiment", "zone"])
    ):
        actions_by_zone[(exp, zone)] = {
            "wall_times": grp["wall_time"].to_list(),
            "red": grp["red_coef"].to_list(),
            "white": grp["white_coef"].to_list(),
            "blue": grp["blue_coef"].to_list(),
        }

    logging.info(f"Found {len(zone_summary)} (experiment, zone) groups")

    # 3. Determine grid layout
    rows = zone_summary.to_dicts()

    thumb_w = args.thumb_width
    thumb_h = args.thumb_height
    label_w = args.label_width
    header_h = 50

    # Group rows by agent for drawing agent group labels
    agent_groups = {}
    for row in rows:
        agent = row["agent"]
        if agent not in agent_groups:
            agent_groups[agent] = []
        agent_groups[agent].append(row)

    n_zones = len(rows)
    plot_w = args.plot_width
    canvas_width = label_w + 2 * thumb_w + plot_w
    canvas_height = header_h + n_zones * thumb_h

    logging.info(
        f"Creating mosaic: {canvas_width}x{canvas_height} ({n_zones} zones, {len(agent_groups)} agents)"
    )

    # Colors (black-on-white scheme)
    bg_color = (255, 255, 255)
    text_color = (0, 0, 0)
    subtext_color = (80, 80, 80)
    sep_color = (200, 200, 200)

    # 4. Create canvas
    mosaic = Image.new("RGB", (canvas_width, canvas_height), color=bg_color)
    draw = ImageDraw.Draw(mosaic)

    font_large = load_font(22)
    font_small = load_font(16)
    font_header = load_font(24)

    # 5. Draw column headers
    header_y = header_h // 2
    for label, cx in (
        ("Initial", label_w + thumb_w // 2),
        ("Final", label_w + thumb_w + thumb_w // 2),
        ("Actions (R/W/B)", label_w + 2 * thumb_w + plot_w // 2),
    ):
        draw.text((cx, header_y), label, font=font_header, fill=text_color, anchor="mm")

    # 6. Draw each row
    resample = getattr(Image, "Resampling", Image).LANCZOS
    y_offset = header_h
    current_agent = None
    agent_start_y = y_offset

    for i, row in enumerate(rows):
        experiment = row["experiment"]
        zone = row["zone"]
        agent = row["agent"]
        first_img_path = row["first_image_path"]
        last_img_path = row["last_image_path"]
        first_time = row["first_time"]
        last_time = row["last_time"]
        first_wt = row["first_wall_time"]
        last_wt = row["last_wall_time"]

        y = y_offset + i * thumb_h

        # Track agent grouping for bracket/label
        if agent != current_agent:
            if current_agent is not None:
                _draw_agent_label(
                    draw, current_agent, agent_start_y, y, label_w, font_large, text_color, sep_color
                )
            current_agent = agent
            agent_start_y = y

        # Draw zone label on the right side of label area
        zone_label = f"E{experiment}Z{zone}"
        draw.text(
            (label_w - 10, y + thumb_h // 2),
            zone_label,
            font=font_small,
            fill=subtext_color,
            anchor="rm",
        )

        # Resolve and draw initial frame
        first_resolved = resolve_image(first_img_path, experiment, zone, first_time)
        if first_resolved:
            _paste_thumbnail(mosaic, first_resolved, label_w, y, thumb_w, thumb_h, resample)
        else:
            logging.warning(f"No image found for E{experiment}Z{zone} initial frame")

        # Resolve and draw final frame
        last_resolved = resolve_image(last_img_path, experiment, zone, last_time)
        if last_resolved:
            _paste_thumbnail(mosaic, last_resolved, label_w + thumb_w, y, thumb_w, thumb_h, resample)
        else:
            logging.warning(f"No image found for E{experiment}Z{zone} final frame")

        # Draw day labels on the images
        day_font = load_font(14)
        first_day = int(round(first_wt)) if first_wt is not None else "?"
        last_day = int(round(last_wt)) if last_wt is not None else "?"
        draw.text(
            (label_w + thumb_w - 5, y + thumb_h - 5),
            f"Day {first_day}",
            font=day_font,
            fill=(255, 255, 255),
            anchor="rb",
        )
        draw.text(
            (label_w + 2 * thumb_w - 5, y + thumb_h - 5),
            f"Day {last_day}",
            font=day_font,
            fill=(255, 255, 255),
            anchor="rb",
        )

        # Render and paste actions area plot
        zone_actions = actions_by_zone.get((experiment, zone))
        if zone_actions:
            plot_img = render_actions_plot(
                zone_actions["wall_times"],
                zone_actions["red"],
                zone_actions["white"],
                zone_actions["blue"],
                width_px=plot_w,
                height_px=thumb_h,
                initial_day=initial_day,
                final_day=final_day,
                max_day=final_day - 1,
            )
            mosaic.paste(plot_img, (label_w + 2 * thumb_w, y))

        # Horizontal separator
        draw.line([(0, y + thumb_h), (canvas_width, y + thumb_h)], fill=sep_color)

    # Draw last agent group label
    if current_agent is not None:
        _draw_agent_label(
            draw, current_agent, agent_start_y, y_offset + n_zones * thumb_h, label_w, font_large, text_color, sep_color
        )

    # Vertical separators
    for vx in (label_w, label_w + thumb_w, label_w + 2 * thumb_w):
        draw.line([(vx, 0), (vx, canvas_height)], fill=sep_color, width=2)

    # 7. Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(out_path, quality=92)
    logging.info(f"Mosaic saved to {out_path}")


def _draw_agent_label(
    draw: ImageDraw.ImageDraw,
    agent: str,
    y_start: int,
    y_end: int,
    label_w: int,
    font,
    text_color: tuple = (0, 0, 0),
    sep_color: tuple = (200, 200, 200),
):
    """Draw the agent name centered vertically in the label column for its group of zones."""
    mid_y = (y_start + y_end) // 2
    display_name = agent.replace("_", "\n")
    draw.multiline_text(
        (10, mid_y),
        display_name,
        font=font,
        fill=text_color,
        anchor="lm",
    )
    draw.line(
        [(0, y_start), (label_w - 15, y_start)],
        fill=sep_color,
        width=2,
    )


def _paste_thumbnail(
    canvas: Image.Image,
    img_path: Path,
    x: int,
    y: int,
    thumb_w: int,
    thumb_h: int,
    resample,
):
    """Load an image, fit within thumb bounds (letterbox), and paste onto canvas."""
    try:
        with Image.open(img_path) as img:
            img_ratio = img.width / img.height
            thumb_ratio = thumb_w / thumb_h

            # Fit inside (letterbox/pillarbox) — never crop
            if img_ratio > thumb_ratio:
                new_width = thumb_w
                new_height = int(thumb_w / img_ratio)
            else:
                new_height = thumb_h
                new_width = int(thumb_h * img_ratio)

            img = img.resize((new_width, new_height), resample=resample)

            # Center on a white cell
            cell = Image.new("RGB", (thumb_w, thumb_h), (255, 255, 255))
            paste_x = (thumb_w - new_width) // 2
            paste_y = (thumb_h - new_height) // 2
            cell.paste(img, (paste_x, paste_y))

            canvas.paste(cell, (x, y))
    except Exception as e:
        logging.warning(f"Failed to process image {img_path}: {e}")


if __name__ == "__main__":
    main()
