"""
Create a frame strip for E16: horizontal rows of raw tray images at 6-hour intervals
with stacked-area action coefficient plots underneath, grouped by agent.
"""

import argparse
import datetime
import io
import logging
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from PIL import Image, ImageDraw

matplotlib.use("Agg")

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import VERSION
from visualization.create_mosaic import load_font
from visualization.plot_e16_metrics import build_layout, load_episode_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

UTC = datetime.timezone.utc
IMAGE_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{6})\+0000_(left|right)\.jpg$")


def get_zone_info(df: pl.DataFrame, experiment: int) -> dict[int, dict]:
    """Extract per-zone root dir and ref_time (UTC) from parquet."""
    exp_df = df.filter(
        (pl.col("experiment") == experiment) & pl.col("image_path").is_not_null()
    )
    info = {}
    for zone in range(1, 13):
        z = exp_df.filter(pl.col("zone") == zone)
        if z.is_empty():
            continue
        # Get zone root directory
        img_path = z["image_path"][0]
        zone_root = Path(img_path.split("/processed/")[0])
        # Get ref_time: time at wall_time=0 (or closest), converted to UTC
        ref_row = z.sort((pl.col("wall_time")).abs()).head(1)
        ref_time_local = ref_row["time"][0]
        ref_time_utc = ref_time_local.astimezone(UTC)
        info[zone] = {"zone_root": zone_root, "ref_time_utc": ref_time_utc}
    return info


def _round_to_nearest_5min(ts_utc: datetime.datetime) -> datetime.datetime:
    """Round UTC datetime to the nearest 5-minute boundary."""
    ts_utc = ts_utc.astimezone(UTC)
    epoch = int(ts_utc.timestamp())
    rounded = int(round(epoch / 300.0) * 300)
    return datetime.datetime.fromtimestamp(rounded, tz=UTC)


def build_image_time_index(zone_root: Path) -> dict[datetime.datetime, Path]:
    """Index raw images by nearest 5-minute UTC timestamp; prefer left if both exist."""
    img_dir = zone_root / "images"
    idx: dict[datetime.datetime, Path] = {}
    if not img_dir.exists():
        return idx

    for p in sorted(img_dir.glob("*.jpg")):
        m = IMAGE_TS_RE.match(p.name)
        if not m:
            continue
        ts_raw = datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H%M%S").replace(
            tzinfo=UTC
        )
        ts_key = _round_to_nearest_5min(ts_raw)
        side = m.group(2)
        existing = idx.get(ts_key)
        if existing is None:
            idx[ts_key] = p
        elif side == "left" and existing.name.endswith("_right.jpg"):
            idx[ts_key] = p
    return idx


def backfilled_image_path(
    image_idx: dict[datetime.datetime, Path],
    target_utc: datetime.datetime,
    max_backfill_minutes: int = 15,
    step_minutes: int = 5,
) -> Path | None:
    """Forward-fill behavior: use latest image at or before target within tolerance."""
    base = _round_to_nearest_5min(target_utc)
    max_steps = max_backfill_minutes // step_minutes
    step = datetime.timedelta(minutes=step_minutes)
    for k in range(max_steps + 1):
        candidate = base - k * step
        path = image_idx.get(candidate)
        if path is not None:
            return path
    return None


def paste_thumbnail(
    canvas: Image.Image,
    img_path: Path,
    x: int,
    y: int,
    w: int,
    h: int,
):
    try:
        with Image.open(img_path) as img:
            img.thumbnail((w, h))
            cell = Image.new("RGB", (w, h), (30, 30, 30))
            cell.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
            canvas.paste(cell, (x, y))
    except Exception as e:
        logging.warning(f"Failed to load {img_path}: {e}")


def load_zone_actions(zone_root: Path, ref_time_utc: datetime.datetime) -> tuple[list, list, list, list]:
    """Load raw action coefficients at full 5-min resolution from raw CSVs (vectorized)."""
    from config import BLUE as BLUE_VEC
    from config import RED as RED_VEC
    from config import WHITE as WHITE_VEC

    csv_files = sorted(zone_root.glob("raw_*.csv"))
    if not csv_files:
        return [], [], [], []

    action_cols = ["action.0", "action.1", "action.2", "action.3", "action.4", "action.5"]
    dfs = []
    for f in csv_files:
        try:
            df = pl.read_csv(f, columns=["time"] + action_cols, try_parse_dates=True)
            dfs.append(df)
        except Exception as e:
            logging.warning(f"Could not read {f}: {e}")

    if not dfs:
        return [], [], [], []

    combined = (
        pl.concat(dfs)
        .with_columns(pl.col("time").dt.convert_time_zone("UTC"))
        .sort("time")
        .unique("time", keep="first")
    )

    # Compute wall_time as float days from ref_time
    ref_epoch_us = int(ref_time_utc.timestamp() * 1e6)
    wt_arr = (combined["time"].dt.epoch(time_unit="us") - ref_epoch_us) / (86400.0 * 1e6)

    # Build (N, 6) action matrix
    A = np.column_stack([combined[c].to_numpy().astype(float) for c in action_cols])

    # Remove zero-action rows (lights off)
    nonzero = A.sum(axis=1) > 0
    A = A[nonzero]
    wt_arr = wt_arr.to_numpy()[nonzero]

    if len(A) == 0:
        return [], [], [], []

    # Batch least-squares: basis (6,3), solve for all rows simultaneously
    basis = np.column_stack([RED_VEC, WHITE_VEC, BLUE_VEC])  # (6, 3)
    coefs, _, _, _ = np.linalg.lstsq(basis, A.T, rcond=None)  # (3, N)
    coefs = np.clip(coefs, 0.0, 1.0)
    coef_sum = coefs.sum(axis=0, keepdims=True)
    coef_sum[coef_sum == 0] = 1.0
    coefs /= coef_sum  # normalize rows to sum=1

    return wt_arr.tolist(), coefs[0].tolist(), coefs[1].tolist(), coefs[2].tolist()


NIGHT_FRAC = 11 / 24  # lights off (11h after ref)


def render_actions_strip(
    wt_raw: np.ndarray,
    red_raw: np.ndarray,
    white_raw: np.ndarray,
    blue_raw: np.ndarray,
    targets: list[float],
    interval: float,
    thumb: int,
    height_px: int,
    dpi: int = 100,
) -> Image.Image:
    """
    Step-function action strip: one bar per frame, aligned under each thumbnail.
    For each target wall_time, look up the nearest action sample.
    Nighttime frames (fractional >= NIGHT_FRAC) are black.
    """
    n = len(targets)
    width_px = n * thumb

    # Look up nearest action for each target
    r_vals, w_vals, b_vals, is_night = [], [], [], []
    for wt in targets:
        frac = wt % 1.0
        if frac >= NIGHT_FRAC:
            r_vals.append(0)
            w_vals.append(0)
            b_vals.append(0)
            is_night.append(True)
            continue
        is_night.append(False)
        if len(wt_raw) == 0:
            r_vals.append(0)
            w_vals.append(0)
            b_vals.append(1)
            continue
        idx = np.argmin(np.abs(wt_raw - wt))
        r_vals.append(float(red_raw[idx]))
        w_vals.append(float(white_raw[idx]))
        b_vals.append(float(blue_raw[idx]))

    half = interval / 2
    x_min = targets[0] - half
    x_max = targets[-1] + half

    fig_w = width_px / dpi
    fig_h = height_px / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    bar_w = interval * 0.999  # slight gap for visual separation

    for i, wt in enumerate(targets):
        x = wt - bar_w / 2
        if is_night[i]:
            ax.bar(x, 1.0, width=bar_w, align="edge", color="black", linewidth=0)
        else:
            r, w, b = r_vals[i], w_vals[i], b_vals[i]
            ax.bar(x, r,         width=bar_w, align="edge", bottom=0,       color="#e05050", linewidth=0)
            ax.bar(x, w,         width=bar_w, align="edge", bottom=r,       color="#aaaaaa", linewidth=0)
            ax.bar(x, b,         width=bar_w, align="edge", bottom=r + w,   color="#5070e0", linewidth=0)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="black")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((width_px, height_px))


def main():
    parser = argparse.ArgumentParser(
        description="Create a frame strip for E16 grouped by agent."
    )
    parser.add_argument(
        "--parquet", "-p",
        default=f"/data/plant-rl/offline/{VERSION}/mixed-{VERSION}.parquet",
    )
    parser.add_argument("--experiment", "-e", type=int, default=16)
    parser.add_argument("--out", "-o", default="results/e16_frame_strip.jpg")
    parser.add_argument("--thumb-size", type=int, default=120)
    parser.add_argument("--plot-height", type=int, default=80)
    parser.add_argument("--label-width", type=int, default=160)
    parser.add_argument("--interval", type=float, default=1/3, help="8h = 1/3 days")
    parser.add_argument("--max-day", type=float, default=13.0)
    parser.add_argument(
        "--image-backfill-minutes",
        type=int,
        default=15,
        help="Forward-fill image lookup tolerance in minutes (past only).",
    )
    args = parser.parse_args()

    logging.info(f"Reading parquet: {args.parquet}")
    df = pl.read_parquet(args.parquet)

    # Build agent order and zone map from data
    pdf = load_episode_metrics(Path(args.parquet), args.experiment)
    agent_order, zone_map, _, _, _ = build_layout(pdf)
    logging.info(f"Agents: {agent_order}, Zone map: {zone_map}")

    zone_info = get_zone_info(df, args.experiment)
    logging.info(f"Found info for zones: {sorted(zone_info)}")

    # Target wall_times: 0.0, interval, 2*interval, ... + Day 14 9:30 AM only
    frames_per_day = round(1.0 / args.interval)
    targets: list[float] = []
    for d in range(int(args.max_day) + 1):
        for k in range(frames_per_day):
            targets.append(round(d + k / frames_per_day, 6))
    targets.append(int(args.max_day) + 1.0)  # Day 14 9:30 AM
    n_frames = len(targets)

    thumb = args.thumb_size
    plot_h = args.plot_height
    label_w = args.label_width
    header_h = 40
    strip_w = n_frames * thumb
    zone_row_h = thumb + plot_h
    n_zones = sum(len(z) for z in zone_map.values())

    canvas_w = label_w + strip_w
    canvas_h = header_h + n_zones * zone_row_h

    logging.info(f"Canvas: {canvas_w}x{canvas_h}, {n_frames} frames/zone")

    bg = (255, 255, 255)
    sep_color = (200, 200, 200)
    text_color = (0, 0, 0)
    subtext_color = (80, 80, 80)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg)
    draw = ImageDraw.Draw(canvas)
    font_header = load_font(14)
    font_agent = load_font(18)
    font_zone = load_font(14)

    # Day headers (0..14)
    for day in range(int(args.max_day) + 2):
        cx = label_w + day * frames_per_day * thumb + (frames_per_day * thumb) // 2
        draw.text((cx, header_h // 2), f"Day {day}", font=font_header, fill=text_color, anchor="mm")
        x_sep = label_w + day * frames_per_day * thumb
        draw.line([(x_sep, 0), (x_sep, canvas_h)], fill=sep_color)

    row_idx = 0
    for agent in agent_order:
        zones = zone_map[agent]
        agent_start_y = header_h + row_idx * zone_row_h

        for zone in zones:
            y = header_h + row_idx * zone_row_h

            draw.text(
                (label_w - 10, y + thumb // 2),
                f"Z{zone}",
                font=font_zone,
                fill=subtext_color,
                anchor="rm",
            )

            info = zone_info.get(zone)
            if info is None:
                row_idx += 1
                continue

            zone_root = info["zone_root"]
            ref_utc = info["ref_time_utc"]
            image_idx = build_image_time_index(zone_root)

            # Paste raw frames
            for fi, wt in enumerate(targets):
                x = label_w + fi * thumb
                target_utc = ref_utc + datetime.timedelta(days=wt)
                img_path = backfilled_image_path(
                    image_idx,
                    target_utc,
                    max_backfill_minutes=args.image_backfill_minutes,
                )
                if img_path is not None:
                    paste_thumbnail(canvas, img_path, x, y, thumb, thumb)

            # Action coefficient step-function strip (one bar per frame)
            wts, reds, whs, blues = load_zone_actions(zone_root, ref_utc)
            wts_arr = np.array(wts)
            reds_arr = np.array(reds)
            whs_arr = np.array(whs)
            blues_arr = np.array(blues)
            plot_img = render_actions_strip(
                wts_arr, reds_arr, whs_arr, blues_arr,
                targets=targets,
                interval=args.interval,
                thumb=thumb,
                height_px=plot_h,
            )
            canvas.paste(plot_img, (label_w, y + thumb))

            draw.line([(0, y + zone_row_h), (canvas_w, y + zone_row_h)], fill=sep_color)
            row_idx += 1

        # Agent label
        agent_end_y = header_h + row_idx * zone_row_h
        mid_y = (agent_start_y + agent_end_y) // 2
        draw.multiline_text(
            (10, mid_y),
            agent.replace("_", "\n"),
            font=font_agent,
            fill=text_color,
            anchor="lm",
        )
        draw.line([(0, agent_end_y), (canvas_w, agent_end_y)], fill=(150, 150, 150), width=2)

    draw.line([(label_w, 0), (label_w, canvas_h)], fill=sep_color, width=2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    logging.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
