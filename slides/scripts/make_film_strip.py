"""
Build a comparison composite for one or more E17 zones:

  - Top: 3 (or N) raw camera frames per zone (no labels)
  - Plant area panel: per-plant trajectories + per-zone mean overlaid
  - Reward panel:     per-plant trajectories + per-zone mean overlaid
  - Action panel(s):  one stacked R/W/B/night step-function per zone, shifted
                      RIGHT by one day so action_t aligns with the area / reward
                      it produced (the colour the audience sees here is the
                      colour responsible for the growth above).

Source of truth for frame timestamps: the `image_name` column of each zone's
per-zone parquet.

Usage:
    python make_film_strip.py 7              # single zone
    python make_film_strip.py 6,7 0,7,14     # two zones, specific days
"""

import subprocess
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from PIL import Image

# ── CLI ─────────────────────────────────────────────────────────────────
ZONES = (
    [int(z) for z in sys.argv[1].split(",")]
    if len(sys.argv) > 1
    else [7]
)
DAYS = (
    [int(d) for d in sys.argv[2].split(",")]
    if len(sys.argv) > 2
    else None
)

# ── Layout / style ──────────────────────────────────────────────────────
TILE_W  = 600
TILE_H  = 450
GAP     = 6
LABEL_W = 160  # left zone-colored label tile
BG     = "#FFFFFF"
INK    = "#111111"
MUTED  = "#9A9A93"
GRID   = "#E6E5DE"
RED    = "#D04A3C"
BLUE   = "#3D5CC8"
WHITE_BAR = "#B0AFA8"

# Per-zone overlay colours (colour-blind safe pink + green).
ZONE_COLORS = {
    7:  "#30A667",   # green (BlueRed)
    6:  "#CF50A0",   # pink  (RedBlue)
    9:  "#3D5CC8",   # blue
    10: "#A85FB2",   # purple
}
ZONE_LABELS = {
    6:  r"$\pi_A$",
    7:  r"$\pi_B$",
    9:  r"$\pi_C$",
    10: r"$\pi_D$",
}
# Per-zone source-image quad (4 corners, TL→TR→BR→BL) bounding a 3×3 pot
# grid in the top-left of the chamber frame. The strip cells warp this quad
# to a square. Camera mounts differ between zones — these are eyeballed from
# /tmp/z6_grid_overlay.jpg and /tmp/z7_grid_overlay.jpg.
ZONE_SRC_QUAD = {
    6: [(195, 310), (710, 260), (680, 863), (125, 883)],
    7: [(135, 225), (685, 125), (705, 693), (125, 783)],
}

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "photos"
ROOT = Path("/data/plant-rl/online/E17/P1")


def zone_paths(zone: int):
    zdir = next(ROOT.glob(f"*/alliance-zone{zone:02d}"))
    return {
        "dir":     zdir,
        "hourly":  zdir / "processed" / "v27" / f"E17_Z{zone}_hourly.parquet",
        "daily":   zdir / "processed" / "v27" / f"E17_Z{zone}.parquet",
        "viz_dir": zdir / "processed" / "v27" / "visualizations",
        "images":  zdir / "images",
    }


def colour_for(zone: int) -> str:
    return ZONE_COLORS.get(zone, f"C{zone % 10}")


# ── Frame strip per zone ────────────────────────────────────────────────
def daily_frames(zone: int) -> list[tuple[int, date, str]]:
    p = zone_paths(zone)
    parquet = p["hourly"] if p["hourly"].exists() else p["daily"]
    df = (
        pl.read_parquet(parquet, columns=["time", "plant_id", "image_name", "day"])
        .unique(subset=["time"])
        .sort("time")
    )
    out = []
    for d, t, name in zip(df["day"], df["time"], df["image_name"]):
        # day 0 9:30 happens before the hourly grid; daily parquet still emits it.
        # The per-day "first observation at 09:30 local" is what we want.
        out.append((int(d), t.date(), name))
    # Deduplicate by day, keep first (earliest) timestamp per day.
    seen = set()
    deduped = []
    for d, dte, n in out:
        if d in seen:
            continue
        seen.add(d)
        deduped.append((d, dte, n))
    return deduped


def build_strip(zone: int, frames, use_viz: bool, out: Path):
    """Photo strip. RAW frames are taken at +5 min (T003500 instead of T003000)
    so the experimental R/W/B light is visible — the 9:30 daily observation
    itself is captured under a neutral white flash, where every zone looks
    the same.  Viz frames stay on the 9:30 timestamp (that's what the pipeline
    annotates).

    Prepended with a zone-colored label tile so each strip row is visually
    linked to the matching trajectory colour in the feature panels below.
    """
    p = zone_paths(zone)
    tile_dir = Path(f"/tmp/zone_strip_tiles_z{zone}")
    tile_dir.mkdir(parents=True, exist_ok=True)
    for f in tile_dir.glob("*.jpg"):
        f.unlink()

    # Build a label tile via matplotlib (so LaTeX π subscript renders cleanly).
    label_tile = tile_dir / "label.jpg"
    dpi = 150
    fig = plt.figure(figsize=(LABEL_W / dpi, TILE_H / dpi), dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_facecolor(BG); ax.set_axis_off()
    ax.text(0.5, 0.5, ZONE_LABELS.get(zone, f"Z{zone}"),
            color=colour_for(zone), fontsize=96,
            ha="center", va="center", transform=ax.transAxes)
    fig.savefig(label_tile, dpi=dpi, facecolor=BG, pad_inches=0)
    plt.close(fig)

    tile_paths = []
    for day, d, name in frames:
        if use_viz:
            src = p["viz_dir"] / f"E17_Z{zone}_{d:%Y-%m-%d}T093000_viz.jpg"
        else:
            # Shift to the +5 min frame so coloured lights are visible.
            shifted = name.replace("T003000", "T003500")
            src = p["images"] / shifted
            if not src.exists():
                # Fall back to the original 9:30 frame.
                src = p["images"] / name
        if not src.exists():
            print(f"  Z{zone} day {day}: missing {src}")
            continue
        dst = tile_dir / f"day{day:02d}.jpg"
        subprocess.run(
            ["convert", str(src), "-resize", f"{TILE_W}x{TILE_H}^",
             "-gravity", "center", "-extent", f"{TILE_W}x{TILE_H}",
             "-quality", "92", str(dst)],
            check=True,
        )
        tile_paths.append(dst)

    if not tile_paths:
        raise RuntimeError(f"Z{zone}: no tiles produced")

    cmd = ["convert", str(label_tile)]
    for tp in tile_paths:
        cmd += ["-size", f"{GAP}x{TILE_H}", f"xc:{BG}", str(tp)]
    cmd += ["+append", "-quality", "92", str(out)]
    subprocess.run(cmd, check=True)
    print(f"saved -> {out}  (Z{zone}, {len(tile_paths)} tiles)")
    return out


# ── Helpers for the features panel ──────────────────────────────────────
GAP_DAYS = 0.2  # night-gap threshold

def break_at_gaps(x: np.ndarray, y: np.ndarray):
    """Insert NaNs at time gaps > GAP_DAYS so matplotlib breaks the line."""
    if len(x) == 0:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    dx = np.diff(x)
    breaks = np.where(dx > GAP_DAYS)[0]
    if len(breaks) == 0:
        return x, y
    xs, ys = [], []
    start = 0
    for b in breaks:
        xs.append(x[start : b + 1]); xs.append(np.array([np.nan]))
        ys.append(y[start : b + 1]); ys.append(np.array([np.nan]))
        start = b + 1
    xs.append(x[start:]); ys.append(y[start:])
    return np.concatenate(xs), np.concatenate(ys)


def drop_area_spikes(y: np.ndarray, window: int = 7, k: float = 3.0):
    """Asymmetric MAD spike filter — only nulls upward spikes."""
    if len(y) < window:
        return y
    from numpy.lib.stride_tricks import sliding_window_view
    pad = window // 2
    padded = np.pad(y, pad, mode="edge")
    win = sliding_window_view(padded, window)
    med = np.nanmedian(win, axis=1)
    mad = np.nanmedian(np.abs(win - med[:, None]), axis=1)
    mad = np.where(mad < 1e-6, 1.0, mad)
    spike = (y - med) > k * 1.4826 * mad
    spike |= y > 30.0
    return np.where(spike, np.nan, y)


def load_zone_data(zone: int) -> pl.DataFrame:
    """Return the hourly parquet for `zone`, or fall back to daily.

    Uses `area_after_tukey` for the per-step area (frame-level MAD outlier
    removal — see plant-cv/api/pipeline/app/analysis.py) and exposes it as
    `clean_area`. This deliberately skips the EWM-based cleaning, which
    was tuned for the daily monotonic-growth setting and clips legitimate
    within-day downward moves (parabolic turgor cycle).
    """
    p = zone_paths(zone)
    src = p["hourly"] if p["hourly"].exists() else p["daily"]
    print(f"  Z{zone}: loading {src.name}")
    df = pl.read_parquet(
        src,
        columns=["plant_id", "wall_time", "day", "clean_area",
                 "red_coef", "white_coef", "blue_coef", "reward"],
    )
    # Drop the partial day-14 trailing data (one stray point would render as
    # an isolated dot at the right edge of the lineplots).
    return df.filter(pl.col("wall_time") < 14.0)


def _find_coeffs(src_quad, dst_quad):
    """Solve for PIL perspective transform coefficients (output→source)."""
    A, b = [], []
    for (sx, sy), (dx, dy) in zip(src_quad, dst_quad):
        A.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        A.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
        b.append(sx); b.append(sy)
    return tuple(np.linalg.solve(np.asarray(A, float),
                                 np.asarray(b, float)).tolist())


def load_warped(image_path: Path, src_quad, side: int = 800) -> np.ndarray:
    """Perspective-warp a 4-corner quad from the source image into a `side`×
    `side` square. `src_quad` is TL, TR, BR, BL in source pixel coords."""
    img = Image.open(image_path).convert("RGB")
    dst_quad = [(0, 0), (side, 0), (side, side), (0, side)]
    coeffs = _find_coeffs(src_quad, dst_quad)
    warped = img.transform(
        (side, side), Image.Transform.PERSPECTIVE, coeffs,
        Image.Resampling.BICUBIC,
    )
    return np.asarray(warped)


# Back-compat shim used by the standalone strip JPGs (still want a quick
# crop there since they don't have per-zone calibration info handy).
def load_square_left(image_path: Path, zoom: float = 0.265,
                     x_offset_frac: float = 0.02,
                     y_offset_frac: float = 0.0) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    side = int(min(w, h) * zoom)
    x0 = max(0, int(w * x_offset_frac))
    y0 = max(0, int(h * y_offset_frac))
    img = img.crop((x0, y0, x0 + side, y0 + side))
    return np.asarray(img)


def shifted_image_path(zone: int, image_name: str) -> Path:
    """Per-day raw frame at +5 min so the experimental colour is visible."""
    p = zone_paths(zone)
    shifted = image_name.replace("T003000", "T003500")
    cand = p["images"] / shifted
    return cand if cand.exists() else p["images"] / image_name


# ── Features panel: strip(s) + area + reward + per-zone action ──────────
def build_features_panel(width_px: int, out: Path):
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 1.0,
        "axes.labelcolor": INK,
        "axes.titlesize": 22,
        "axes.titleweight": "regular",
        "axes.labelsize": 18,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "font.family": ["Inter", "Liberation Sans", "DejaVu Sans", "sans-serif"],
        "font.size": 16,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 14,
    })

    # Load all zones
    data = {z: load_zone_data(z) for z in ZONES}
    actions_by_zone = {
        z: (
            d.unique(subset=["day"]).sort("day")
             .select(["day", "red_coef", "white_coef", "blue_coef"])
             .to_pandas()
        )
        for z, d in data.items()
    }

    # Layout: one strip row per zone, then state / reward / action.
    # Whole figure is locked to 16:9 so it drops cleanly onto a slide.
    n_strip = len(ZONES)
    heights = [0.55] * n_strip + [1.3, 0.9]
    n_rows  = n_strip + 2

    dpi = 150
    fig_w_in = width_px / dpi
    fig_h_in = fig_w_in * 9.0 / 16.0
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(fig_w_in, fig_h_in), dpi=dpi, sharex=True,
        gridspec_kw={"height_ratios": heights, "hspace": 0.18},
    )
    strip_axes = list(axes[:n_strip])
    ax_area, ax_rew = axes[n_strip], axes[n_strip + 1]
    ax_act = None

    # ── Strip rows — square photos, one per day, at x=[d, d+1]
    for ax, z in zip(strip_axes, ZONES):
        frames = daily_frames(z)
        # Only days that fit on the [0, 14] axis; trim partial day 14.
        frames = [(d, dte, n) for d, dte, n in frames if d < 14]
        src_quad = ZONE_SRC_QUAD.get(z)
        for day, _, name in frames:
            try:
                src_img = shifted_image_path(z, name)
                if src_quad is not None:
                    img = load_warped(src_img, src_quad, side=800)
                else:
                    img = load_square_left(src_img)
            except Exception as e:
                print(f"  Z{z} day {day}: image load failed ({e})")
                continue
            ax.imshow(img, extent=(day, day + 1, 0, 1),
                      aspect="auto", interpolation="bilinear")
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis="both", which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylabel(
            ZONE_LABELS.get(z, f"Z{z}"),
            color=colour_for(z), fontsize=34, rotation=0,
            ha="right", va="center", labelpad=18,
        )

    # Align the day band to the hourly observation window (9:30–19:30 local =
    # 10 h, wall_time d to d + 0.4167). The chamber's true photoperiod is
    # 12 h (9:00–21:00) but observations bracket those edges, so anchoring
    # to the data avoids the "night starts after the last point" gap.
    LIGHTS_ON_OFFSET = 0.0
    PHOTOPERIOD       = 10.0 / 24.0   # ≈ 0.4167
    N_DAYS            = 15

    # Grey night-band overlays + no gridlines on every panel.
    panels_to_shade = [ax_area, ax_rew] + (
        [ax_act] if ax_act is not None else []
    )
    for ax in panels_to_shade:
        for d in range(N_DAYS):
            night_start = d + LIGHTS_ON_OFFSET + PHOTOPERIOD
            night_end   = d + 1 + LIGHTS_ON_OFFSET
            ax.axvspan(night_start, night_end,
                       color=MUTED, alpha=0.12, linewidth=0)
        ax.grid(False)

    # ── Plant area — uses parquet's post-pipeline `clean_area` (EWM-cleaned
    # by plant-cv with the patched downward threshold). No extra outlier
    # filtering in the plot — the pipeline handles it.
    # We still exclude "dead" plants (final area below threshold) and
    # restrict to plants with full timestamp coverage; both are sample
    # selection, not outlier filtering.
    DEAD_FINAL_THRESHOLD_CM2 = 1.5
    # Cache the fixed-cohort plant sets per zone so the reward panel can
    # reuse them (avoids sample-composition drift in mean_area).
    zone_cohort: dict[int, set] = {}
    for z in ZONES:
        df = data[z]
        pdf = df.filter(pl.col("clean_area").is_not_null()).to_pandas()
        # Alive by final-day area.
        finals = (
            pdf.sort_values("wall_time")
               .groupby("plant_id")["clean_area"]
               .last()
        )
        alive = set(finals[finals >= DEAD_FINAL_THRESHOLD_CM2].index)
        # Require plants present at every observed timestamp — fixes the
        # composition jitter where new plants joining the mean would jerk
        # the average around day 1.
        all_ts = pdf["wall_time"].round(3).unique()
        n_ts = len(all_ts)
        per_pid = pdf.assign(_t=pdf["wall_time"].round(3)) \
                     .groupby("plant_id")["_t"].nunique()
        full_coverage = set(per_pid[per_pid == n_ts].index)
        cohort = alive & full_coverage
        zone_cohort[z] = cohort
        pdf = pdf[pdf["plant_id"].isin(cohort)]
        series_by_t: dict[float, list[float]] = {}
        for pid, g in pdf.groupby("plant_id"):
            x = g["wall_time"].to_numpy()
            y = g["clean_area"].to_numpy()
            order = np.argsort(x); x, y = x[order], y[order]
            for xi, yi in zip(x, y):
                if np.isfinite(yi):
                    series_by_t.setdefault(round(float(xi), 3), []).append(float(yi))
            xx, yy = break_at_gaps(x, y)
            ax_area.plot(xx, yy, color=colour_for(z), alpha=0.20, linewidth=0.5)
        ts = np.array(sorted(series_by_t.keys()))
        means = np.array([np.mean(series_by_t[t]) for t in ts])
        mx, my = break_at_gaps(ts, means)
        ax_area.plot(mx, my, color=colour_for(z), linewidth=2.4, alpha=0.95)
        print(f"  Z{z}: cohort {len(cohort)} plants (alive ∩ full-coverage)")

    ax_area.set_ylabel("Leaf\nArea\n(cm²)", rotation=0, ha="right",
                       va="center", labelpad=18, fontsize=22)
    ax_area.set_xlim(0, 14)
    ax_area.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True))

    # ── Reward — option 1: mean over per-plant rewards (from parquet's
    # `reward` column which is log(area_t) - log(area_{t-1}) per plant).
    # Averaging happens AFTER the log-diff so each plant contributes its own
    # per-step reward to the mean.
    rew_means_for_scale: list[float] = []
    zone_return: list[tuple[int, float]] = []
    for z in ZONES:
        df = data[z]
        if "reward" not in df.columns:
            continue
        cohort = zone_cohort.get(z, set())
        pdf = df.to_pandas()
        pdf = pdf[pdf["plant_id"].isin(cohort)]
        rewards_by_t: dict[float, list[float]] = {}
        for pid, g in pdf.groupby("plant_id"):
            x = g["wall_time"].to_numpy()
            y = g["reward"].to_numpy(dtype=float)
            order = np.argsort(x); x, y = x[order], y[order]
            for xi, yi in zip(x, y):
                if np.isfinite(yi):
                    rewards_by_t.setdefault(round(float(xi), 3), []).append(float(yi))
        rts = np.array(sorted(rewards_by_t.keys()))
        rmean = np.array([np.mean(rewards_by_t[t]) for t in rts])
        rx, ry = break_at_gaps(rts, rmean)
        ax_rew.plot(rx, ry, color=colour_for(z), linewidth=2.2, alpha=0.75)
        rew_means_for_scale.extend(rmean[np.isfinite(rmean)].tolist())

        # Total return = Σ mean rewards = mean of per-plant log-growth ratios.
        finite = rmean[np.isfinite(rmean)]
        if finite.size:
            ret = float(finite.sum())
            zone_return.append((z, ret))

    if rew_means_for_scale:
        arr = np.asarray(rew_means_for_scale)
        # Extra headroom at top to keep formula + return labels off the data.
        ax_rew.set_ylim(
            float(arr.min()) - 0.02,
            float(arr.max()) * 1.45 + 0.02,
        )

    # Return annotations on ONE line at top-right of the reward panel, each
    # zone in its own colour. Separator stays grey for readability.
    if zone_return:
        # Render right-to-left so each text knows where the next one ends.
        x = 0.985
        for i, (z, ret) in enumerate(reversed(zone_return)):
            idx = len(zone_return) - 1 - i
            label = ['A', 'B', 'C', 'D'][idx]
            txt = rf"$\mathrm{{Return}}_{label} = {ret:.1f}$"
            t = ax_rew.text(
                x, 0.85, txt,
                transform=ax_rew.transAxes,
                ha="right", va="bottom",
                color=colour_for(z),
                fontsize=20,
            )
            # Estimate text width in axes fraction to lay out a separator
            # before the next one.
            renderer = fig.canvas.get_renderer()
            bbox = t.get_window_extent(renderer=renderer)
            ax_bbox = ax_rew.get_window_extent(renderer=renderer)
            frac_w = bbox.width / ax_bbox.width
            if i < len(zone_return) - 1:
                # Small gap (no separator) between consecutive labels.
                x = x - frac_w - 0.025
    ax_rew.axhline(0, color=MUTED, linewidth=0.7, alpha=0.6)
    ax_rew.set_ylabel("Reward", rotation=0, ha="right", va="center",
                      labelpad=18, fontsize=22)
    ax_rew.set_xlim(0, 14)
    ax_rew.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax_rew.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))
    # Formula and Return labels share the SAME y and va so their bbox bottoms
    # sit on the same line.
    ax_rew.text(
        0.012, 0.85,
        r"$r_t = \log \mathrm{area}_t - \log \mathrm{area}_{t-1}$",
        transform=ax_rew.transAxes,
        ha="left", va="bottom",
        fontsize=20, color=INK,
    )

    # Bottom panel: explicit day-number ticks.
    ax_rew.set_xlabel("Day", fontsize=24, labelpad=10)
    ax_rew.set_xticks(list(range(0, 15)))
    ax_rew.tick_params(axis="x", which="major", labelsize=18)

    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"saved -> {out}")
    return out


# ── Composite stacking ──────────────────────────────────────────────────
def vstack_images(parts: list[Path], out: Path, gap_px: int = 24):
    imgs = [Image.open(p).convert("RGB") for p in parts]
    # Match all widths to the widest, scaling proportionally
    target_w = max(im.width for im in imgs)
    scaled = []
    for im in imgs:
        if im.width != target_w:
            ratio = target_w / im.width
            im = im.resize((target_w, int(im.height * ratio)), Image.LANCZOS)
        scaled.append(im)
    total_h = sum(im.height for im in scaled) + gap_px * (len(scaled) - 1)
    canvas = Image.new("RGB", (target_w, total_h), BG)
    y = 0
    for im in scaled:
        canvas.paste(im, (0, y))
        y += im.height + gap_px
    canvas.save(out, "PNG")
    print(f"saved -> {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Zones: {ZONES}")

    # Still produce the standalone strip JPGs (handy for slides as separate
    # artifacts), but the composite is now a single matplotlib figure that
    # places photos on the same x-axis as the data plots below.
    for z in ZONES:
        frames = daily_frames(z)
        if DAYS is not None:
            keep = set(DAYS)
            frames_sel = [f for f in frames if f[0] in keep]
        else:
            frames_sel = frames
        print(f"  Z{z}: standalone strip {len(frames_sel)} tiles")
        build_strip(z, frames_sel, use_viz=False,
                    out=OUT_DIR / f"z{z:02d}_film_strip.jpg")
        build_strip(z, frames_sel, use_viz=True,
                    out=OUT_DIR / f"z{z:02d}_film_strip_viz.jpg")

    suffix = "_".join(f"z{z:02d}" for z in ZONES)
    # Wider canvas now that the strip is integrated and shows 14 tiles.
    width_px = 14 * 150 + 200    # ~14 days * ~150px square + y-label area
    build_features_panel(width_px, OUT_DIR / f"{suffix}_composite.png")


if __name__ == "__main__":
    main()
