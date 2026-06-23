"""Shared visual style constants and helpers for all visualization scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Standard figure DPI
# ---------------------------------------------------------------------------
DPI = 200

# ---------------------------------------------------------------------------
# E16 agent×zone palette (used by build_layout in plot_e16_metrics)
# ---------------------------------------------------------------------------
PALETTE_BASE: list[str] = [
    "#4e9ac7",
    "#e85c4a",
    "#6fbf3e",
    "#a050b0",
    "#e89c2a",
    "#2ab0a0",
    "#c06030",
    "#8080c0",
]

PALETTE_SHADES: list[list[str]] = [
    ["#a8d4ef", "#4e9ac7", "#1e6a96"],
    ["#f4a99e", "#e85c4a", "#b52c1e"],
    ["#b5e089", "#6fbf3e", "#3a8a1a"],
    ["#d9a8e8", "#a050b0", "#6a2080"],
    ["#f7d49a", "#e89c2a", "#a06000"],
    ["#9adfd8", "#2ab0a0", "#0a7a70"],
    ["#e8b898", "#c06030", "#803010"],
    ["#c0c0e8", "#8080c0", "#404090"],
]

PALETTE_BG: list[str] = [
    "#eaf4fb",
    "#fdecea",
    "#eef9e6",
    "#f5eafb",
    "#fef6e8",
    "#e8f9f7",
    "#faeee8",
    "#eeeef8",
]

# ---------------------------------------------------------------------------
# E18 zone colours (zones 1, 3, 4, 11 only)
# ---------------------------------------------------------------------------
E18_ZONE_COLORS: dict[int, str] = {
    1: "#e74c3c",
    3: "#3498db",
    4: "#2ecc71",
    11: "#9b59b6",
}

# ---------------------------------------------------------------------------
# Action-coefficient channel colours (R/W/B)
# ---------------------------------------------------------------------------
COEF_COLORS: dict[str, str] = {
    "red_coef": "#e63946",
    "white_coef": "#adb5bd",
    "blue_coef": "#457b9d",
}
COEF_LABELS: dict[str, str] = {
    "red_coef": "Red",
    "white_coef": "White",
    "blue_coef": "Blue",
}

# ---------------------------------------------------------------------------
# Tol rainbow palette (E18 return/energy plots)
# ---------------------------------------------------------------------------
TOL_RAINBOW_11: list[str] = [
    "#882E72",
    "#1965B0",
    "#7BAFDE",
    "#4EB265",
    "#CAE0AB",
    "#F7CB45",
    "#EE8026",
    "#E65518",
    "#DC050C",
    "#72190E",
    "#42150A",
]

PARETO_COLOR: str = "#2C4875"


def build_palette(n: int) -> list[str]:
    """Return an n-colour Tol rainbow palette, falling back to TOL_RAINBOW_11."""
    try:
        import matplotlib.colors as mcolors
        import tol_colors as tc

        return [mcolors.to_hex(c) for c in tc.rainbow_discrete(n).colors]
    except ImportError:
        return [TOL_RAINBOW_11[i % len(TOL_RAINBOW_11)] for i in range(n)]


# ---------------------------------------------------------------------------
# Global theme helper
# ---------------------------------------------------------------------------
def set_theme(style: str = "white") -> None:
    """Apply a consistent seaborn theme + sans-serif font."""
    sns.set_theme(style=style)
    plt.rcParams.update({"font.family": "sans-serif"})


# ---------------------------------------------------------------------------
# savefig helper
# ---------------------------------------------------------------------------
def savefig(
    fig: plt.Figure,
    path: Path | str,
    dpi: int = DPI,
    bbox_inches: str = "tight",
    **kw,
) -> None:
    """Save *fig* to *path*, create parent dirs, close figure, and print path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **kw)
    plt.close(fig)
    print(f"Saved → {path}")
