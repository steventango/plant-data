import polars as pl

OLD_COLS = [
    "agent_action",
    "reward",
    "terminal",
    "return",
    "mean_clean_area",
    "env_time",
    "in_bounds",
    "area",
    "convex_hull_area",
    "solidity",
    "perimeter",
    "width",
    "height",
    "longest_path",
    "center_of_mass_x",
    "center_of_mass_y",
    "convex_hull_vertices",
    "object_in_frame",
    "ellipse_center_x",
    "ellipse_center_y",
    "ellipse_major_axis",
    "ellipse_minor_axis",
    "ellipse_angle",
    "ellipse_eccentricity",
]


def transform_drop_old_cols(df: pl.DataFrame) -> pl.DataFrame:
    drop_cols = []
    for col in df.columns:
        if (
            col.startswith("state.")
            or col.startswith("calibrated_action.")
            or col.startswith("agent_action.")
            or col.startswith("uema_area.")
            or col.startswith("upper_area.")
            or col.startswith("lower_area.")
        ) or col in OLD_COLS:
            drop_cols.append(col)

    df = df.drop(drop_cols)
    return df
