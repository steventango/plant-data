import polars as pl

df = pl.read_parquet(
    "/data/plant-rl/online/E13/P1/Dirichlet1/alliance-zone01/processed/v16/E13_Z1.parquet"
)
pl.Config.set_tbl_rows(1000)
pl.Config.set_tbl_cols(20)

cols = [
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
    "ellipse_center_x",
    "ellipse_center_y",
    "ellipse_major_axis",
    "ellipse_minor_axis",
    "ellipse_angle",
    "ellipse_eccentricity",
]

print(df.select(cols).describe())