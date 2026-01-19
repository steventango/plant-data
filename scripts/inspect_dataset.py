import polars as pl

df = pl.scan_parquet("/data/plant-rl/offline/v20/mixed-v20.parquet")
pl.Config.set_tbl_rows(1000)
pl.Config.set_tbl_cols(20)

print(
    df.filter(
        (pl.col("experiment") == 14)
        & (pl.col("zone") == 1)
        & (pl.col("plant_id") == 55)
    )
    .select(
        "wall_time",
        "clean_area",
        "clean_solidity",
        "red_coef",
        "white_coef",
        "blue_coef",
        "reward",
        "terminal",
        "truncated",
        "outlier",
    )
    .collect()
)

# cols = [
#     "area",
#     "clean_area",
# ]

# print(df.select(cols).describe())


# df = df.with_columns(
#     (("/data/plant-rl/online/E13/P1/Dirichlet1/alliance-zone01/processed/v17/" + pl.col("image_path")).str.replace(".jpg", "_segment.jpg", literal=True)).alias("image_path")
# )

# for i in range(18):
#     filtered_df =  df.filter(
#         (pl.col("experiment") == 13) & (pl.col("zone") == 1) & (pl.col("plant_id") == i)
#     ).select([
#         "plant_id",
#         "time",
#         "wall_time",
#         "area",
#         "uema_area",
#         "clean_area",
#         "reward",
#         "image_path"
#     ]).collect(engine="streaming")
#     if filtered_df["clean_area"].len() > 3 and filtered_df["clean_area"][3] > 10:
#         continue
#     print(filtered_df)
#     print("\n".join(filtered_df["image_path"].to_list()))

# check columns wall_time, clean_* for nan / inf values
