import polars as pl

df = pl.read_parquet(
    "/data/plant-rl/online/E13/P1/Dirichlet1/alliance-zone01/processed/v16/E13_Z1.parquet"
)
pl.Config.set_tbl_rows(1000)


# count plant bolting by day
print(df.group_by("day").agg((pl.col("bolted_pred") > 0.5).sum()).sort("day"))

for plant_id in df.select("plant_id").unique().to_numpy():
    print(
        df.filter(df["plant_id"] == plant_id).select(
            "day", "time", "reward", "bolted_pred", "terminal", "image_name"
        )
    )
