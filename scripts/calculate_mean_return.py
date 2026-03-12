import polars as pl

pl.Config.set_tbl_rows(1000)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(1000)

df = pl.read_parquet("/data/plant-rl/offline/v23/mixed-v23.parquet")

# Filter out rows after terminal state
# shift reward forwards by 1 over each plant trajectory
df_filtered = df.with_columns(
    pl.col("reward").shift(1).over(["experiment", "zone", "plant_id"])
)
df_filtered = df_filtered.filter(~pl.col("terminal"))

# Calculate total return for each plant trajectory
plant_returns = df_filtered.group_by(["experiment", "zone", "plant_id"]).agg(
    pl.col("reward").sum().alias("total_return")
)

# Calculate mean total return for each experiment
experiment_means = (
    plant_returns.group_by("experiment", "zone")
    .agg(
        pl.col("total_return").mean().alias("mean_total_return"),
        pl.col("total_return").count().alias("num_plants"),
    )
    .sort(["experiment", "mean_total_return"], descending=[False, True])
)

print(experiment_means)

print(
    experiment_means.group_by("experiment").agg(
        pl.col("mean_total_return").max() - pl.col("mean_total_return").min()
    )
)
