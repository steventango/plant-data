import os

import pandas as pd
import pyarrow.parquet as pq


def analyze_parquet_size(file_path):
    # 1. Check basic file stats
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    metadata = pq.read_metadata(file_path)

    print(f"--- Analysis for: {file_path} ---")
    print(f"Total File Size: {file_size_mb:.2f} MB")
    print(f"Total Rows: {metadata.num_rows}")
    print(f"Row Groups: {metadata.num_row_groups}")
    print("-" * 30)

    # 2. Aggregate stats per column across all row groups
    # We want to know: Which column is physically taking up the most space on disk?
    column_stats = {}

    schema = metadata.schema

    # Iterate through every row group and every column chunk
    for i in range(metadata.num_row_groups):
        row_group = metadata.row_group(i)
        for j in range(row_group.num_columns):
            col_chunk = row_group.column(j)
            col_name = col_chunk.path_in_schema

            if col_name not in column_stats:
                column_stats[col_name] = {
                    "compressed_bytes": 0,
                    "uncompressed_bytes": 0,
                    "type": schema.column(j).physical_type,
                }

            column_stats[col_name]["compressed_bytes"] += (
                col_chunk.total_compressed_size
            )
            column_stats[col_name]["uncompressed_bytes"] += (
                col_chunk.total_uncompressed_size
            )

    # 3. Convert to DataFrame for easy viewing
    df_stats = pd.DataFrame.from_dict(column_stats, orient="index")

    # Calculate savings ratio (higher is better compression)
    df_stats["compression_ratio"] = (
        df_stats["uncompressed_bytes"] / df_stats["compressed_bytes"]
    )

    # Calculate % of total file size this column occupies
    total_compressed = df_stats["compressed_bytes"].sum()
    df_stats["%_of_file"] = (df_stats["compressed_bytes"] / total_compressed) * 100

    # Convert bytes to MB for readability
    df_stats["compressed_MB"] = df_stats["compressed_bytes"] / (1024 * 1024)
    df_stats["uncompressed_MB"] = df_stats["uncompressed_bytes"] / (1024 * 1024)

    # Sort by the columns taking up the most space
    df_stats = df_stats.sort_values(by="compressed_bytes", ascending=False)

    # Select clean columns to display
    display_cols = ["type", "compressed_MB", "%_of_file", "compression_ratio"]
    print(df_stats[display_cols].round(2))

    return df_stats


# --- USAGE ---
# Replace with your actual file path
analyze_parquet_size("/data/plant-rl/offline/v21/mixed-v21.parquet")
