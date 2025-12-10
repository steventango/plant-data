import polars as pl


# Configuration for clean area calculation
# Tukey outlier detection (across plants per timestep)
TUKEY_K_UPPER = 1.5  # Conservative k factor for Tukey fence

# EWM outlier detection (within each plant over time)
CLEAN_AREA_LOWER_THRESHOLD = 0.5  # Reject if area < (1 - threshold) * ewm_mean
CLEAN_AREA_UPPER_THRESHOLD = 1.5  # Reject if area > (1 + threshold) * ewm_mean
MINIMUM_AREA_COUNT = 1  # Minimum observations before applying outlier detection
EWM_BETA = 0.1  # Decay factor for EWM (higher = smoother); alpha = 1 - beta

# Morphology features that should be replaced together with area when outlier detected
MORPHOLOGY_FEATURES = [
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


def apply_tukey_outlier_detection(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply conservative Tukey outlier detection across plants per timestep.

    For each (experiment, zone, time) group, compute Tukey fences and set
    outlier large areas to 0 (treating them as failed measurements).

    Uses a conservative k=3.0 (vs standard 1.5) to only flag extreme outliers.
    Tukey fences: [Q1 - k*IQR, Q3 + k*IQR]
    """
    # Compute Q1, Q3, IQR per timestep across all plants
    df = df.with_columns(
        [
            pl.col("area")
            .quantile(0.25)
            .over(["experiment", "zone", "time"])
            .alias("_q1"),
            pl.col("area")
            .quantile(0.75)
            .over(["experiment", "zone", "time"])
            .alias("_q3"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("_q3") - pl.col("_q1")).alias("_iqr"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("_q3") + TUKEY_K_UPPER * pl.col("_iqr")).alias("_upper_fence"),
        ]
    )

    # Flag outliers and set their area to 0 (will be treated as invalid)
    df = df.with_columns(
        [
            pl.when(pl.col("area") > pl.col("_upper_fence"))
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("tukey_outlier"),
        ]
    )

    # Set outlier areas to 0 (they will be replaced by EWM logic later)
    df = df.with_columns(
        [
            pl.when(pl.col("tukey_outlier"))
            .then(pl.lit(0.0))
            .otherwise(pl.col("area"))
            .alias("area_after_tukey"),
        ]
    )

    # Clean up temporary columns
    df = df.drop(["_q1", "_q3", "_iqr", "_upper_fence"])

    return df


def compute_clean_features_for_plant(plant_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute clean morphology features for a single plant's time series.

    This function implements robust feature cleaning by:
    1. Using area_after_tukey (with cross-plant outliers already set to 0)
    2. Computing an EWM (exponential weighted mean) of the area
    3. Detecting within-plant outliers that deviate too much from the EWM
    4. Replacing ALL morphology features with the previous clean values when outlier detected

    The computation must be done row-by-row because:
    - The EWM should only be updated with non-outlier values
    - Outlier detection depends on the current EWM state
    - The replacement values depend on prior clean values
    """
    # Sort by time to ensure proper ordering
    plant_df = plant_df.sort("time")

    # Use area_after_tukey if available, otherwise fall back to area
    area_col = "area_after_tukey" if "area_after_tukey" in plant_df.columns else "area"

    # Extract areas as a list for row-by-row processing
    areas = plant_df[area_col].to_list()
    n = len(areas)

    if n == 0:
        return plant_df.with_columns(pl.lit(None).cast(pl.Float64).alias("clean_area"))

    # Determine which morphology features are present in the dataframe
    available_features = [f for f in MORPHOLOGY_FEATURES if f in plant_df.columns]

    # Extract all feature values as lists for row-by-row processing
    feature_values = {f: plant_df[f].to_list() for f in available_features}

    # Initialize tracking variables for all features
    clean_features = {f: [] for f in available_features}
    clean_features["clean_area"] = []  # Always create clean_area
    ewm_values = []
    is_outlier_list = []

    # EWM parameters: alpha = 1 - beta (beta=0.9 means alpha=0.1)
    alpha = 1.0 - EWM_BETA

    # State variables for EWM calculation (with adjust=True behavior)
    ewm_sum = 0.0
    ewm_weight = 0.0

    # Previous clean values for all features
    prev_clean_values: dict[str, float | None] = {f: 0.0 for f in available_features}
    prev_clean_area = 0.0
    area_count = 0

    for i in range(n):
        area = areas[i]

        # Handle null/missing areas or areas set to 0 by Tukey detection
        if area is None or area <= 0:
            # Use previous clean values for all features
            for f in available_features:
                clean_features[f].append(prev_clean_values[f])
            clean_features["clean_area"].append(prev_clean_area)
            ewm_values.append(ewm_sum / ewm_weight if ewm_weight > 0 else None)
            is_outlier_list.append(True)  # Treat as outlier
            continue

        # Calculate current EWM mean
        if ewm_weight > 0:
            current_ewm = ewm_sum / ewm_weight
        else:
            current_ewm = area  # First observation

        ewm_values.append(current_ewm)

        # Determine if this is an outlier (within-plant temporal check)
        is_outlier = False
        if area_count >= MINIMUM_AREA_COUNT and prev_clean_area is not None:
            lower_bound = (1 - CLEAN_AREA_LOWER_THRESHOLD) * current_ewm
            upper_bound = (1 + CLEAN_AREA_UPPER_THRESHOLD) * current_ewm
            is_outlier = area < lower_bound or area > upper_bound

        is_outlier_list.append(is_outlier)

        if is_outlier:
            # Use previous clean values for ALL features
            for f in available_features:
                clean_features[f].append(prev_clean_values[f])
            clean_features["clean_area"].append(prev_clean_area)
        else:
            # Accept current values as clean for ALL features
            for f in available_features:
                current_val = feature_values[f][i]
                clean_features[f].append(current_val)
                prev_clean_values[f] = current_val

            # Special handling for clean_area (may come from area_after_tukey)
            clean_features["clean_area"].append(area)
            prev_clean_area = area

            # Update EWM with this valid observation
            ewm_sum = ewm_sum * (1 - alpha) + area
            ewm_weight = ewm_weight * (1 - alpha) + 1.0
            area_count += 1

    # Build columns to add
    new_columns = [
        pl.Series("clean_area", clean_features["clean_area"]).cast(pl.Float64),
        pl.Series("uema_area", ewm_values).cast(pl.Float64),
        pl.Series("is_outlier", is_outlier_list),
    ]

    # Add clean versions of all morphology features (prefixed with "clean_")
    for f in available_features:
        if f != "area":  # clean_area already added
            new_columns.append(
                pl.Series(f"clean_{f}", clean_features[f]).cast(pl.Float64)
            )

    result = plant_df.with_columns(new_columns)

    return result


def compute_clean_features_for_group(group_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute clean morphology features for all plants in an (experiment, zone) group.
    
    Processes each plant_id independently to maintain separate EWM states.
    """    
    # Process each plant independently
    result_frames = []
    for plant_id, plant_group in group_df.group_by("plant_id", maintain_order=True):
        cleaned_plant = compute_clean_features_for_plant(plant_group)
        result_frames.append(cleaned_plant)
    
    if not result_frames:
        return group_df.with_columns(pl.lit(None).cast(pl.Float64).alias("clean_area"))
    
    return pl.concat(result_frames)


# Keep the old function name as an alias for backwards compatibility
def compute_clean_area_for_group(group_df: pl.DataFrame) -> pl.DataFrame:
    """Alias for compute_clean_features_for_group for backwards compatibility."""
    return compute_clean_features_for_group(group_df)


def transform_area(df: pl.DataFrame) -> pl.DataFrame:
    """
    Transform the area column to compute clean_area and clean morphology features
    using a hybrid approach:

    1. Cross-plant Tukey outlier detection: For each timestep, apply conservative
       Tukey fences (k=1.5) across all plants to detect extreme outliers
       (e.g., masked pot instead of plant). These are set to 0.

    2. Within-plant EWM outlier detection: For each plant's time series, use an
       exponential weighted mean to detect temporal outliers that deviate too
       much from the plant's recent history. When an outlier is detected,
       ALL morphology features are replaced with the previous clean values.

    This handles:
    - Vision pipeline failures (areas too big - masked pot instead of plant)
    - Failed plant detection (areas too small or zero)

    Output columns:
    - clean_area: Cleaned area values
    - uema_area: Exponential weighted mean of area (for debugging/visualization)
    - is_outlier: Boolean flag indicating if the observation was an outlier
    - clean_<feature>: Cleaned versions of all morphology features
    """
    # Ensure we have the required columns
    if "area" not in df.columns:
        raise ValueError("DataFrame must have 'area' column")

    # Step 1: Apply Tukey outlier detection across plants per timestep
    df = apply_tukey_outlier_detection(df)

    # Step 2: Group by experiment and zone, apply within-plant EWM cleaning
    result_frames = []

    for (exp, zone), group in df.group_by(["experiment", "zone"], maintain_order=True):
        cleaned_group = compute_clean_features_for_group(group)
        result_frames.append(cleaned_group)

    if not result_frames:
        return df.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("clean_area"),
                pl.lit(None).cast(pl.Float64).alias("uema_area"),
            ]
        )

    # Concatenate all groups back together
    result = pl.concat(result_frames)

    # Clean up intermediate column
    if "area_after_tukey" in result.columns:
        result = result.drop("area_after_tukey")

    return result


def transform_wall_time(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("time")
        .min()
        .over(["experiment", "zone"])
        .dt.truncate("1d")
        .alias("first_day_midnight")
    )
    df = df.with_columns(
        (pl.col("first_day_midnight") + pl.duration(hours=9, minutes=30)).alias(
            "ref_time"
        )
    )
    df = df.with_columns(
        ((pl.col("time") - pl.col("ref_time")) / pl.duration(days=1)).alias("wall_time")
    )
    df = df.drop(["first_day_midnight", "ref_time"])
    return df


def transform_state(df: pl.DataFrame) -> pl.DataFrame:
    """
    Transform state-related columns including clean area computation.
    """
    df = transform_area(df)
    df = transform_wall_time(df)
    df = df.with_columns(
        pl.col("clean_area")
        .mean()
        .over("experiment", "zone", "time")
        .alias("mean_clean_area"),
    )
    return df
