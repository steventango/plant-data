from zoneinfo import ZoneInfo

import numpy as np


VERSION = "v23"
VISION_VERSION = "v6"

GOOD_ZONE_DAYS = {
    "E11/zone1": [1, 2, 3, 4, 5, 6, 7, 9, 10, 11],
    "E11/zone2": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11],
    "E11/zone3": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
    "E11/zone4": [1, 2, 3, 4, 5, 6, 7, 9, 10],
    # "E11/zone5": [],
    "E11/zone6": [1, 2, 3, 4, 5, 6, 7],
    # maybe add 9, but the daily curve was very noisy
    # "E11/zone7": [],
    "E11/zone8": [1, 2, 3, 4, 5, 6, 7],
    "E11/zone9": [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12],
    "E11/zone10": [1, 2, 3, 4, 5, 6, 7, 9, 10, 11],
    "E11/zone11": [1, 2, 3, 4, 5, 6, 7, 9, 10, 11],
    "E11/zone12": [1, 2, 3, 4, 5, 6, 7, 9, 10, 11],
    "E12/zone1": [0, 1, 2, 3, 4, 5, 6, 7],
    "E12/zone2": [0, 1, 2, 3, 4, 5, 6, 7],
    "E12/zone3": [0, 1, 2, 3, 4, 5, 6, 7],
    "E12/zone4": [0, 1, 2, 3, 4, 5, 6, 7],
    "E12/zone5": [0, 1, 2, 3, 4, 5, 6, 7],
    "E12/zone6": [0, 1, 2, 3, 4, 5, 6, 7],
    # "E12/zone7": [],
    "E12/zone8": [0, 1, 2, 3, 4, 5, 6, 7],
    "E12/zone9": [0, 1, 2, 3, 4, 5, 6, 7],
    "E12/zone10": [0, 1, 2, 3, 4, 5, 6, 7],
    "E12/zone11": [0, 1, 2, 3, 4, 5, 6, 7],
    "E12/zone12": [0, 1, 2, 3, 4, 5, 6, 7],
    "E13/zone1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "E13/zone2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    # E13/zone2, questionable data on day 8, but we keep it anyway
    # we only got measurements @ 9:16 and 9:52
    "E13/zone3": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "E13/zone4": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "E13/zone5": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "E13/zone6": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    # "E13/zone7": [],
    "E13/zone8": [0, 1, 2, 3, 4, 5],  # bad data on days 6, 7
    "E13/zone9": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "E13/zone10": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "E13/zone11": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "E13/zone12": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "E14/zone1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "E14/zone2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "E14/zone3": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "E14/zone4": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "E14/zone5": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "E14/zone6": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    # "E14/zone7": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "E14/zone8": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "E14/zone9": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "E14/zone10": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "E14/zone11": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "E14/zone12": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
}
TIMEZONE = "America/Edmonton"
tzinfo = ZoneInfo(TIMEZONE)
RED = np.array([9.71409574, 34.97074468, 4.01515957, 0.0, 56.3, 6.13067376])
WHITE = np.array([19.6875, 70.875, 8.1375, 0.0, 6.3, 12.425])
BLUE = np.array([69.6875, 29.33653846, 3.36826923, 0.0, 2.60769231, 5.14294872])

COLS = [
    "wall_time",
    "clean_area",
    "clean_convex_hull_area",
    "clean_solidity",
    "clean_perimeter",
    "clean_width",
    "clean_height",
    "clean_longest_path",
    "clean_center_of_mass_x",
    "clean_center_of_mass_y",
    "clean_convex_hull_vertices",
    "clean_ellipse_center_x",
    "clean_ellipse_center_y",
    "clean_ellipse_major_axis",
    "clean_ellipse_minor_axis",
    "clean_ellipse_angle",
    "clean_ellipse_eccentricity",
    "clean_blue-yellow_frequencies_mean",
    "clean_blue_frequencies_mean",
    "clean_green-magenta_frequencies_mean",
    "clean_green_frequencies_mean",
    "clean_hue_circular_mean",
    "clean_hue_circular_std",
    "clean_hue_frequencies_mean",
    "clean_lightness_frequencies_mean",
    "clean_red_frequencies_mean",
    "clean_saturation_frequencies_mean",
    "clean_value_frequencies_mean",
    "log_clean_area",
    "red_coef_trace_0.9",
    "white_coef_trace_0.9",
    "blue_coef_trace_0.9",
]
