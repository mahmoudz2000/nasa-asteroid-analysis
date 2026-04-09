"""
nasa_asteroid_ds.py
NASA Asteroid Data Analysis (MMN 15 )
mahmoud zangri 322221557
//
This module loads the NASA asteroid CSV dataset, cleans it, performs
basic analysis and produces several plots using matplotlib.
//

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Constants (recommended)


DEFAULT_CSV_FILE = "nasa.csv"

# Column names (exactly as they appear in nasa.csv)
COL_NAME = "Name"
COL_CLOSE_DATE = "Close Approach Date"
COL_ABS_MAG = "Absolute Magnitude"
COL_DIAM_KM_MIN = "Est Dia in KM(min)"
COL_DIAM_KM_MAX = "Est Dia in KM(max)"
# The miss-distance column sometimes appears with a space after the dot in
# different CSV exports. To avoid KeyError, we resolve the actual column name
# dynamically using these candidates.
COL_MISS_KM_CANDIDATES = (
    "Miss Dist.(kilometers)",
    "Miss Dist. (kilometers)",
    "Miss Dist.( kilometers)",
    "Miss Dist. ( kilometers)",
)
COL_ORBIT_ID = "Orbit ID"
COL_MIN_ORBIT_INT = "Minimum Orbit Intersection"
COL_MPH = "Miles per hour"
COL_HAZ = "Hazardous"

# Columns to drop in details_data / data_details
DROP_COLS = ["Neo Reference ID", "Orbiting Body", "Equinox"]

# Plot settings
DIAMETER_BINS = 100
ORBIT_INTERSECTION_BINS = 10


def _resolve_column(df, candidates):
    """Return the first column name from *candidates* that exists in df.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to check.
    candidates : tuple[str, ...]
        Possible column names.

    Returns
    -------
    str or None
        Existing column name, or None if not found.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None



# Required functions (names may appear in assignment PDFs)


def data_load(file_name=DEFAULT_CSV_FILE):
    """Load a CSV file and return a pandas DataFrame.

    The function prints a clear message to standard output in common
    failure cases (missing file, empty file, or other read error).

    Parameters
    ----------
    file_name : str
        Path/name of the CSV file.

    Returns
    -------
    pandas.DataFrame or None
        Loaded DataFrame on success, otherwise None.
    """
    try:
        return pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Error: file '{file_name}' not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: file '{file_name}' is empty or invalid.")
    except Exception as exc:
        print(f"Error: failed to load '{file_name}': {exc}")
    return None


def data_mask(df):
    """Keep only asteroids with close approach date from year 2000 onward.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame that must contain the close approach date column.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame (copy).
    """
    if COL_CLOSE_DATE not in df.columns:
        raise KeyError(f"Missing required column: {COL_CLOSE_DATE}")

    dates = pd.to_datetime(df[COL_CLOSE_DATE], errors="coerce")
    filtered = df.loc[dates.dt.year >= 2000].copy()
    return filtered


def details_data(df):
    """Drop unnecessary columns and return (rows, cols, column_names).

    The assignment asks to remove: Neo Reference ID, Orbiting Body, Equinox.
    The DataFrame is modified in-place (as many course examples do).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to clean.

    Returns
    -------
    tuple
        (number_of_rows, number_of_columns, list_of_column_names)
    """
    missing = [c for c in DROP_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns required for removal: {missing}")

    df.drop(columns=DROP_COLS, inplace=True)
    return df.shape[0], df.shape[1], list(df.columns)


def magnitude_absolute_max(df):
    """Return (Name, max Absolute Magnitude).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing Name and Absolute Magnitude.

    Returns
    -------
    tuple
        (asteroid_name, max_absolute_magnitude)
    """
    if COL_NAME not in df.columns or COL_ABS_MAG not in df.columns:
        raise KeyError("Missing required columns for this function.")

    idx = df[COL_ABS_MAG].idxmax()
    return int(df.loc[idx, COL_NAME]), float(df.loc[idx, COL_ABS_MAG])


def earth_to_closest(df):
    """Return the Name of the asteroid closest to Earth (by km miss distance).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing Name and Miss Dist.(kilometers).

    Returns
    -------
    int
        Asteroid Name of the closest approach.
    """
    miss_col = _resolve_column(df, COL_MISS_KM_CANDIDATES)
    if COL_NAME not in df.columns or miss_col is None:
        raise KeyError("Missing required columns for this function.")

    idx = df[miss_col].idxmin()
    return int(df.loc[idx, COL_NAME])


def orbit_common(df):
    """Return a dictionary: {Orbit ID: number of asteroids in that orbit}.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing Orbit ID.

    Returns
    -------
    dict
        Mapping from orbit id (int) to count (int).
    """
    if COL_ORBIT_ID not in df.columns:
        raise KeyError(f"Missing required column: {COL_ORBIT_ID}")

    counts = df[COL_ORBIT_ID].value_counts().to_dict()
    return {int(k): int(v) for k, v in counts.items()}


def diameter_max_min(df):
    """Count asteroids with Est Dia in KM(max) above the average.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing Est Dia in KM(max).

    Returns
    -------
    int
        Count of asteroids above the average of the max diameter.
    """
    if COL_DIAM_KM_MAX not in df.columns:
        raise KeyError(f"Missing required column: {COL_DIAM_KM_MAX}")

    avg_max = df[COL_DIAM_KM_MAX].mean()
    return int((df[COL_DIAM_KM_MAX] > avg_max).sum())


def diameter_hist_plt(df):
    """Plot a histogram of asteroid average diameters (100 bins).

    The average diameter for each asteroid is:
        (Est Dia in KM(min) + Est Dia in KM(max)) / 2

    The plot includes a title, legend and axis labels.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the diameter columns.
    """
    if COL_DIAM_KM_MIN not in df.columns or COL_DIAM_KM_MAX not in df.columns:
        raise KeyError("Missing required diameter columns.")

    avg_diam = (df[COL_DIAM_KM_MIN] + df[COL_DIAM_KM_MAX]) / 2
    plt.figure()
    plt.hist(avg_diam, bins=DIAMETER_BINS, edgecolor="black", label="Average diameter (km)")
    plt.title("Distribution of Average diameter size")
    plt.xlabel("Average Value")
    plt.ylabel("Count")
    plt.legend()
    plt.show()


def orbit_common_hist_plt(df):
    """Plot a histogram of Minimum Orbit Intersection (10 continuous bins).

    Bins are continuous and span from the minimum value to the maximum value.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing Minimum Orbit Intersection.
    """
    if COL_MIN_ORBIT_INT not in df.columns:
        raise KeyError(f"Missing required column: {COL_MIN_ORBIT_INT}")

    values = df[COL_MIN_ORBIT_INT].dropna()
    mn = values.min()
    mx = values.max()
    bins = np.linspace(mn, mx, ORBIT_INTERSECTION_BINS + 1)

    plt.figure()
    plt.hist(values, bins=bins, edgecolor="black", label="Min Orbit Intersection")
    plt.title("Distribution of Asteroids by Minimum Orbit Intersection")
    plt.xlabel("Min Orbit Intersection")
    plt.ylabel("Number of Asteroids")
    plt.legend()
    plt.show()


def hazard_pie_plt(df):
    """Plot a pie chart of hazardous vs non-hazardous asteroids.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the Hazardous column (boolean).
    """
    if COL_HAZ not in df.columns:
        raise KeyError(f"Missing required column: {COL_HAZ}")

    counts = df[COL_HAZ].value_counts()
    # Keep boolean labels (True/False) to match the common solution style.
    labels = ["False", "True"]
    sizes = [counts.get(False, 0), counts.get(True, 0)]

    plt.figure()
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Percentage of Hazardous and Non-Hazardous Asteroids")
    plt.axis("equal")
    plt.show()


def magnitude_motion_linear_plt(df):
    """Check for a linear relationship between miss distance and speed.

    This function draws a scatter plot of:
      x = Miss Dist.(kilometers)
      y = Miles per hour
    and adds a simple least-squares regression line.

    Correlation explanation (in plain words):
    In this dataset, the relationship is typically weak-to-moderate positive.
    That means as miss distance increases, speed tends to increase slightly,
    but there is a lot of scatter, so the correlation is not strong.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing Miss Dist.(kilometers) and Miles per hour.
    """
    miss_col = _resolve_column(df, COL_MISS_KM_CANDIDATES)
    if miss_col is None or COL_MPH not in df.columns:
        raise KeyError("Missing required columns for this function.")

    x = df[miss_col].astype(float)
    y = df[COL_MPH].astype(float)

    # Simple linear regression with numpy (slope and intercept)
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 500)
    y_line = slope * x_line + intercept

    plt.figure()
    plt.scatter(x, y, s=10, alpha=0.5, label="Data points")
    plt.plot(x_line, y_line, linewidth=2, label="Regression line")
    plt.title(f"Linear Regression: {miss_col} vs {COL_MPH}")
    plt.xlabel(miss_col)
    plt.ylabel("Miles per hour")
    plt.legend()
    plt.show()



# Compatibility aliases (if your PDF uses the other names)


def load_data(file_name=DEFAULT_CSV_FILE):
    """Alias to data_load (some versions of the assignment use load_data)."""
    return data_load(file_name)


def mask_data(df):
    """Alias to data_mask (some versions of the assignment use mask_data)."""
    return data_mask(df)


def data_details(df):
    """Alias to details_data (some versions of the assignment use data_details)."""
    return details_data(df)


def max_absolute_magnitude(df):
    """Alias to magnitude_absolute_max (some versions use max_absolute_magnitude)."""
    return magnitude_absolute_max(df)


def closest_to_earth(df):
    """Alias to earth_to_closest (some versions use closest_to_earth)."""
    return earth_to_closest(df)


def common_orbit(df):
    """Alias to orbit_common (some versions use common_orbit)."""
    return orbit_common(df)


def min_max_diameter(df):
    """Alias to diameter_max_min (some versions use min_max_diameter)."""
    return diameter_max_min(df)


def plt_hist_diameter(df):
    """Alias to diameter_hist_plt (some versions use plt_hist_diameter)."""
    return diameter_hist_plt(df)


def plt_hist_common_orbit(df):
    """Alias to orbit_common_hist_plt (some versions use plt_hist_common_orbit)."""
    return orbit_common_hist_plt(df)


def plt_pie_hazard(df):
    """Alias to hazard_pie_plt (some versions use plt_pie_hazard)."""
    return hazard_pie_plt(df)


def plt_linear_motion_magnitude(df):
    """Alias to magnitude_motion_linear_plt (some versions use plt_linear_motion_magnitude)."""
    return magnitude_motion_linear_plt(df)
