from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent

# Any data outside these conditions are set to NaN
CUTOFFS = [
    "host_mass > 0 and host_mass < 20",
    "redshift > 0",
]
COLUMNS = [
    "redshift",
    "mag",
    "mag_sigma",
    "stretch",
    "stretch_sigma",
    "color",
    "color_sigma",
    "host_mass",
    "host_mass_sigma",
]


def get_zpeg(zpeg_num):
    dirpath = DATA_PATH / "zpeg"
    filename = f"JLA_SDSSphot_A_nodustcorr_ZPEG_{zpeg_num}_rev1.csv"
    df = pd.read_csv(dirpath / filename)

    df = df.rename(
        columns={
            "zcmb": "redshift",
            "mb": "mag",
            "x1": "stretch",
            "color": "color",
            "ZPEG_StMass": "host_mass",
        }
    ).pipe(_fill_missing_cols)

    # Clean host mass:
    df["host_mass"] = pd.to_numeric(df["host_mass"], errors="coerce")
    df.loc[
        df["host_mass"] <= 0, "host_mass"
    ] = np.nan  # Assume non-positive values are NaNs when ZPEG fails to fit
    df["host_mass"] = np.log10(df["host_mass"])  # Set unit to log10 solar mass

    return df


def get_jla():
    dirpath = DATA_PATH / "jla"
    filename = "jla.txt"
    df = pd.read_csv(dirpath / filename, sep="\s+")

    df = df.rename(
        columns={
            "zcmb": "redshift",
            "mb": "mag",
            "dmb": "mag_sigma",
            "x1": "stretch",
            "dx1": "stretch_sigma",
            "color": "color",
            "dcolor": "color_sigma",
            "3rdvar": "host_mass",
            "d3rdvar": "host_mass_sigma",
        }
    ).pipe(_fill_missing_cols)

    return df


def _fill_missing_cols(df):
    missing_cols = set(COLUMNS) - set(df.columns)

    for col in missing_cols:
        if col.endswith("_sigma"):
            df[col] = 0

    return df
