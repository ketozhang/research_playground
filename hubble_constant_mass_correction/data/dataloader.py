from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent

# Any data outside these conditions are set to NaN
CUTOFFS = [
    "host_mass > 0 and host_mass < 20",
    "redshift > 0",
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
    )

    # Clean host mass:
    df.loc[
        df["host_mass"] <= 0, "host_mass"
    ] = np.nan  # Assume non-positive values are NaNs when ZPEG fails to fit
    df["host_mass"] = np.log10(df["host_mass"])  # Set unit to log10 solar mass

    return df


def get_jla():
    dirpath = DATA_PATH / "jla"
    filename = "jla.csv"
    df = pd.read_csv(dirpath / filename)

    df = df.rename(
        columns={
            "zcmb": "redshift",
            "mb": "mag",
            "x1": "stretch",
            "color": "color",
            "3rdvar": "host_mass",
        }
    )

    return df
