from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent


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

    df["host_mass"] = np.log10(df["host_mass"])

    return df
