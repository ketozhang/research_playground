from pathlib import Path

import numpy as np
import pandas as pd
import prospect

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
    filename = f"zpeg_{zpeg_num}.csv"
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


def get_prospector(**kwargs):
    dirpath = DATA_PATH / "prospector"
    filename = "prospector.csv"

    if (dirpath / filename).exists():
        return pd.read_csv(dirpath / filename)
    else:
        return _generate_data_from_prospector(dirpath / filename, **kwargs)


def _generate_data_from_prospector(savefile=None, burn="auto"):
    df = get_jla().drop(columns=["host_mass"])

    # Get HDI regions for host mass
    def get_host_mass_mean_sd(filename, burn=burn):
        res, _, _ = prospect.io.read_results.results_from(filename)

        mass_idx = res["theta_labels"].index("mass")
        trace = np.log10(res["chain"][:, :, mass_idx])

        if burn == "auto":
            burn = trace.shape[1] // 2
        trace_burn = trace[:, burn:]

        return np.mean(trace_burn), np.std(trace_burn)

    idx_without_results = []
    for idx in df.index:
        try:
            glob = (DATA_PATH / "prospector" / "mass_chains").glob(
                f"SDSSphot_A_{idx}*.h5"
            )
            filename = next(glob)
        except StopIteration:
            idx_without_results.append(idx)
            continue

        host_mass_mean, host_mass_sd = get_host_mass_mean_sd(str(filename))
        df.loc[idx, "host_mass"] = host_mass_mean
        df.loc[idx, "host_mass_sigma"] = host_mass_sd

    if savefile:
        df.to_csv(savefile, index=False)

    return df


def _fill_missing_cols(df):
    missing_cols = set(COLUMNS) - set(df.columns)

    for col in missing_cols:
        if col.endswith("_sigma"):
            df[col] = 0

    return df
