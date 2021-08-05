import logging
import multiprocessing as mp
import sys
from datetime import datetime as dt
from pathlib import Path

from scipy import stats as st

from data import dataloader
from models.model import Model

DEFAULT_NPROCS = 2 * mp.cpu_count() + 1


def run(data, savefile):
    Path(savefile).parent.mkdir(exist_ok=True, parents=True)
    model = Model(
        H0=70,
        host_mass_correction_model="step",
        # param_rvs={
        #     "sigma_int": st.halfcauchy(scale=1),
        #     "Om0": st.uniform(0.25, 0.1),
        #     "alpha": st.norm(0, 1),
        #     "beta": st.norm(3, 0.2),
        #     "M_B": st.norm(-19, 10),
        #     # "host_mass": self.get_host_mass_rv(self.X["host_mass"]),
        #     "loc": st.uniform(9.9, 0.2),  # mean=a*scale, var=a*scale**2
        #     "size": st.uniform(0.06, 0.02),
        #     "slope": st.norm(0, 1),
        # },
    )

    nsteps = 10000
    nwalkers = 24
    with mp.Pool(min(DEFAULT_NPROCS, nwalkers)) as pool:
        model.fit(
            data=data,
            nwalkers=nwalkers,
            nsteps=nsteps,
            savefile=savefile,
            sampler_kwargs={"pool": pool},
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    logfile = "log" + "_".join(args) + ".txt"
    logging.basicConfig(filename=f"logs/{logfile}")
    timestamp = dt.utcnow().strftime("%Y-%m-%dT%H%M")

    # JLA
    # data = dataloader.get_jla()
    # run(data, f"results/jla/results_{timestamp}.hd5")

    # ZPEG
    for i in range(1, 12 + 1):
        data = dataloader.get_zpeg(i)
        run(data, f"results/zpeg/{i}/results_{timestamp}.hd5")
        # f"results/jla/results_{dt.utcnow().strftime('%Y-%m-%dT%H%M')}.hd5"
