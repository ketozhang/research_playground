import logging
import multiprocessing as mp
import sys
from datetime import datetime as dt

from scipy import stats as st

from data import dataloader
from models.model import Model

DEFAULT_NPROCS = 2 * mp.cpu_count() + 1


def main(num, *args, **kwargs):
    sne_data = dataloader.get_zpeg(num)
    model = Model(
        sne_data,
        H0=70,
        host_mass_correction_model="step",
        #     param_rvs = {
        #         "sigma_m_B": st.halfnorm(1),
        #         "Om0": st.uniform(0.25, 0.25),
        #         "alpha": st.norm(1.5, 0.1),
        #         "beta": st.norm(3, 0.1),
        #         "M_B": st.norm(-19, 10),
        #         # "host_mass": self.get_host_mass_rv(self.X["host_mass"]),
        #         "loc": st.uniform(8, 2),
        #         "size": st.norm(0.05, 0.01),
        #         "slope": st.norm(0, 1),
        #     },
        param_rvs={
            "sigma_int": st.halfcauchy(scale=1),
            "Om0": st.uniform(0.25, 0.1),
            "alpha": st.norm(0, 1),
            "beta": st.norm(3, 0.2),
            "M_B": st.norm(-19, 10),
            # "host_mass": self.get_host_mass_rv(self.X["host_mass"]),
            "loc": st.gamma(a=100, scale=0.1),  # mean=a*scale, var=a*scale**2
            "size": st.uniform(0, 0.1),
            "slope": st.norm(0, 1),
        },
    )

    nsteps = 10000
    nwalkers = 24

    with mp.Pool(min(DEFAULT_NPROCS, nwalkers)) as pool:
        sampler = model.get_sampler(
            nwalkers,
            savefile=f"results/zpeg/results_{num}_{dt.utcnow().strftime('%Y-%m-%dT%H%M')}.hd5",
            pool=pool,
        )
        sampler.run_mcmc(
            nsteps=nsteps, initial_state=model.sample_priors(nwalkers), progress=True
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    logfile = "_".join(args)
    logging.basicConfig(filename=f"logs/{logfile}")
    main(*args)
