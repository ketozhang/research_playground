from pathlib import Path

import arviz as az
import emcee
import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3.distributions import Interpolated
from scipy import stats as st

from . import mass_corrections
from .distmod_cosmology import get_mu_th
from .distmod_tripp import get_mu_tripp


class Model:
    def __init__(
        self,
        sne_data,
        H0,
        host_mass_correction_model="step",
        param_rvs=None,
        overwrite=False,
    ):
        r"""SNe standardization model with mass step correction

        \mu = m_B - (M_B + \alpha \cdot x + \beta \cdot c + \delta m_*) + \epsilon

        More useful for MCMC is to rewrite this in the form of "true = model + error". Note the sign for the error
        term doesn't matter.

        m_B = [\mu + M_B + \alpha \cdot x + \beta \cdot c + \delta m_*] + \epsilon

        Parameters
        ----------
        sne_data : pd.DataFrame
            SN params to calculate the observed distance modulus. The following keys/columns must exist

                mag : array-like
                    B-band apparent magnitude
                redshift : array-like
                stretch : array-like
                    SALT2 x1 parameter
                color : array-like
                    SALT2 c parameter
                host_mass : array-like
                    An array of host stellar mass with each element per SN must be one of the following:
                        * float, the observed host mass is deterministic without uncertainty
                        * array-like of size 2 with columns being `"host_mass"` and `"host_mass_err"`,
                            the observed host mass is normally distributed with mu and sigma taken from
                            the keys respectively.
                        * array-like of size >2, the observed host mass is distributed as the given posterior samples

        H0 : float
            Hubble constant
        host_mass_correction_model : str
            The host mass correction model to use None, "step", or "sigmoid"
        param_rvs : dict
            With key being the parameter name and values being a random variable (`scipy.stats.rv_continuous`)
        """
        # Handles design matrix
        if isinstance(sne_data, pd.DataFrame):
            self.y = sne_data["mag"]
            self.X = sne_data.drop(columns=["mag"])
        else:
            raise ValueError("Arg `sne_data` must be `pd.DataFrame`.")

        self.H0 = H0
        self.nrows = self.X.shape[0]

        # Handle parameters
        self.param_rvs = {
            "sigma_int": st.halfnorm(1),
            "Om0": st.uniform(0, 1),
            "alpha": st.norm(0, 10),
            "beta": st.norm(0, 10),
            "M_B": st.norm(0, 100),
            # "host_mass": self.get_host_mass_rv(self.X["host_mass"]),
            "loc": st.uniform(0, 1),
            "size": st.norm(0, 1),
            "slope": st.norm(0, 1),
        }
        if param_rvs:
            # Overwrite default priors in `param_rvs`
            # with user-specified values.
            assert (
                len(set(param_rvs.keys()) - set(self.param_rvs.keys())) == 0
            ), "Arg `param_priors` contains unsupported keys"
            self.param_rvs.update(param_rvs)

        self.host_mass_correction_model = host_mass_correction_model
        if self.host_mass_correction_model == "step":
            self.param_rvs.pop("slope")
        elif self.host_mass_correction_model == "sigmoid":
            pass
        else:
            raise ValueError()

        self.nparams = len(self.param_rvs)

        # META
        self._nwalkers = None

    @property
    def param_names(self):
        return tuple(self.param_rvs.keys())

    def get_sampler(self, nwalkers=None, savefile=None, **kwargs):
        if nwalkers is not None:
            self._nwalkers = nwalkers

        if savefile is not None:
            storage_backend = emcee.backends.HDFBackend(savefile)
            with storage_backend.open("a") as f:
                f.create_dataset("parameter_names", data=list(self.param_rvs.keys()))
        else:
            storage_backend = None

        sampler = emcee.EnsembleSampler(
            nwalkers=self._nwalkers,
            ndim=self.nparams,
            log_prob_fn=self.logprob,
            args=(self.y, self.X, self.H0),
            parameter_names=list(self.param_rvs.keys()),
            backend=storage_backend,
            **kwargs
        )
        return sampler

    def sample_priors(self, size=1):
        """Returns the prior sampled as a matrix of shape = (size, ndims)"""
        return np.array([rv.rvs(size) for rv in self.param_rvs.values()]).T

    def get_hubble_residual(self, burn=0):
        sampler = self.get_sampler()
        trace = az.from_emcee(sampler, var_names=list(self.param_rvs.keys())).sel(
            slice=(burn, None)
        )
        params = trace.get("posterior").mean()
        mu_theoretical = get_mu_th(self.H0, params["Om0"], self.X["redshift"])
        m_obs = self.y
        m_pred = self._get_abs_mag_and_err(
            mu_theoretical,
            params["M_B"],
            params["alpha"],
            self.X["stretch"],
            params["beta"],
            self.X["color"],
            mass_correction,
        )

        return m_obs - m_pred

    #################
    # BAYESIAN MODEL
    #################
    def logprob(self, params, y, X, H0):
        lnp = self.logprior(params)

        if not np.isfinite(lnp):
            return -np.inf

        lnl = self.loglike(params, y, X, H0)
        return lnp + lnl

    def logprior(self, params):
        lnp = 0

        for key, param_rv in self.param_rvs.items():
            param_lnp = param_rv.logpdf(params[key])
            if np.isfinite(param_lnp):
                lnp += param_lnp
            else:
                return -np.inf

        return lnp

    def loglike(self, params, y, X, H0):
        from .distmod_cosmology import get_mu_th

        try:
            mu_theoretical = get_mu_th(H0, params["Om0"], X["redshift"])
        except ValueError as e:
            raise e

        if self.host_mass_correction_model == "step":
            mass_correction = mass_corrections.step(
                X["host_mass"], params["loc"], params["size"]
            )
        elif self.host_mass_correction_model == "sigmoid":
            mass_correction = mass_corrections.sigmoid(
                X["host_mass"], params["loc"], params["size"], params["slope"]
            )
        else:
            raise AssertionError("`self.host_mass_correction_model` is invalid")

        m_obs = self.y
        m_obs_sigma = X["mag_sigma"]

        m_pred = self._get_abs_mag_and_err(
            mu_theoretical,
            params["M_B"],
            params["alpha"],
            X["stretch"],
            params["beta"],
            X["color"],
            mass_correction,
        )
        m_pred_sigma = np.sqrt(
            (X["stretch_sigma"] * params["alpha"]) ** 2
            + (X["color_sigma"] * params["beta"]) ** 2
            + params["sigma_int"] ** 2
        )

        lnl = (
            st.norm(m_pred, np.sqrt(m_obs_sigma ** 2 + m_pred_sigma ** 2))
            .logpdf(m_obs)
            .sum()
        )
        return lnl if np.isfinite(lnl) else -np.inf

    ##################
    # PRIVATE METHODS
    ##################

    def _get_abs_mag_and_err(
        self, mu_th, M_B, alpha, stretch, beta, color, mass_correction=0
    ):
        return mu_th + M_B + alpha * stretch + beta * color + mass_correction

    def get_host_mass_rv(self, host_mass):
        # Return different types of pm.Model depending on the first element of `host_mass`.
        if np.isscalar(host_mass[0]):
            return st.norm(loc=host_mass, scale=0)
        elif host_mass.shape[1] == 2:
            mus, sigmas = host_mass[:, 0], host_mass[:, 1]
            return st.norm(loc=mus, scale=sigmas)
        else:
            return self._interpolate_posterior_samples(r"$M_gal$", host_mass)

    def _interpolate_posterior_samples(self, name, samples):
        """https://docs.pymc.io/notebooks/updating_priors.html"""
        smin, smax = np.min(samples), np.max(samples)
        width = smax - smin
        x = np.linspace(smin, smax, 100)
        y = st.gaussian_kde(samples)(x)

        # what was never sampled should have a small probability but not 0,
        # so we'll extend the domain and use linear approximation of density on it
        x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
        y = np.concatenate([[0], y, [0]])
        return Interpolated(name, x, y)
