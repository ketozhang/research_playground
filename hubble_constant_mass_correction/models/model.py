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
        H0,
        host_mass_correction_model="step",
        param_rvs=None,
    ):
        r"""SNe standardization model with mass step correction

        $$ \mu(z; \Omega_{m,0}) = m_B - (M_B + \alpha \cdot x + \beta \cdot c + \Delta m_*) + \epsilon $$

        More useful for MCMC is to rewrite this in the form of "true = model + error". Note the sign for the error
        term doesn't matter.

        $$ m_B = [\mu(z; \Omega_{m,0}) + M_B + \alpha \cdot x + \beta \cdot c + \Delta m_*] + \epsilon $$

        Parameters
        ----------
        H0 : float
            Hubble constant
        host_mass_correction_model : str
            The host mass correction model to use None, "step", or "sigmoid"
        param_rvs : dict
            With key being the parameter name and values being a random variable (`scipy.stats.rv_continuous`)
        """

        self.H0 = H0

        # Handle parameters
        self.param_rvs = {
            "sigma_int": st.halfcauchy(scale=1),
            "Om0": st.uniform(0.25, 0.1),
            "alpha": st.norm(0, 1),
            "beta": st.norm(3, 0.6),
            "M_B": st.norm(-19, 10),
            # "host_mass": self.get_host_mass_rv(self.X["host_mass"]),
            "loc": st.gamma(a=100, scale=0.1),  # mean=a*scale, var=a*scale**2
            "size": st.halfcauchy(scale=0.1),
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

    def fit(
        self,
        data=None,
        nwalkers=8,
        nsteps=1000,
        initial_state=None,
        progress=True,
        savefile=None,
        sampler_kwargs={},
        run_mcmc_kwargs={},
    ):
        """Fit model to SNe dataset.

        Parameters
        ----------
        data : pd.DataFrame
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
        """
        # Handles design matrix, store x, y, and nrows instance attributes
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise ValueError("Arg `data` must be `pd.DataFrame`.")
        self.nrows = self.data.shape[0]

        # Start MCMC fitter
        if nwalkers is not None:
            self._nwalkers = nwalkers

        if savefile is not None:
            storage_backend = emcee.backends.HDFBackend(savefile)
            with storage_backend.open("a") as f:
                f.create_dataset("parameter_names", data=list(self.param_rvs.keys()))
        else:
            storage_backend = None

        self.sampler = emcee.EnsembleSampler(
            nwalkers=self._nwalkers,
            ndim=self.nparams,
            log_prob_fn=self.logprob,
            parameter_names=list(self.param_rvs.keys()),
            backend=storage_backend,
            **sampler_kwargs
        )

        initial_state = (
            initial_state
            if (initial_state is not None)
            else self.sample_priors(nwalkers)
        )
        self.sampler.run_mcmc(
            nsteps=nsteps,
            initial_state=initial_state,
            progress=progress,
            **run_mcmc_kwargs
        )

        return self.sampler

    def predict(self, redshift, stretch, color, params, host_mass=None):
        mass_correction = (
            0 if host_mass is None else self.get_mass_correction(host_mass, params)
        )

        mu_th = get_mu_th(self.H0, params["Om0"], redshift)
        m_pred = (
            mu_th
            + params["M_B"]
            + (params["alpha"] * stretch)
            + (params["beta"] * color)
            + mass_correction
        )
        return m_pred

    def predict_mu(self, mag, stretch, color, params, host_mass=None):
        mass_correction, _ = (
            (0, 0)
            if host_mass is None
            else self.get_mass_correction(host_mass, 0, params)
        )
        return (
            get_mu_tripp(
                mag, stretch, color, params["alpha"], params["beta"], params["M_B"]
            )
            + mass_correction
        )

    def get_mass_correction(self, host_mass, host_mass_sigma, params):
        if self.host_mass_correction_model == "step":
            mass_correction = mass_corrections.step(
                host_mass, params["loc"], params["size"]
            )
            mass_correction_sigma = mass_corrections.step_sigma(
                host_mass, host_mass_sigma, params["loc"], params["size"]
            )
        elif self.host_mass_correction_model == "sigmoid":
            mass_correction = mass_corrections.sigmoid(
                host_mass, params["loc"], params["size"], params["slope"]
            )

            mass_correction_sigma = mass_corrections.sigmoid_sigma(
                host_mass,
                host_mass_sigma,
                params["loc"],
                params["size"],
                params["slope"],
            )
        else:
            raise AssertionError("`self.host_mass_correction_model` is invalid")

        return mass_correction, mass_correction_sigma

    def sample_priors(self, size=1):
        """Returns the prior sampled as a matrix of shape = (size, ndims)"""
        return np.array([rv.rvs(size) for rv in self.param_rvs.values()]).T

    #################
    # BAYESIAN MODEL
    #################
    def logprob(self, params):
        lnp = self.logprior(params)

        if not np.isfinite(lnp):
            return -np.inf

        lnl = self.loglike(params)
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

    def loglike(self, params):
        mass_correction, mass_correction_sigma = self.get_mass_correction(
            self.data["host_mass"], self.data["host_mass_sigma"], params
        )

        mu_obs = (
            self.predict_mu(
                **(self.data[["mag", "stretch", "color"]].to_dict("series")),
                params=params
            )
            + mass_correction
        )

        mu_sigma = np.sqrt(
            # Variance terms
            self.data["mag_sigma"] ** 2
            + (self.data["stretch_sigma"] * params["alpha"]) ** 2
            + (self.data["color_sigma"] * params["beta"]) ** 2
            + mass_correction_sigma ** 2
            + params["sigma_int"] ** 2
            # Covariance terms
            + 2 * params["alpha"] * self.data["cov_mag_stretch"]
            + 2 * params["beta"] * self.data["cov_mag_color"]
            + 2 * params["alpha"] * params["beta"] * self.data["cov_stretch_color"]
        )

        mu_th = get_mu_th(self.H0, params["Om0"], self.data["redshift"])

        lnl = self._loglike(mu_obs, mu_th, mu_sigma).sum()
        return lnl if np.isfinite(lnl) else -np.inf

    def _loglike(self, mu_obs, mu_th, mu_sigma):
        """Calculates the single sample log likelihood as the log PDF of the normal distribution."""
        return st.norm(mu_th, mu_sigma).logpdf(mu_obs)

    ##################
    # PRIVATE METHODS
    ##################

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


class Chi2Model(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _loglike(self, mu_obs, mu_th, mu_sigma):
        """Calculates the single sample log likelihoood as the log PDF of the normal distribution without the
        normalization factor. This effectively removes the so-called regularization effect of the optimizer."""
        return np.log(((mu_obs - mu_th) / mu_sigma) ** 2)
