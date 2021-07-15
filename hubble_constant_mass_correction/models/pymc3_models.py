import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3.distributions import Interpolated
from scipy import stats

from . import mass_corrections
from .distmod_cosmology import get_mu_th
from .distmod_tripp import get_mu_tripp


class Model(pm.Model):
    def __init__(self, sne_data, hubble_constant, host_mass_correction_model):
        r"""SNe standardization model with mass step correction

        \mu = m_B - (M_B + \alpha \cdot x + \beta \cdot c + \delta m_*) + \epsilon

        More useful for MCMC is to rewrite this in the form of "true = model + error". Note the sign for the error
        term doesn't matter.

        m_B = [\mu + M_B + \alpha \cdot x + \beta \cdot c + \delta m_*] + \epsilon

        Parameters
        ----------
        sne_data : np.recarray or pd.DataFrame
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
                        the observed host mass is normally distributed with mu and sigma taken from the keys respectively.
                    * array-like of size >2, the observed host mass is distributed as the given posterior samples

        hubble_constant : float
        host_mass_correction_model : str
            The host mass correction model to use None, "step", or "sigmoid"
        """
        super().__init__()

        if isinstance(sne_data, pd.DataFrame):
            self.sne_data = sne_data.to_records(index=False)
        elif isinstance(sne_data, np.recarray):
            pass
        else:
            raise ValueError(
                "Arg `sne_data` is neither `np.recarray` nor `pd.DataFrame`."
            )

        self.sne_size = self.sne_data.shape[0]
        self.hubble_constant = hubble_constant
        self.host_mass_correction_model = host_mass_correction_model

        self.pooled_vars = []
        self.unpooled_vars = []

        self.prior()
        self.likelihood()

    def prior(self):
        # Scatter prior
        self.sigma_mag = pm.HalfNormal(
            name=r"$\sigma_{m_B}$", sigma=10, shape=self.sne_size
        )

        # Cosmology
        self.omega_m = pm.Uniform(name=r"$\Omega_m$")

        #################################
        # Define the tripp params' prior
        #################################
        self.alpha = pm.Normal(r"$\alpha$", sigma=100, shape=self.sne_size)
        self.beta = pm.Normal(r"$\beta$", sigma=100, shape=self.sne_size)
        self.abs_mag = pm.Normal("$M_B$", sigma=100, shape=1)

        self.unpooled_vars += [self.alpha, self.beta]
        self.pooled_vars.append(self.abs_mag)

        ###########################################
        # Define the mass correction params' prior
        ###########################################

        # Fit a (hyper)prior to the host mass data
        host_mass = self._fit_host_mass(self.sne_data["host_mass"])

        # Define the hyperpriors depending on which host mass correction model
        if self.host_mass_correction_model == "step":
            self.loc = pm.Normal(r"loc", sigma=100, shape=1)
            self.size = pm.Normal(r"size", sigma=100, shape=1)

            self.pooled_vars += [self.loc, self.size]

            mass_correction_f = mass_corrections.step
            mass_correction_args = (host_mass, self.loc, self.size)
        elif self.host_mass_correction_model == "sigmoid":
            self.loc = pm.Normal(r"loc", sigma=100, shape=1)
            self.size = pm.Normal(r"size", sigma=10, shape=1)
            self.slope = pm.Normal(r"slope", sigma=100, shape=1)

            self.pooled_vars += [self.loc, self.size, self.slope]

            mass_correction_f = mass_corrections.sigmoid
            mass_correction_args = (host_mass, self.loc, self.size, self.slope)
        else:
            raise ValueError()

        self.mass_correction = pm.Deterministic(
            r"$\Delta M$", mass_correction_f(*mass_correction_args)
        )
        self.unpooled_vars += [self.mass_correction]

    def likelihood(self):
        distmod_cosmology = get_mu_th(
            self.hubble_constant, self.omega_m, self.sne_data["redshift"]
        )

        mag_true = (
            distmod_cosmology
            + self.abs_mag
            + self.alpha * self.sne_data["stretch"]
            + self.beta * self.sne_data["color"]
            + self.mass_correction
        )
        self.mag_likelihood = pm.Normal(
            name="$m_B$",
            mu=mag_true,
            sigma=self.sigma_mag,
            observed=self.sne_data["mag"],
        )

    ##################
    # PRIVATE METHODS
    ##################
    def _fit_host_mass(self, host_mass):
        # Return different types of pm.Model depending on the first element of `host_mass`.
        if np.isscalar(host_mass[0]):
            return host_mass
        elif host_mass.shape[1] == 2:
            mus, sigmas = host_mass[:, 0], host_mass[:, 1]
            return pm.Normal(name=self.name, mu=mus, sigma=sigmas)
        else:
            return self._interpolate_posterior_samples(self.name, host_mass)

    def _interpolate_posterior_samples(self, name, samples):
        """https://docs.pymc.io/notebooks/updating_priors.html"""
        smin, smax = np.min(samples), np.max(samples)
        width = smax - smin
        x = np.linspace(smin, smax, 100)
        y = stats.gaussian_kde(samples)(x)

        # what was never sampled should have a small probability but not 0,
        # so we'll extend the domain and use linear approximation of density on it
        x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
        y = np.concatenate([[0], y, [0]])
        return Interpolated(name, x, y)
