import numpy as np
import pymc3 as pm
from pymc3.distributions import Interpolated
from scipy import stats


def calculate_mass_step_sigmoid(host_mass, x0, y0, alpha, beta):
    """Using a sigmoid fit, calculates the mass step for each value in `host_mass` and returns the mass step as an array"""
    pass


def calculate_mass_step_stepwise(host_mass, cutoff=1e10):
    """Using a stepwise fit, calculates the mass step for each value in `host_mass` and returns the mass step as an
    array"""
    pass


def calculate_distmod_empirical(
    alpha,
    beta,
    abs_mag,
    mass_step_fn,
    mass_step_params={},
):
    pass


def calculate_hubble_residual(
    alpha,
    beta,
    abs_mag,
    mass_step_fn,
    mass_step_params,
    redshift,
    hubble_constant,
    omega_m,
):
    """Returns the deterministic hubble residual"""
    distmod_empirical = calculate_distmod_empirical(
        alpha,
        beta,
        abs_mag,
        mass_step_fn,
        mass_step_params,
    )
    distmod_theoretical = calculate_distmod_theoretical(
        redshift,
        hubble_constant,
        omega_m,
    )
    return distmod_empirical - distmod_theoretical


def calculate_distmod_theoretical(redshift, hubble_constant, omega_m):
    pass


class BaseModel(pm.Model):
    def __init__(self):
        super().__init__()

    def fit(self, hr, host_mass):
        """[summary]

        Parameters
        ----------
        hr : array_like
            An array observed Hubble residuals of size N with each element is a float per SN.
        host_mass : array_like
            An array of host stellar mass with each element per SN must be one of the following:
                * float, the observed host mass is deterministic without uncertainty
                * array_like of size 2 with columns being `"host_mass"` and `"host_mass_err"`,
                    the observed host mass is normally distributed with mu and sigma taken from the keys respectively.
                * array_like of size >2, the observed host mass is distributed as the given posterior samples

        """
        # From now on, all variables assigned to a pm.Model subclass instances
        # are automatically added to the instance attribute (e.g., `self.var`)
        self._set_priors(host_mass)
        self._set_model()
        self._set_likelihood(hr)

    ##################
    # PRIVATE METHODS
    ##################
    def _set_priors(self, host_mass):
        # Likelihood rv prior
        hr_sigma = pm.HalfNormal("hr_sigma")

        # Distance modulus empirical (tripp equation + mass step correction)
        alpha = pm.Normal("alpha")
        beta = pm.Normal("beta")
        abs_mag = pm.Normal("abs_mag")
        host_mass = self._fit_host_mass(host_mass)

        # Distance modulus theoretical (cosmology)
        hubble_constant = pm.Normal("hubble_constant")
        omega_m = pm.Normal("omega_m")  # FIXME: Need to bound for percentage 0 to 1

    def _set_model(self):
        hr_true = pm.Deterministic(
            "HR_true",
            calculate_hubble_residual(
                self.alpha,
                self.beta,
                self.abs_mag,
                self.host_mass,
                redshift,
                self.hubble_constant,
                self.omega_m,
            ),
        )
        return hr_true

    def _set_likelihood(self, hr_obs):
        """
        HR ~ Normal
        """
        hr = pm.Normal(
            name="HR_obs", mu=self.hr_true, sigma=self.hr_sigma, observed=hr_obs
        )
        return hr

    def _fit_host_mass(self, host_mass):
        name = "host_mass"

        # Return different types of pm.Model depending on the first element of `host_mass`.
        # See docstring of `self.fit`
        if np.isscalar(host_mass[0]):
            return pm.Deterministic(name, host_mass)
        elif host_mass.shape[1] == 2:
            mus, sigmas = host_mass[:, 0], host_mass[:, 1]
            return pm.Normal(name, mu=mus, sigma=sigmas)
        else:
            return self._interpolate_posterior_samples(name, host_mass)

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
