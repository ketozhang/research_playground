import numpy as np
from scipy import stats as st
from scipy.special import expit  # Logistic func but with floating point stability

SAMPLE_SIZE = 100  # TODO: Increase to > 1000 with better CPU

# As slope reaches infinity logistic tends to the heaviside step function
HEAVISIDE_SLOPE = 1e10


def sigmoid(host_mass, loc, size, slope):
    """Calculates the sigmoid (a more general logistic function) scaled by a constant factor `size`."""
    return size * expit(slope * (host_mass - loc))


def step(host_mass, loc, size):
    # return np.where(host_mass < loc, 0, size)
    return sigmoid(host_mass, loc, size, slope=HEAVISIDE_SLOPE)


def sigmoid_sigma(
    host_mass, host_mass_sigma, loc, slope, size, sample_size=SAMPLE_SIZE
):
    """Estimates the standard deviation of sigmoid(X) where X~Normal(mu, sigma) by MC sampling."""
    x = st.norm(host_mass, host_mass_sigma).rvs((sample_size, len(host_mass)))
    sigmoid_x = sigmoid(x, loc, size, slope)
    return np.std(sigmoid_x, axis=0, ddof=1)


def step_sigma(host_mass, host_mass_sigma, loc, size, sample_size=SAMPLE_SIZE):
    return sigmoid_sigma(
        host_mass, host_mass_sigma, loc, size, HEAVISIDE_SLOPE, sample_size=sample_size
    )
