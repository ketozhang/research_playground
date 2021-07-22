import numpy as np


def sigmoid(host_mass, loc, size, slope):
    return size / (1 + np.exp(-slope * (host_mass - loc)))


def step(host_mass, loc, size):
    return np.where(host_mass < loc, 0, size)
