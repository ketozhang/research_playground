from pymc3 import math


def sigmoid(host_mass, loc, size, slope):
    return size / (1 + math.exp(-slope * (host_mass - loc)))


def step(host_mass, loc, size):
    return math.where(math.lt(host_mass, loc), 0, size)
