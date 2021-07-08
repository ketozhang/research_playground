import astropy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from distmod_cosmology import get_mu_th
from distmod_tripp import get_mu_tripp


def get_mu_obs(m_b, x1, color, alpha, beta, M_B, host_mass, loc, size, slope, model=None):
    mu_tripp = get_mu_tripp(m_b, x1, color, alpha, beta, M_B)
    if model == None:
        return mu_tripp
    if model == "sigmoid":
        return mu_tripp + sigmoid(host_mass, loc, size, slope)
    if model == "step":
        return mu_tripp + step(host_mass, loc, size)


def get_hr(*, H0, Om0, z, m_b, x1, color, alpha, beta, M_B, host_mass, loc, size, slope, model=None):
    mu_obs = get_mu_obs(m_b, x1, color, alpha, beta, M_B, host_mass, loc, size, slope, model=model)
    mu_th = get_mu_th(H0, Om0, z)
    return  mu_obs - mu_th


###################  read files ######################

df1 = open_file(1)
df2 = open_file(2)
df3 = open_file(3)
df4 = open_file(4)
df5 = open_file(5)
df6 = open_file(6)
df7 = open_file(7)
df8 = open_file(8)
df9 = open_file(9)
df10 = open_file(10)
df11 = open_file(11)
df12 = open_file(12)


########## working on file 1 ######################
print len(df1)
df1 = df1[df1["ZPEG_StMass"] > 0]  ## only keep SN that have a host mass estimate
print len(df1)

H0 = 70
Om0 = 0.3
z = df1["zcmb"]
m_b = df1["mb"]
x1 = df1['x1']
color = df1['color']
alpha = 0.
beta = 0.
M_B = -19.1
host_mass = np.log10(df1["ZPEG_StMass"]) ## in solar mass
loc = 10 #solar mass
size = 0.2
slope = 1
model = "sigmoid"


HR1 = get_hr(H0, Om0, z, m_b, x1, color, alpha, beta, M_B, host_mass, loc, size, slope, model=model)

plt.figure(1)
plt.plot(host_mass, sigmoid(host_mass, loc, size, slope), "r.")
plt.xlabel("Log10 Mass (M_sun)")
plt.ylabel("Delta HR (mag)")
plt.show()

plt.figure(2)
plt.plot(z, HR1, "ko")
plt.xlabel("Z_CMB")
plt.ylabel("HR (mag)")
plt.show()







