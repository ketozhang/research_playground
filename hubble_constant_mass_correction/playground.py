import arviz as az
import pymc3 as pm

from data import dataloader
from models.pymc3_models import Model

df = dataloader.get_zpeg(1)
sne_data = df[["mag", "redshift", "stretch", "color", "host_mass"]]

model = Model(
    sne_data, hubble_constant=70, omega_m=0.3, host_mass_correction_model="step"
)
with model:
    trace = pm.sample(1000, chains=1, tune=1000)
    az.plot_trace(trace)
