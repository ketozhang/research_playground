from pathlib import Path

import arviz as az
import emcee
import matplotlib.pyplot as plt
import numpy as np
from corner import corner


class Results:
    def __init__(self, results_fpath, model=None):
        self.results_fpath = Path(results_fpath)
        self.hdf = emcee.backends.HDFBackend(str(self.results_fpath))
        self.model = model

        with self.hdf.open("r") as f:
            self.labels = list(f["parameter_names"].asstr())

        self.trace = az.from_emcee(self.hdf, var_names=self.labels)

    @property
    def parameter_names(self):
        return list(self.trace.posterior.data_vars.keys())

    def get_trace(self, burn=0):
        if burn == "auto":
            burn = self.trace.posterior.dims["draw"] // 2

        return self.trace.sel(draw=slice(burn, None))

    def estimate_parameters(self, estimator, burn="auto"):
        trace = self.get_trace(burn=burn)

        if estimator == "mean":
            trace_mean = trace.get("posterior").mean()
            params = {k: trace_mean.get(k).values for k in trace_mean}
        elif estimator == "hdi":
            trace_mean = az.hdi(trace.get("posterior"), hdi_prob=0.68).mean()
            params = {k: trace_mean.get(k).values for k in trace_mean}
        else:
            raise NotImplementedError()

        return params

    def get_hr(self, data, H0=70, host_mass=None, estimator="mean", burn="auto"):
        from models.distmod_cosmology import get_mu_th

        params = self.estimate_parameters(estimator, burn=burn)

        mu_th = get_mu_th(H0, params["Om0"], data["redshift"])

        mu_pred = self.model.predict_mu(
            data["mag"], data["stretch"], data["color"], params, host_mass
        )
        return mu_pred - mu_th

    def plot_hr_vs_x(
        self,
        x,
        data,
        estimator="mean",
        burn="auto",
        ax=None,
        mass_step_label=None,
        **kwargs
    ):
        """
        x : np.array
            Values to plot on x-axis
        """
        hr = self.get_hr(data, **kwargs)
        params = self.estimate_parameters(estimator, burn=burn)

        # Plot scatter
        ax = ax or plt.axes()
        ax.scatter(x, hr, c="k", alpha=0.33)

        # Plot binned means
        bins = np.linspace(np.min(x), np.max(x), 11)
        n, _ = np.histogram(x, bins)
        sum_hr, _ = np.histogram(x, bins, weights=hr)
        mean_hr = sum_hr / n

        bins_midpoint = (bins[:-1] + bins[1:]) / 2
        ax.scatter(bins_midpoint, mean_hr, marker="s", s=100, c="k", edgecolor="w")

        loc = params["loc"]
        xlim = ax.get_xlim()
        meanx, meany = np.mean(x[x < loc]), np.mean(hr[x < loc])
        ax.hlines(meany, xlim[0], loc, color="r")
        ax.scatter(meanx, meany, color="r", s=50, marker="s", edgecolors="k")

        meanx, meany = np.mean(x[x >= loc]), np.mean(hr[x >= loc])
        ax.hlines(meany, loc, xlim[1], color="r")
        ax.scatter(meanx, meany, color="r", s=50, marker="s", edgecolors="k")

        # Plot mass step
        ax.axvline(params["loc"], c="k", lw=2, label=mass_step_label)

        return ax

    def plot_hr_vs_host_mass(self, data, **kwargs):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            nrows=2, ncols=2, figsize=(16, 8), sharex="col", sharey="row"
        )

        self.plot_hr_vs_x(data["host_mass"], data, ax=ax1, mass_step_label="Mean")
        self.plot_hr_vs_x(data["host_mass"], data, host_mass=data["host_mass"], ax=ax2)
        ax1.set_title("w/o mass correction")
        ax2.set_title("w/ mass correction")
        ax1.legend()

        self.plot_hr_vs_x(
            data["host_mass"],
            data,
            ax=ax3,
            estimator="hdi",
            mass_step_label="68% HDI Mean",
        )
        self.plot_hr_vs_x(
            data["host_mass"],
            data,
            host_mass=data["host_mass"],
            estimator="hdi",
            ax=ax4,
        )
        ax3.legend()

        fig.subplots_adjust(wspace=0, hspace=0)
        fig.supxlabel("$\log_{10}(M/M_\odot)$")
        fig.supylabel("HR (mag)")

        return fig

    def plot_trace(self):
        trace = self.get_trace()
        return az.plot_trace(trace)

    def plot_corner(self, burn="auto", **kwargs):
        trace = self.get_trace(burn)
        fig = corner(trace, show_titles=True, **kwargs)
        fig.subplots_adjust(hspace=0, wspace=0)
