"""Visualization of POD singular values and ROM relative errors."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.rcParams.update({
    "font.size": 35,
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.weight": "light",
    "lines.linewidth": 1.5,
    "figure.figsize": (10, 8),
    "legend.fontsize": 30,
    "axes.labelsize": 40,
    "legend.loc": "best",
    "lines.markersize": 8,
    "legend.frameon": True,
})

class Plot:
    """Visualization of singular values and relative errors."""

    LABELS = {"velocity": "velocity", "c-trace": r"tr($\mathbf{c}$)", "mesh_coor2": "height"}
    TRAIN = dict(color="black", marker="+", markersize=14, markeredgewidth=2,
                 linestyle="none", zorder=1)

    def __init__(self, fom, online):
        self.fom, self.online = fom, online

    def plot_singular_values(self, pod):
        _, ax = plt.subplots(figsize=(8, 6))
        for field, d in pod.data.items():
            s = d["svals"]
            ax.semilogy(np.arange(1, len(s) + 1), s, "o-", label=self.LABELS.get(field, field))
        ax.set(xlabel="index", ylabel=r"$\sigma_i/\sigma_1$")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.show()

    def plot_relative_error(self, param_labels=(r"$\lambda$", r"$\beta$"),
                            fields=("velocity", "c-trace"),
                            titles=(r"$\varepsilon_u$", r"$\varepsilon_c$"),
                            vmin=1e-6, vmax=1e-2, n_levels=8, colormap="coolwarm"):
        cols = self.fom.param_cols
        Xtr, Xte = self.fom.parameters_train[:, cols], self.fom.parameters_test[:, cols]
        one_d = len(cols) == 1
        cmap, norm = plt.get_cmap(colormap, n_levels), LogNorm(vmin, vmax)

        fig, axes = plt.subplots(1, len(fields), figsize=(8 * len(fields), 6),
                                 layout="constrained")
        for ax, field, title in zip(axes, fields, titles):
            err = self.online.errors[field]
            if one_d:
                ax.semilogy(Xte[:, 0], err, "o", markersize=10, zorder=2)
                ax.plot(Xtr[:, 0], np.full(len(Xtr), 0.03),
                        transform=ax.get_xaxis_transform(), **self.TRAIN)
            else:
                ax.plot(Xtr[:, 0], Xtr[:, 1], **self.TRAIN)
                sc = ax.scatter(Xte[:, 0], Xte[:, 1], c=np.clip(err, vmin, vmax), s=180,
                                cmap=cmap, norm=norm, zorder=2)
            ax.set(xlabel=param_labels[0], title=title)
            ax.grid(alpha=0.3)

        axes[0].set_ylabel("relative error" if one_d else param_labels[1])
        if not one_d:
            fig.colorbar(sc, ax=axes, location="bottom")
        plt.show()