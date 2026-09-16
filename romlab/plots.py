"""
Visualization Module for Reduced-Order Modeling
Generates PDF plots of FOM and ROM solutions.
"""

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

    def __init__(self, fom, online):
        self.fom = fom
        self.online = online

    def plot_singular_values(self, pod):
        if not pod.data:
            raise ValueError("call compute_basis() first")
        labels = {"velocity": "velocity", "c-trace": r"tr($\mathbf{c}$)",
                  "mesh_coor2": "height"}

        _, ax = plt.subplots(figsize=(8, 6))
        for field in pod.computed_fields:
            s = pod.data[field]["svals"]
            ax.semilogy(np.arange(1, len(s) + 1), s, "o-",
                        label=labels.get(field, field))
        ax.set(xlabel="index", ylabel=r"$\sigma_i/\sigma_1$")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.show()

    def plot_relative_error(self, param_labels=(r"$\lambda$", r"$\beta$"),
                            fields=("velocity", "c-trace"),
                            titles=(r"$\varepsilon_u$", r"$\varepsilon_c$"),
                            vmin=1e-6, vmax=1e-2, n_levels=8, colormap="coolwarm"):
        cols = list(self.fom.param_cols)
        X_train = self.fom.parameters_train[:, cols]
        X_test = self.fom.parameters_test[:, cols]
        one_d = len(cols) == 1

        fig, axes = plt.subplots(1, len(fields), figsize=(8 * len(fields), 6),
                                 layout="constrained")
        train = dict(color="black", marker="+", markersize=14,
                     markeredgewidth=2, linestyle="none", zorder=1)
        cmap, norm = plt.get_cmap(colormap, n_levels), LogNorm(vmin, vmax)

        for ax, field, title in zip(axes, fields, titles):
            err = self._error(field)
            if one_d:
                ax.semilogy(X_test[:, 0], err, "o", markersize=10, zorder=2)
                ax.plot(X_train[:, 0], np.full(len(X_train), 0.03),
                        transform=ax.get_xaxis_transform(), **train)
            else:
                ax.plot(X_train[:, 0], X_train[:, 1], **train)
                sc = ax.scatter(X_test[:, 0], X_test[:, 1], s=180, zorder=2,
                                c=np.clip(err, vmin, vmax), cmap=cmap, norm=norm)
            ax.set_xlabel(param_labels[0])
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("relative error" if one_d else param_labels[1])
        if not one_d:
            fig.colorbar(sc, ax=axes, location="bottom")
        plt.show()
        plt.close(fig)

    def _error(self, field):
        """Per-test-sample relative error for one field."""
        if field not in self.online.errors:
            raise ValueError(f"no error for '{field}'; run "
                             f"online.reconstruct_sol({field!r}) first")
        return np.asarray(self.online.errors[field], dtype=float)