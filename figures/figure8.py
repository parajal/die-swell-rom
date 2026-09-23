"""Figure 8: point-wise relative errors over the (lambda, beta) plane.

Reproduces Figure 8 of the extrudate-swell ROM paper: the relative l2 error of
the ROM at each test parameter for the velocity field (eps_u) and the trace of
the conformation tensor (eps_c), scattered over the (lambda, beta) domain. Plus
markers (+) are the training samples; filled circles are the test parameters
coloured by their relative error on a shared log scale.

Run: python figures/figure8.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from romlab import FOM, POD, OfflinePhase, OnlinePhase  # noqa: E402  (also sets rcParams)

DATA_DIR = ROOT / "data" / "2d" / "lambda-beta"
TRAIN_FILES = ["snapshots.txt", "c-trace.txt", "mesh_coor2.txt", "parameters.txt"]
TEST_FILES = ["snapshots_test.txt", "c-trace_test.txt", "mesh_coor2_test.txt", "parameters_test.txt"]
FIELDS = ("velocity", "c-trace")
TITLES = {"velocity": r"$\varepsilon_u$", "c-trace": r"$\varepsilon_c$"}
NLEVELS = 8
PANEL = (8, 6)  # size of each subplot


def style():
    plt.rcParams.update({
        "font.size": 30, "text.usetex": True, "font.family": "serif",
        "legend.fontsize": 30, "axes.labelsize": 30, "axes.titlesize": 30,
        "xtick.labelsize": 30, "ytick.labelsize": 30, "legend.loc": "best",
        "lines.markersize": 12, "legend.frameon": True, "grid.alpha": 0.3,
    })


def main():
    plt.switch_backend("TkAgg")  # ensure an interactive backend to show the figure
    style()

    fom = FOM(DATA_DIR, TRAIN_FILES, TEST_FILES, param_cols=[0, 1], seed=42)
    fom.info()
    pod = POD(fom, centering=False, eps=1e-6)
    pod.compute_basis()
    offline = OfflinePhase(fom, pod, x_scaler="minmax")
    offline.train_rom()
    online = OnlinePhase(fom, pod, offline)
    online.reconstruct_sol()
    online.info()

    lam_tr, bet_tr = fom.parameters_train[:, 0], fom.parameters_train[:, 1]
    lam_te, bet_te = fom.parameters_test[:, 0], fom.parameters_test[:, 1]

    # Unified log colour scale, rounded out to whole decades.
    all_err = np.concatenate([online.errors[f] for f in FIELDS])
    vmin = 10.0 ** np.floor(np.log10(all_err[all_err > 0].min()))
    vmax = 10.0 ** np.ceil(np.log10(all_err.max()))
    cmap = matplotlib.colormaps.get_cmap("coolwarm").resampled(NLEVELS)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, len(FIELDS), figsize=(PANEL[0] * len(FIELDS), PANEL[1] + 2),
                             constrained_layout=True)
    for col, (ax, field) in enumerate(zip(axes, FIELDS)):
        ax.plot(lam_tr, bet_tr, "+", color="k", markersize=12, markeredgewidth=2.0,
                linestyle="none", zorder=5)
        sc = ax.scatter(lam_te, bet_te, c=np.clip(online.errors[field], vmin, vmax),
                        s=200, linewidths=1.0, cmap=cmap, norm=norm, zorder=2)
        ax.set_title(TITLES[field])
        ax.set_xlabel(r"$\lambda$", labelpad=10)
        ax.set_xticks([1, 5, 10])
        ax.set_yticks([0.1, 0.5, 0.9])
        ax.tick_params(axis="both", pad=8)
        ax.set_box_aspect(PANEL[1] / PANEL[0])  # force each axes to an 8:6 box
        if col == 0:
            ax.set_ylabel(r"$\beta$", labelpad=10, rotation=0)

    decades = np.arange(int(np.log10(vmin)), int(np.log10(vmax)) + 1)
    ticks = 10.0 ** decades
    cb = fig.colorbar(sc, ax=axes, location="bottom", ticks=ticks, shrink=1.0, aspect=60)
    cb.ax.set_xticklabels([rf"$10^{{{p}}}$" for p in decades])

    plt.savefig(Path(__file__).resolve().parent / "figure8.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
