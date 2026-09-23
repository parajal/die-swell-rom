"""Figure 11: time-averaged point-wise relative errors over (lambda, beta).

Reproduces Figure 11 of the extrudate-swell ROM paper (time-dependent case). The
ROM is trained on mu = (lambda, beta, t) using the lambda-beta-time snapshots
(10x10 grid in (lambda, beta) x 5 logarithmically spaced time instances). For
each test parameter the relative l2 error is averaged over the sampled time
instances and scattered over the (lambda, beta) plane, for the velocity field
(eps_u) and the trace of the conformation tensor (eps_c). Plus markers (+) are
the training samples.

The time-averaged errors are cached in pointwise_error_time_2p.txt: if it exists
the whole ROM computation is skipped and the figure is redrawn from it;
otherwise it is computed from scratch and written.

Run: python figures/figure11.py
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

DATA_DIR = ROOT / "data" / "2d" / "lambda-beta-time"
TRAIN_FILES = ["snapshots.txt", "c-trace.txt", "parameters.txt"]  # no mesh_coor2 for this case
TEST_FILES = ["snapshots_test.txt", "c-trace_test.txt", "parameters_test.txt"]
FIELDS = ("velocity", "c-trace")  # no mesh_coor2 for this case
TITLES = {"velocity": r"$\varepsilon_u$", "c-trace": r"$\varepsilon_c$"}
PARAM_COLS = [0, 1, 3]  # (lambda, beta, t); col 2 is the fixed alpha
NLEVELS = 8
PANEL = (8, 6)
CACHE = Path(__file__).resolve().parent / "pointwise_error_time_2p.txt"


def style():
    plt.rcParams.update({
        "font.size": 30, "text.usetex": True, "font.family": "serif",
        "legend.fontsize": 30, "axes.labelsize": 30, "axes.titlesize": 30,
        "xtick.labelsize": 30, "ytick.labelsize": 30, "legend.loc": "best",
        "lines.markersize": 12, "legend.frameon": True, "grid.alpha": 0.3,
    })


def compute_errors():
    """Train the (lambda, beta, t) ROM; return per-(lambda,beta) time-averaged errors."""
    fom = FOM(DATA_DIR, TRAIN_FILES, TEST_FILES, param_cols=PARAM_COLS, fields=FIELDS, seed=42)
    fom.info()
    pod = POD(fom, centering=True, eps=1e-5)
    pod.compute_basis()
    pod.info()
    offline = OfflinePhase(fom, pod, x_scaler="minmax")
    offline.train_rom()
    online = OnlinePhase(fom, pod, offline)
    online.reconstruct_sol()
    online.info()

    # Average each test point's error over its 5 time instances -> one value per (lambda, beta).
    lam_beta = np.round(fom.parameters_test[:, :2], 6)
    uniq, inv = np.unique(lam_beta, axis=0, return_inverse=True)
    avg = {f: np.array([online.errors[f][inv == k].mean() for k in range(len(uniq))])
           for f in FIELDS}
    return uniq[:, 0], uniq[:, 1], avg


def save_cache(lam, bet, avg):
    data = np.column_stack([lam, bet, *(avg[f] for f in FIELDS)])
    np.savetxt(CACHE, data, fmt="%.8e", header="lambda  beta  " + "  ".join(FIELDS))


def load_cache():
    data = np.atleast_2d(np.loadtxt(CACHE))
    return data[:, 0], data[:, 1], {f: data[:, i + 2] for i, f in enumerate(FIELDS)}


def main():
    plt.switch_backend("TkAgg")  # ensure an interactive backend to show the figure
    style()

    if CACHE.exists():
        print(f"Reusing cached errors, skipping computation: {CACHE}")
        lam_te, bet_te, avg = load_cache()
    else:
        lam_te, bet_te, avg = compute_errors()
        save_cache(lam_te, bet_te, avg)
        print(f"Saved errors: {CACHE}")

    # Training (lambda, beta) locations for the + markers (cheap; no ROM needed).
    ptr = np.atleast_2d(np.loadtxt(DATA_DIR / "parameters.txt"))[:, :2]
    tr = np.unique(np.round(ptr, 6), axis=0)

    all_err = np.concatenate([avg[f] for f in FIELDS])
    vmin = 10.0 ** np.floor(np.log10(all_err[all_err > 0].min()))
    vmax = 10.0 ** np.ceil(np.log10(all_err.max()))
    cmap = matplotlib.colormaps.get_cmap("coolwarm").resampled(NLEVELS)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, len(FIELDS), figsize=(PANEL[0] * len(FIELDS), PANEL[1] + 2),
                             constrained_layout=True)
    for col, (ax, field) in enumerate(zip(axes, FIELDS)):
        ax.plot(tr[:, 0], tr[:, 1], "+", color="k", markersize=12, markeredgewidth=2.0,
                linestyle="none", zorder=5)
        sc = ax.scatter(lam_te, bet_te, c=np.clip(avg[field], vmin, vmax), s=200,
                        linewidths=1.0, cmap=cmap, norm=norm, zorder=2)
        ax.set_title(TITLES[field])
        ax.set_xlabel(r"$\lambda$", labelpad=10)
        ax.set_xticks([1, 5, 10])
        ax.set_yticks([0.1, 0.5, 0.9])
        ax.tick_params(axis="both", pad=8)
        ax.set_box_aspect(PANEL[1] / PANEL[0])
        if col == 0:
            ax.set_ylabel(r"$\beta$", labelpad=10, rotation=0)

    decades = np.arange(int(np.log10(vmin)), int(np.log10(vmax)) + 1)
    cb = fig.colorbar(sc, ax=axes, location="bottom", ticks=10.0 ** decades, shrink=1.0, aspect=60)
    cb.ax.set_xticklabels([rf"$10^{{{p}}}$" for p in decades])

    plt.savefig(Path(__file__).resolve().parent / "figure11.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
