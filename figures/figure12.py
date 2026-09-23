"""Figure 12: ROM error convergence with POD modes, time-dependent case.

Reproduces Figure 12 of the extrudate-swell ROM paper: the time-averaged mean
relative l2 error over the test set as a function of the number of retained POD
modes, for the velocity field and the trace of the conformation tensor, in the
transient (lambda, beta, t) problem. For each mode count the POD is truncated to
that many modes (POD.compute_basis(nmodes=...)), the GPR surrogate retrained and
the test snapshots reconstructed, and the mean relative error recorded.

The convergence table is cached in mean_error_conv_time_2p.txt: if it exists the
whole computation is skipped and the figure redrawn from it; otherwise it is
computed from scratch and written.

Run: python figures/figure12.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from romlab import FOM, POD, OfflinePhase, OnlinePhase  # noqa: E402  (also sets rcParams)

DATA_DIR = ROOT / "data" / "2d" / "lambda-beta-time"
TRAIN_FILES = ["snapshots.txt", "c-trace.txt", "parameters.txt"]  # no mesh_coor2 for this case
TEST_FILES = ["snapshots_test.txt", "c-trace_test.txt", "parameters_test.txt"]
FIELDS = ("velocity", "c-trace")
LABELS = {"velocity": "velocity", "c-trace": r"tr($\mathbf{c}$)"}
COLORS = {"velocity": "#1f77b4", "c-trace": "#ff7f0e"}
PARAM_COLS = [0, 1, 3]  # (lambda, beta, t); col 2 is the fixed alpha
NMODES = [5, 10, 15, 20, 30, 40, 60, 80]  # POD mode counts to sweep
CACHE = Path(__file__).resolve().parent / "mean_error_conv_time_2p.txt"


def style():
    """Undo romlab's global rcParams; serif LaTeX to match figure9."""
    plt.rcParams.update({
        "font.size": 30, "text.usetex": True, "font.family": "serif",
        "font.weight": "light", "lines.linewidth": 1.5, "legend.fontsize": 30,
        "axes.labelsize": 30, "legend.loc": "best", "lines.markersize": 8,
        "legend.frameon": True,
    })


def compute_convergence():
    """Sweep the number of POD modes; return (modes, {field: mean test errors})."""
    fom = FOM(DATA_DIR, TRAIN_FILES, TEST_FILES, param_cols=PARAM_COLS, fields=FIELDS, seed=42)
    fom.info()

    errors = {f: [] for f in FIELDS}
    for r in NMODES:
        pod = POD(fom, centering=True, eps=1e-5)
        pod.compute_basis(nmodes=r)  # force exactly r modes per field
        offline = OfflinePhase(fom, pod, x_scaler="minmax")
        offline.train_rom()
        online = OnlinePhase(fom, pod, offline)
        online.reconstruct_sol()
        for f in FIELDS:
            errors[f].append(online.mean_errors[f])
        print(f"r={r:2d}: " + ", ".join(f"{f}={online.mean_errors[f]:.3e}" for f in FIELDS))

    return np.array(NMODES), {f: np.array(errors[f]) for f in FIELDS}


def save_cache(modes, errors):
    data = np.column_stack([modes, *(errors[f] for f in FIELDS)])
    np.savetxt(CACHE, data, fmt=["%d", *["%.8e"] * len(FIELDS)],
               header="r  " + "  ".join(FIELDS))


def load_cache():
    data = np.atleast_2d(np.loadtxt(CACHE))
    modes = data[:, 0].astype(int)
    return modes, {f: data[:, i + 1] for i, f in enumerate(FIELDS)}


def main():
    plt.switch_backend("TkAgg")  # ensure an interactive backend to show the figure
    style()

    if CACHE.exists():
        print(f"Reusing cached convergence, skipping computation: {CACHE}")
        modes, errors = load_cache()
    else:
        modes, errors = compute_convergence()
        save_cache(modes, errors)
        print(f"Saved convergence: {CACHE}")

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for field in FIELDS:
        ax.semilogy(modes, errors[field], "o-", color=COLORS[field], label=LABELS[field])

    ax.set(xlabel="$r$", ylabel=r"$\bar{\varepsilon}$")
    ax.set_xticks([20, 40, 60, 80])
    ax.set_xlim(modes[0], modes[-1])
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.savefig(Path(__file__).resolve().parent / "figure12.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
