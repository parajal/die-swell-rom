"""Figure 6: ROM error convergence with the number of POD modes.

Reproduces Figure 6 of the extrudate-swell ROM paper: the mean relative l2 error
over the test set as a function of the number of retained POD modes r, for the
velocity field, the trace of the conformation tensor, and the mesh-deformation
(height) field. For each r the ROM is rebuilt on the leading r POD modes -- the
GPR surrogate is retrained and the test snapshots reconstructed -- and the mean
relative error of Eq. (36) is recorded.

Run: python figures/figure6.py
"""

import sys
import types
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from romlab import FOM, OfflinePhase, OnlinePhase  # noqa: E402  (also sets rcParams)

DATA_DIR = ROOT / "data" / "2d" / "lambda"
TRAIN_FILES = ["snapshots.txt", "c-trace.txt", "mesh_coor2.txt", "parameters.txt"]
TEST_FILES = ["snapshots_test.txt", "c-trace_test.txt", "mesh_coor2_test.txt", "parameters_test.txt"]
FIELDS = ("velocity", "c-trace", "mesh_coor2")
LABELS = {"velocity": "velocity", "c-trace": r"tr($\mathbf{c}$)", "mesh_coor2": "mesh-deformation"}
COLORS = {"velocity": "#1f77b4", "c-trace": "#ff7f0e", "mesh_coor2": "#2ca02c"}
CENTERING = False  # plain SVD spectrum (paper Algorithm 1), gives a full rank-M basis


def style():
    """Undo romlab's global rcParams; serif LaTeX to match figure4."""
    plt.rcParams.update({
        "font.size": 35, "text.usetex": True, "font.family": "serif",
        "font.weight": "light", "lines.linewidth": 1.5, "legend.fontsize": 30,
        "axes.labelsize": 35, "legend.loc": "best", "lines.markersize": 8,
        "legend.frameon": True,
    })


def full_pod(train):
    """Full (untruncated) POD of one field's training snapshots."""
    lifting = train.mean(axis=0) * CENTERING
    X = train - lifting
    U, _, _ = np.linalg.svd(X.T, full_matrices=False)  # (n_dof, M)
    return {"basis": U, "coeffs_train": X @ U, "lifting": lifting}


def mean_error_at_r(fom, field, pod_full, r):
    """Retrain the ROM on the leading r modes; return the mean relative test error."""
    d = pod_full[field]
    trunc = types.SimpleNamespace(data={field: {
        "basis": d["basis"][:, :r],
        "coeffs_train": d["coeffs_train"][:, :r],
        "lifting": d["lifting"],
    }})
    offline = OfflinePhase(fom, trunc, x_scaler="minmax")
    offline.train_rom()
    online = OnlinePhase(fom, trunc, offline)
    online.reconstruct_sol()
    return online.mean_errors[field]


def main():
    plt.switch_backend("TkAgg")  # ensure an interactive backend to show the figure
    style()

    fom = FOM(DATA_DIR, TRAIN_FILES, TEST_FILES, param_cols=[0], seed=42)
    fom.info()

    pod_full = {f: full_pod(fom.train[f]) for f in FIELDS}
    n_max = min(pod_full[f]["basis"].shape[1] for f in FIELDS)
    modes = np.arange(1, n_max + 1)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for field in FIELDS:
        errors = [mean_error_at_r(fom, field, pod_full, r) for r in modes]
        ax.semilogy(modes, errors, "o-", color=COLORS[field], label=LABELS[field])

    ax.set(xlabel="$r$", ylabel=r"$\bar{\varepsilon}$")
    ax.set_xticks(np.arange(2, n_max + 1, 2))
    ax.set_xlim(1, n_max)
    ax.set_ylim(1e-7, 1e-1)
    ax.set_yticks([1e-7, 1e-4, 1e-1])
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.savefig(Path(__file__).resolve().parent / "figure6.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
