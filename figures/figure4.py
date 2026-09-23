"""Figure 4: normalized POD singular values.

Reproduces Figure 4 of the extrudate-swell ROM paper: the normalized singular
value spectra sigma_i / sigma_1 for the velocity field, the trace of the
conformation tensor, and the mesh-deformation (height) field, over the
one-parameter sweep lambda in [1, 10], beta = 0.2.

Run: python figures/figure4.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from romlab import FOM, POD  

DATA_DIR = ROOT / "data" / "2d" / "lambda"
TRAIN_FILES = ["snapshots.txt", "c-trace.txt", "mesh_coor2.txt", "parameters.txt"]
TEST_FILES = ["snapshots_test.txt", "c-trace_test.txt", "mesh_coor2_test.txt", "parameters_test.txt"]
FIELDS = ("velocity", "c-trace", "mesh_coor2")
LABELS = {"velocity": "velocity", "c-trace": r"tr($\mathbf{c}$)", "mesh_coor2": "mesh-deformation"}
COLORS = {"velocity": "#1f77b4", "c-trace": "#ff7f0e", "mesh_coor2": "#2ca02c"}


def style():
    """Undo romlab's global rcParams; serif LaTeX to match figure3."""
    plt.rcParams.update({
        "font.size": 35, "text.usetex": True, "font.family": "serif",
        "font.weight": "light", "lines.linewidth": 1.5, "legend.fontsize": 30,
        "axes.labelsize": 35, "legend.loc": "best", "lines.markersize": 8,
        "legend.frameon": True,
    })


def main():
    plt.switch_backend("TkAgg")  # ensure an interactive backend to show the figure
    style()

    fom = FOM(DATA_DIR, TRAIN_FILES, TEST_FILES, param_cols=[0], seed=42)
    fom.info()

    pod = POD(fom, centering=False, eps=1e-6)  # plain SVD spectrum (paper Algorithm 1)
    pod.compute_basis()

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for field in FIELDS:
        s = pod.data[field]["svals"]
        ax.semilogy(np.arange(1, len(s) + 1), s, "o-",
                    color=COLORS[field], label=LABELS[field])
    ax.set(xlabel="index", ylabel=r"$\sigma_i/\sigma_1$")
    ax.set_xticks([1, 5, 10])
    ax.set_xlim(1, 10)
    ax.set_ylim(1e-10, 1e0)
    ax.set_yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    plt.savefig(Path(__file__).resolve().parent / "figure4.pdf", bbox_inches = "tight")
    plt.show()


if __name__ == "__main__":
    main()
