"""Figure 7: first three reduced coefficients over the (lambda, beta) plane.

Two-parameter analogue of Figure 3 for the extrudate-swell ROM paper. The
training set is a 10x10 tensor grid in (lambda, beta); a POD basis is built and a
GPR surrogate maps (lambda, beta) to the reduced coefficients. Each of the first
three coefficients is drawn as a GPR predicted-mean surface over the parameter
plane, with the projected training coefficients as points, for the velocity
field (top row) and the trace of the conformation tensor (bottom row).

Run: python figures/figure7.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from romlab import FOM, POD, OfflinePhase  # noqa: E402  (also sets global rcParams)

DATA_DIR = ROOT / "data" / "2d" / "lambda-beta"
TRAIN_FILES = ["snapshots.txt", "c-trace.txt", "mesh_coor2.txt", "parameters.txt"]
TEST_FILES = ["snapshots_test.txt", "c-trace_test.txt", "mesh_coor2_test.txt", "parameters_test.txt"]
FIELDS = ("velocity", "c-trace")
ROW_LABELS = {"velocity": "velocity", "c-trace": r"tr($\mathbf{c}$)"}
ORDINALS = (r"1$^{\mathrm{st}}$", r"2$^{\mathrm{nd}}$", r"3$^{\mathrm{rd}}$")


def style():
    """Undo romlab's global rcParams; serif LaTeX to match the other figures."""
    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif", "font.weight": "light",
        "font.size": 30, "axes.labelsize": 30, "axes.titlesize": 30,
    })


def main():
    plt.switch_backend("TkAgg")  # ensure an interactive backend to show the figure
    style()

    fom = FOM(DATA_DIR, TRAIN_FILES, TEST_FILES, param_cols=[0, 1], seed=42)
    fom.info()
    pod = POD(fom, centering=True, eps=1e-6)
    pod.compute_basis()
    offline = OfflinePhase(fom, pod, x_scaler="minmax")
    offline.train_rom()

    lam, bet = fom.parameters_train[:, 0], fom.parameters_train[:, 1]
    ld = np.linspace(lam.min(), lam.max(), 40)
    bd = np.linspace(bet.min(), bet.max(), 40)
    LL, BB = np.meshgrid(ld, bd)
    query = offline.scaler.transform(np.column_stack([LL.ravel(), BB.ravel()]))

    # 3D axes stay square, so size the figure to the 2x3 grid (plus column gaps)
    # instead of letting spare height turn into a gap between the rows.
    fig, axes = plt.subplots(2, 3, figsize=(19.0, 12.0), subplot_kw={"projection": "3d"})
    fig.subplots_adjust(wspace=0.22, hspace=0.08, left=0.04, right=0.97, top=0.94, bottom=0.04)
    for row, field in enumerate(FIELDS):
        train_coeffs = pod.data[field]["coeffs_train"][:, :3]
        pred = offline.models[field].predict(query)[:, :3]
        for col in range(3):
            ax = axes[row, col]
            Z = pred[:, col].reshape(LL.shape)
            ax.plot_surface(LL, BB, Z, cmap="coolwarm", alpha=0.9,
                            linewidth=0, antialiased=True, rcount=40, ccount=40)
            ax.scatter(lam, bet, train_coeffs[:, col], color="red", s=8, depthshade=False)
            ax.tick_params(axis="both", pad=4)
            ax.set_xlabel(r"$\lambda$", labelpad=22)
            ax.set_ylabel(r"$\beta$", labelpad=22)
            ax.set_xticks([5, 10])
            ax.set_yticks([0.25, 0.50, 0.75])
            ax.zaxis.set_major_locator(MaxNLocator(3))
            ax.view_init(elev=22, azim=-60)
            if row == 0:
                ax.set_title(ORDINALS[col] + " RB coefficient", pad=0)

    for row, field in enumerate(FIELDS):
        pos = axes[row, 0].get_position()
        fig.text(0.01, (pos.y0 + pos.y1) / 2, ROW_LABELS[field], rotation="vertical",
                 va="center", ha="left")

    plt.savefig(Path(__file__).resolve().parent / "figure7.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
