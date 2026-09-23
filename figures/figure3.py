"""Figure 3: first three POD coefficients vs lambda.

Reproduces Figure 3 of the extrudate-swell ROM paper: the first three POD
coefficients for the velocity field (top row) and the trace of the conformation
tensor (bottom row), over the one-parameter sweep lambda in [1, 10], beta = 0.2.
Open circles are the projected training coefficients; the dashed curve is the
GPR predicted mean. Snapshots are centered so the 2nd/3rd coefficient curves
cross zero, as in the paper.

Run: python figures/figure3.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from romlab import FOM, POD, OfflinePhase  # noqa: E402  (also sets global rcParams)

DATA_DIR = ROOT / "data" / "2d" / "lambda"
TRAIN_FILES = ["snapshots.txt", "c-trace.txt", "mesh_coor2.txt", "parameters.txt"]
TEST_FILES = ["snapshots_test.txt", "c-trace_test.txt", "mesh_coor2_test.txt", "parameters_test.txt"]
FIELDS = ("velocity", "c-trace")
ORDINALS = (r"1\textsuperscript{st}", r"2\textsuperscript{nd}", r"3\textsuperscript{rd}")


def integer_sci_yaxis(ax):
    """Integer y tick labels with a common 10^n factor (e.g. 200 -> 2 x10^2).

    Ticks are placed at a 1/2/5 step and the label is divided by that step's
    power of ten, so every label is a whole number (no 1.5 or 2.5).
    """
    lo, hi = ax.get_ylim()
    span = hi - lo
    if span <= 0:
        return
    k = int(np.floor(np.log10(span / 5)))
    base = (span / 5) / 10.0 ** k
    mult = 1 if base < 1.5 else 2 if base < 3.5 else 5 if base < 7.5 else 10
    if mult == 10:
        mult, k = 1, k + 1
    step, scale = mult * 10.0 ** k, 10.0 ** k
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: rf"${round(v / scale):d}$"))
    if k != 0:
        ax.text(0.0, 1.0, rf"$\times 10^{{{k}}}$", transform=ax.transAxes,
                va="bottom", ha="left")


def style():
    """Undo romlab's global LaTeX / large-font rcParams; use bundled mathtext."""
    plt.rcParams.update({
        "font.size": 35,
        "text.usetex": True,
        "font.family": "serif",
        "font.weight": "light",
        "lines.linewidth": 1.5,
        "legend.fontsize": 30,
        "axes.labelsize": 35,
        "legend.loc": "best",
        "lines.markersize": 8,
        "legend.frameon": True
    })


def orient_modes(data, order, field):
    """Fix the arbitrary SVD sign of the first three modes to match the paper.

    Mode 1 increases with lambda; modes 2 and 3 start negative for velocity and
    positive for tr(c). Basis and coefficients flip together, so the
    reconstruction is unchanged.
    """
    if data["nmodes"] < 3:
        raise ValueError(f"{field}: needs three retained modes, has {data['nmodes']}")
    a = data["coeffs_train"][order]
    start = -1.0 if field == "velocity" else 1.0
    direction = np.array([a[-1, 0] - a[0, 0], start * a[0, 1], start * a[0, 2]])
    signs = np.where(direction < 0, -1.0, 1.0)
    data["basis"][:, :3] *= signs
    data["coeffs_train"][:, :3] *= signs


def main():
    plt.switch_backend("TkAgg")  # ensure an interactive backend to show the figure
    style()

    fom = FOM(DATA_DIR, TRAIN_FILES, TEST_FILES, param_cols=[0], seed=42)
    fom.info()

    pod = POD(fom, centering=True, eps=1e-6)
    pod.compute_basis()
    order = np.argsort(fom.parameters_train[:, 0])
    for field in FIELDS:
        orient_modes(pod.data[field], order, field)

    offline = OfflinePhase(fom, pod, x_scaler="minmax")
    offline.train_rom() 

    lam = fom.parameters_train[order, 0]
    dense = np.linspace(lam.min(), lam.max(), 200)
    query = offline.scaler.transform(dense[:, None])

    fig, axes = plt.subplots(2, 3, figsize=(24.0, 12.0), constrained_layout = True)
    for row, field in enumerate(FIELDS):
        train_coeffs = pod.data[field]["coeffs_train"][order, :3]
        pred_coeffs = offline.models[field].predict(query)[:, :3]
        for col, ax in enumerate(axes[row]):
            ax.plot(dense, pred_coeffs[:, col], "--", color="#0072bd", label="Predicted mean")
            ax.plot(lam, train_coeffs[:, col], "o", markersize=5,
                    markerfacecolor="none", markeredgecolor="black", label="Training data")
            ax.set(xlabel=r"$\lambda$", ylabel=ORDINALS[col] + " RB coefficient")
            ax.set_xticks([1, 5, 10])
            ax.set_xlim(1, 10)
            integer_sci_yaxis(ax)
            ax.grid(True, alpha = 0.3)
            if (row, col) == (0, 0):
                ax.legend(loc="lower right")
    plt.savefig(Path(__file__).resolve().parent / "figure3.pdf")
    plt.show()


if __name__ == "__main__":
    main()
