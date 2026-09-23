"""Figure 16: spatial ROM reconstruction errors, 3D cylindrical case.

Based on Figure 16 of the extrudate-swell ROM paper: the pointwise reconstruction
error between the full-order model and the POD-GPR reduced model for the velocity
magnitude and the trace of the conformation tensor, at two representative test
parameters of the three-dimensional configuration, rendered on the swollen
quarter-cylinder (full-order mesh_coor2, swollen radially) with the same view,
element edges and colorbar style as Figure 13.

Rows are the two parameter points; columns are velocity (left) and tr(c) (right).
Each panel has its own logarithmic colour scale spanning LOG_DECADES decades below
its maximum error; smaller errors (including exact zeros) take the lowest colour.

Run: python figures/figure16.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from romlab import FOM, POD, OfflinePhase  # noqa: E402  (also sets rcParams)
from figure13 import swollen_points, render_panel, add_colorbar, fitted_camera  # noqa: E402

DATA_DIR = ROOT / "data" / "3d" / "lambda-beta-3d"
MESH_FILE = DATA_DIR / "mesh.vtk"
TRAIN_FILES = ["snapshots.txt", "c-trace.txt", "parameters.txt"]
TEST_FILES = ["snapshots_test.txt", "c-trace_test.txt", "parameters_test.txt"]
FIELDS = ("velocity", "c-trace")
PLOT_FIELDS = ("velocity", "c-trace")
TITLES = {"velocity": r"$\lVert \boldsymbol{u}_{\mathrm{fom}}-\boldsymbol{u}_{\mathrm{rom}} \rVert$",
          "c-trace": r"$\lvert \mathrm{tr}(\boldsymbol{c})_{\mathrm{fom}}-\mathrm{tr}(\boldsymbol{c})_{\mathrm{rom}} \rvert$"}
TARGETS = [(2.0034555082770935, 0.5837844835063660),
           (7.8795018187719128, 0.6244243653067247)]
PARAM_COLS = [0, 1]
NMODES = 40
LOG_DECADES = 3  # colour range below each panel's maximum error


def style():
    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}", "font.size": 20,
    })


def reconstruct(fom, pod, offline, field):
    X = fom.parameters_test[:, fom.param_cols]
    if offline.scaler is not None:
        X = offline.scaler.transform(X)
    d = pod.data[field]
    coeffs = offline.models[field].predict(X).reshape(len(X), -1)
    return coeffs @ d["basis"].T + d["lifting"]


def node_error(field, rom_row, truth_row):
    if field == "velocity":  # interleaved [ux, uy, uz] -> per-node vector norm
        return np.linalg.norm((rom_row - truth_row).reshape(-1, 3), axis=1)
    return np.abs(rom_row - truth_row)


def main():
    plt.switch_backend("TkAgg")  # ensure an interactive backend to show the figure
    style()

    fom = FOM(DATA_DIR, TRAIN_FILES, TEST_FILES, param_cols=PARAM_COLS, fields=FIELDS, seed=42)
    fom.info()
    pod = POD(fom, centering=True, eps=1e-6)
    pod.compute_basis(nmodes=NMODES)
    offline = OfflinePhase(fom, pod, x_scaler="minmax")
    offline.train_rom()

    lam_beta = fom.parameters_test[:, :2]
    rows = [int(np.argmin(np.hypot(lam_beta[:, 0] - t[0], lam_beta[:, 1] - t[1]))) for t in TARGETS]

    rom = {f: reconstruct(fom, pod, offline, f) for f in PLOT_FIELDS}
    mc2 = np.loadtxt(DATA_DIR / "mesh_coor2_test.txt")  # same row order as parameters_test
    mesh = pv.read(str(MESH_FILE))
    ref_pts = np.asarray(mesh.points).copy()
    pts = {row: swollen_points(ref_pts, mc2[row]) for row in rows}
    cam = fitted_camera(mesh, pts[max(rows, key=lambda r: mc2[r].max())])  # most swollen geometry

    fig, axes = plt.subplots(len(rows), len(PLOT_FIELDS), figsize=(10.0, 6.5))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.94, bottom=0.08, wspace=0.05, hspace=0.65)
    for r, row in enumerate(rows):
        lam, bet = lam_beta[row]
        for c, field in enumerate(PLOT_FIELDS):
            ax = axes[r, c]
            err = node_error(field, rom[field][row], fom.test[field][row])
            vmax = float(err.max())
            clim = (vmax * 10.0 ** -LOG_DECADES, vmax)
            ax.imshow(render_panel(mesh, pts[row], err, clim, cam, log=True))
            ax.axis("off")
            ax.set_title(rf"$(\lambda,\beta)=({lam:.2f},{bet:.2f})$",fontsize = 16)
            add_colorbar(fig, ax, TITLES[field], *clim, log=True)

    plt.savefig(Path(__file__).resolve().parent / "figure16.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
