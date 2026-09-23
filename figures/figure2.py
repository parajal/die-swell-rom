"""Figure 2: FOM velocity magnitude and tr(c) at lambda = 1 and lambda = 10.

Reproduces Figure 2 of the extrudate-swell ROM paper: the full-order velocity
magnitude and trace of the conformation tensor for the 2D planar case at the two
relaxation times lambda = 1 (top row) and lambda = 10 (bottom row). The nodal
fields are drawn on mesh.vtk, deformed in y by the mesh_coor2 (height) field so
each panel shows the corresponding swollen shape. Each panel has its own linear
colour scale.

Run: python figures/figure2.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "2d" / "lambda"
MESH_FILE = DATA_DIR / "mesh.vtk"
LAMBDAS = (1.0, 10.0)
FIELDS = ("velocity", "c-trace")
TITLES = {"velocity": r"$\lVert \boldsymbol{u} \rVert$", "c-trace": r"$\mathrm{tr}(\boldsymbol{c})$"}
CMAP = "coolwarm"


def style():
    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}", "font.size": 16,
    })


def read_mesh():
    """Reference x-coordinates and linear triangles from mesh.vtk (quadratic tris)."""
    mesh = pv.read(str(MESH_FILE))
    x = np.asarray(mesh.points)[:, 0]
    v0, v1, v2, e01, e12, e20 = mesh.cells.reshape(-1, 7)[:, 1:].T
    tris = np.vstack([np.c_[v0, e01, e20], np.c_[e01, v1, e12],
                      np.c_[e20, e12, v2], np.c_[e01, e12, e20]])
    return x, tris


def nodal_values(field, row):
    """Per-node scalar to plot: velocity magnitude or tr(c)."""
    if field == "velocity":  # interleaved [ux, uy] -> per-node magnitude
        return np.linalg.norm(row.reshape(-1, 2), axis=1)
    return row


def main():
    plt.switch_backend("TkAgg")  # ensure an interactive backend to show the figure
    style()

    params = np.atleast_2d(np.loadtxt(DATA_DIR / "parameters.txt"))
    data = {"velocity": np.loadtxt(DATA_DIR / "snapshots.txt"),
            "c-trace": np.loadtxt(DATA_DIR / "c-trace.txt")}
    mc2 = np.loadtxt(DATA_DIR / "mesh_coor2.txt")
    x, tris = read_mesh()
    rows = [int(np.argmin(np.abs(params[:, 0] - L))) for L in LAMBDAS]

    ymax = max(mc2[row].max() for row in rows)  # common height so all panels match size

    fig, axes = plt.subplots(len(LAMBDAS), len(FIELDS), figsize=(16.0, 6.0))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.03, wspace=0.12, hspace=0.9)
    for r, (lam, row) in enumerate(zip(LAMBDAS, rows)):
        tri = mtri.Triangulation(x, mc2[row], tris)  # swollen mesh: y = height field
        for c, field in enumerate(FIELDS):
            ax = axes[r, c]
            vals = nodal_values(field, data[field][row])
            tcf = ax.tricontourf(tri, vals, levels=6, cmap=CMAP)
            ax.set_aspect("equal")
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(0, ymax)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(rf"{TITLES[field]}, $\lambda={lam:.0f}$")
            cax = ax.inset_axes([0.0, -0.22, 1.0, 0.07])  # colorbar spans exactly the plot width
            fig.colorbar(tcf, cax=cax, orientation="horizontal")

    plt.savefig(Path(__file__).resolve().parent / "figure2.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
