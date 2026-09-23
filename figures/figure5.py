"""Figure 5: spatial reconstruction error fields.

Based on Figure 5 of the extrudate-swell ROM paper: the absolute reconstruction
error between the full-order model and the POD-GPR reduced model for the
velocity magnitude and the trace of the conformation tensor, at the two
representative test parameters lambda ~ 1.5 (top row) and lambda ~ 9.5 (bottom
row).

Each panel is rendered with PyVista on the ROM-predicted swollen mesh: the
element connectivity and reference x-coordinates come from mesh.vtk, while the
mesh_coor2 (height) field is itself reconstructed by the ROM and used to shift
every node in y. The four screenshots share one camera, one log color scale and
one horizontal colorbar so they are directly comparable.

Run: python figures/figure5.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from romlab import FOM, POD, OfflinePhase  # noqa: E402  (also sets global rcParams)

DATA_DIR = ROOT / "data" / "2d" / "lambda"
MESH_FILE = DATA_DIR / "mesh.vtk"
TRAIN_FILES = ["snapshots.txt", "c-trace.txt", "mesh_coor2.txt", "parameters.txt"]
TEST_FILES = ["snapshots_test.txt", "c-trace_test.txt", "mesh_coor2_test.txt", "parameters_test.txt"]
TARGET_LAMBDAS = (1.5, 9.5)  # representative test parameters
CMAP, NCOLORS = "coolwarm", 10  # discrete log color levels shared by all panels


def style():
    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}", "font.size":20,
    })


def reconstruct(fom, pod, offline, field):
    """ROM prediction u_rom(mu) = Phi a(mu) + lifting for every test parameter."""
    X = fom.parameters_test[:, fom.param_cols]
    if offline.scaler is not None:
        X = offline.scaler.transform(X)
    d = pod.data[field]
    coeffs = offline.models[field].predict(X).reshape(len(X), -1)
    return coeffs @ d["basis"].T + d["lifting"]


def node_error(field, rom_row, truth_row):
    """Pointwise absolute error |u_rom - u_fom| at each mesh node."""
    if field == "velocity":  # interleaved [ux, uy] -> per-node vector norm
        return np.linalg.norm((rom_row - truth_row).reshape(-1, 2), axis=1)
    return np.abs(rom_row - truth_row)  # scalar tr(c)


def render_panel(mesh, y, err, clim, window, cam):
    """Off-screen PyVista screenshot of one error field on the swollen mesh."""
    m = mesh.copy()
    pts = np.asarray(m.points)
    pts[:, 1], pts[:, 2] = y, 0.0  # ROM-predicted height shifts nodes in y
    m.points = pts
    m["err"] = np.clip(err, *clim)

    pl = pv.Plotter(off_screen=True, window_size=window)
    pl.add_mesh(m, scalars="err", cmap=CMAP, clim=list(clim), n_colors=NCOLORS,
                log_scale=True, show_scalar_bar=False)
    pl.enable_parallel_projection()
    pl.view_xy()
    pl.set_background("white")
    (cx, cy), half = cam
    pl.camera.focal_point = (cx, cy, 0.0)
    pl.camera.position = (cx, cy, 1.0)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.parallel_scale = half
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def main():
    plt.switch_backend("TkAgg")  # ensure an interactive backend to show the figure
    style()

    fom = FOM(DATA_DIR, TRAIN_FILES, TEST_FILES, param_cols=[0], seed=42)
    fom.info()
    pod = POD(fom, centering=True, eps=1e-6)
    pod.compute_basis()
    offline = OfflinePhase(fom, pod, x_scaler="minmax")
    offline.train_rom()

    lam_test = fom.parameters_test[:, 0]
    rows = [int(np.argmin(np.abs(lam_test - t))) for t in TARGET_LAMBDAS]
    fields = ("velocity", "c-trace")
    titles = {"velocity": r"$\lVert \boldsymbol{u}_{\mathrm{fom}}-\boldsymbol{u}_{\mathrm{rom}} \rVert$",
              "c-trace": r"$\lvert \mathrm{tr}(\boldsymbol{c})_{\mathrm{fom}}-\mathrm{tr}(\boldsymbol{c})_{\mathrm{rom}} \rvert$"}

    rom = {f: reconstruct(fom, pod, offline, f)
           for f in ("velocity", "c-trace", "mesh_coor2")}
    mesh = pv.read(str(MESH_FILE))
    x = np.asarray(mesh.points)[:, 0]
    ys = {row: rom["mesh_coor2"][row] for row in rows}

    errs = {(r, c): node_error(field, rom[field][row], fom.test[field][row])
            for r, row in enumerate(rows) for c, field in enumerate(fields)}
    vmax = max(e.max() for e in errs.values())
    vmin = vmax * 1e-6
    clim = (vmin, vmax)

    # One shared camera framing the tallest swollen mesh, so all panels share scale.
    x0, x1 = x.min(), x.max()
    ymax = max(y.max() for y in ys.values()) 
    cam = (((x0 + x1) / 2, ymax / 2), ymax / 2)
    window = (1600, int(1600 * ymax / (x1 - x0)))

    fig, axes = plt.subplots(len(rows), len(fields), figsize=(16.0, 6.0),
                             constrained_layout=True)
    for r, row in enumerate(rows):
        for c, field in enumerate(fields):
            ax = axes[r, c]
            ax.imshow(render_panel(mesh, ys[row], errs[(r, c)], clim, window, cam))
            ax.axis("off")
            ax.set_title(rf"{titles[field]},\ $\lambda={lam_test[row]:.1f}$")

    boundaries = np.logspace(np.log10(vmin), np.log10(vmax), NCOLORS + 1)
    cmap = mpl.colormaps.get_cmap(CMAP).resampled(NCOLORS)
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.BoundaryNorm(boundaries, NCOLORS), cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", boundaries=boundaries,
                        ticks=boundaries, spacing="uniform", shrink=1.0, aspect=60)
    cbar.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:.0e}"))
    cbar.ax.tick_params(length=0)

    plt.savefig(Path(__file__).resolve().parent / "figure5.pdf")
    plt.show()


if __name__ == "__main__":
    main()
