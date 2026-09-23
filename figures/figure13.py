"""Figure 13: FOM velocity magnitude and tr(c), 3D cylindrical case.

Reproduces Figure 13 of the extrudate-swell ROM paper: the full-order velocity
magnitude and trace of the conformation tensor for the three-dimensional
cylindrical configuration at the two relaxation times lambda = 1 (top row) and
lambda = 10 (bottom row), at fixed beta ~ 0.2. The fields are rendered on the
swollen quarter-cylinder: mesh_coor2 gives the deformed y-coordinate and the
swell is applied radially (both y and z are scaled) so the circular cross-section
stays circular. Each panel has its own continuous colour scale.

Run: python figures/figure13.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pyvista as pv
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "3d" / "lambda-beta-3d"
MESH_FILE = DATA_DIR / "mesh.vtk"
LAMBDAS = (1.0, 10.0)
BETA = 0.2  # fixed viscosity ratio (nearest grid value is used)
FIELDS = ("velocity", "c-trace")
TITLES = {"velocity": r"$\lVert \boldsymbol{u} \rVert$", "c-trace": r"$\mathrm{tr}(\boldsymbol{c})$"}
CMAP = "coolwarm"
EDGE_COLOR = "darkblue"  # dark navy element edges
VIEW_DIR = (-0.7, 0.35, 0.62)  # camera sits upstream, above and in front: flow runs lower-left -> upper-right


def style():
    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}", "font.size": 20,
    })


def nodal_values(field, row):
    if field == "velocity":  # interleaved [ux, uy, uz] -> per-node magnitude
        return np.linalg.norm(row.reshape(-1, 3), axis=1)
    return row


def swollen_points(pts, mc2):
    """Radial swell, using symmetry partners where mc2 / y is undefined.

    This quarter-cylinder mesh has matching nodes on its two symmetry planes.
    Match (x, 0, z) to (x, z, 0), retaining radial position; interpolating over
    x alone mixes interior and boundary stretches and dents the free surface.
    """
    y0, z0 = pts[:, 1], pts[:, 2]
    r0 = np.hypot(y0, z0)
    eps = 1e-3 * r0.max()
    s = np.ones(len(pts))  # axis nodes remain on the axis
    valid = y0 > eps
    s[valid] = mc2[valid] / y0[valid]
    missing = np.flatnonzero(~valid & (z0 > eps))
    partners = np.flatnonzero((z0 <= eps) & valid)
    if missing.size:
        if not partners.size:
            raise ValueError("The mesh has no opposite symmetry-plane nodes.")
        # Match in (axial coordinate, radial coordinate), not x alone.
        targets = pts[missing][:, [0, 2]]
        candidates = pts[partners][:, [0, 1]]
        distances2 = ((targets[:, None] - candidates[None, :]) ** 2).sum(axis=2)
        nearest = distances2.argmin(axis=1)
        if np.any(distances2[np.arange(len(missing)), nearest] > eps ** 2):
            raise ValueError("The mesh does not have matching symmetry-plane nodes; "
                             "supply the deformed z-coordinates instead.")
        s[missing] = s[partners[nearest]]
    new = pts.copy()
    new[:, 1] = mc2
    new[:, 2] = s * z0
    return new


def element_edges(m):
    """Curved outlines of the quadratic surface triangles (as in ParaView's 'Surface With Edges').

    One level of nonlinear subdivision splits each 6-node triangle at its mid-side
    nodes; the corner-to-midside segments trace the original element edges, while
    midside-to-midside segments are subdivision artefacts and are dropped.
    """
    s = m.extract_surface(algorithm="dataset_surface", nonlinear_subdivision=1)
    corner = np.zeros(m.n_points, bool)
    corner[m.cells_dict[pv.CellType.QUADRATIC_TETRA][:, :4].ravel()] = True
    # vtkOriginalPointIds is not reliable after subdivision; match by position instead.
    _, nearest = cKDTree(m.points).query(s.points)
    is_corner = corner[nearest]
    tri = s.faces.reshape(-1, 4)[:, 1:]
    seg = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    seg = np.unique(seg[is_corner[seg[:, 0]] != is_corner[seg[:, 1]]], axis=0)
    lines = np.column_stack([np.full(len(seg), 2), seg]).ravel()
    return pv.PolyData(np.asarray(s.points), lines=lines)


def render_panel(mesh, pts, vals, clim, cam, log=False):
    """Off-screen PyVista screenshot of one nodal field on the swollen 3D surface."""
    m = mesh.copy()
    m.points = pts
    m["f"] = np.clip(vals, *clim) if log else vals  # log colour scale cannot take zeros
    surf = m.extract_surface(algorithm="dataset_surface", nonlinear_subdivision=3)
    pl = pv.Plotter(off_screen=True, window_size=(1200, 800))
    pl.add_mesh(surf, scalars="f", cmap=CMAP, clim=list(clim), log_scale=log,
                show_scalar_bar=False, smooth_shading=True)
    # Edges are chords of the curved faces, so nudge them towards the camera to
    # keep the rounded surface from hiding them.
    edges = element_edges(m)
    to_cam = np.asarray(cam[0]) - edges.points
    edges.points = edges.points + 0.01 * to_cam / np.linalg.norm(to_cam, axis=1, keepdims=True)
    pl.add_mesh(edges, color=EDGE_COLOR, line_width=1.0)
    pl.set_background("white")
    pl.camera_position = cam
    # Keep the fitted camera: extra zoom clips the outlet corner at the bottom
    # of the screenshot. The image crop below already removes excess whitespace.
    img = pl.screenshot(return_img=True)
    pl.close()

    mask = (img < 250).any(axis=2)  # crop away the white margin around the mesh
    ys, xs = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    pad = 8
    return img[max(0, ys.min() - pad):min(img.shape[0], ys.max() + pad + 1),
               max(0, xs.min() - pad):min(img.shape[1], xs.max() + pad + 1)]


@mticker.FuncFormatter
def sci(v, _):
    """Tick label as $m \times 10^{e}$."""
    e = int(np.floor(np.log10(abs(v))))
    return rf"${v / 10**e:.1f} \times 10^{{{e}}}$"


def add_colorbar(fig, ax, label, vmin, vmax, log=False):
    """Continuous colorbar with top ticks/label, in the snippet's style.

    Ticks mark both ends and the middle of the range; with log=True the scale is
    logarithmic, the middle tick is the geometric mean and labels are m x 10^e.
    """
    if log:
        norm, ticks, fmt = matplotlib.colors.LogNorm(vmin, vmax), np.geomspace(vmin, vmax, 3), sci
    else:
        norm, ticks = matplotlib.colors.Normalize(vmin, vmax), np.linspace(vmin, vmax, 3)
        fmt = mticker.FormatStrFormatter("%.2f")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cax = ax.inset_axes([0.0, -0.28, 1.0, 0.05])  # full width of the (cropped) render
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=ticks)
    cbar.ax.xaxis.set_major_formatter(fmt)
    cbar.ax.xaxis.set_minor_locator(mticker.NullLocator())
    cbar.ax.tick_params(which="both", length=0, labelsize=15)
    cbar.set_label(label, labelpad=6, fontsize=plt.rcParams["font.size"])
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")


def fitted_camera(mesh, pts):
    """Shared camera (Y up, X along the flow) fitted to the given geometry."""
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 800))
    tmp = mesh.copy(); tmp.points = pts
    plotter.add_mesh(tmp)
    centre = np.array(tmp.center)
    plotter.camera_position = [tuple(centre + 20 * np.asarray(VIEW_DIR)), tuple(centre), (0, 1, 0)]
    plotter.reset_camera()
    cam = plotter.camera_position
    plotter.close()
    return cam


def main():
    plt.switch_backend("TkAgg")  # ensure an interactive backend to show the figure
    style()

    params = np.atleast_2d(np.loadtxt(DATA_DIR / "parameters.txt"))
    data = {"velocity": np.loadtxt(DATA_DIR / "snapshots.txt"),
            "c-trace": np.loadtxt(DATA_DIR / "c-trace.txt")}
    mc2 = np.loadtxt(DATA_DIR / "mesh_coor2.txt")
    mesh = pv.read(str(MESH_FILE))
    ref_pts = np.asarray(mesh.points).copy()
    rows = [int(np.argmin(np.hypot(params[:, 0] - L, params[:, 1] - BETA))) for L in LAMBDAS]

    cam = fitted_camera(mesh, swollen_points(ref_pts, mc2[rows[-1]]))  # most swollen geometry

    fig, axes = plt.subplots(len(LAMBDAS), len(FIELDS), figsize=(10.0, 6.5))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.94, bottom=0.08, wspace=0.05, hspace=0.65)
    for r, (lam, row) in enumerate(zip(LAMBDAS, rows)):
        pts = swollen_points(ref_pts, mc2[row])
        for c, field in enumerate(FIELDS):
            ax = axes[r, c]
            vals = nodal_values(field, data[field][row])
            vmin, vmax = float(vals.min()), float(vals.max())
            ax.imshow(render_panel(mesh, pts, vals, (vmin, vmax), cam))
            ax.axis("off")
            ax.set_title(rf"$\lambda={lam:.0f}$", fontsize = 16)
            add_colorbar(fig, ax, TITLES[field], vmin, vmax)

    plt.savefig(Path(__file__).resolve().parent / "figure13.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
