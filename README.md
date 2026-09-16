# die-swell-rom

**Non-intrusive reduced-order modeling of viscoelastic extrudate (die) swell.**

`die-swell-rom` builds fast, data-driven surrogates of viscoelastic free-surface
flows. It compresses high-fidelity simulation snapshots with **Proper Orthogonal
Decomposition (POD)** and learns the map from physical parameters to the reduced
coordinates with **Gaussian Process Regression (GPR)**. Given a new set of parameters, 
it predicts the velocity field, the polymer stress (trace of the conformation tensor), 
and the shape of the swollen extrudate in a fraction of a second.

---

Each snapshot carries three coupled fields:

| Field         | Symbol            | Meaning                                              |
| ------------- | ----------------- | ---------------------------------------------------- |
| `velocity`    | $\mathbf{u}$      | Nodal velocity components of the melt                |
| `c-trace`     | $\mathrm{tr}(\mathbf{c})$ | Trace of the conformation tensor (polymer stretch/stress) |
| `mesh_coor2`  | $y$               | Deformed free-surface coordinate — the swell shape   |

The flows are parameterized by quantities such as the relaxation time
$\lambda$, the viscosity ratio $\beta$, the mobility/slip $\alpha$, and (for
unsteady cases) time $t$. Any subset of these can be selected as the *active*
ROM inputs.

---

## Repository layout

```
die-swell-rom/
├── romlab/                  # the ROM package (installable, importable)
│   ├── fom.py               # FOM      — load & hold train/test snapshots + parameters
│   ├── pod.py               # POD      — SVD bases, energy truncation, coefficients
│   ├── offline_phase.py     # OfflinePhase — train one GPR per field (params → coeffs)
│   ├── online_phase.py      # OnlinePhase  — predict, reconstruct, score errors
│   └── plots.py             # Plot     — singular-value decay, error maps, field PDFs
│
├── examples/                # runnable Jupyter notebooks — start here
│   ├── rom_1p.ipynb         # 2D, single parameter (λ)
│   ├── rom_2p.ipynb         # 2D, two/three parameters (λ, β, α)
│   ├── rom_3d.ipynb         # 3D extrudate swell
│   └── rom_unsteady.ipynb   # 2D time-dependent (λ, β, t)
│
└── data/                    # snapshot datasets consumed by the examples
    ├── 2d/lambda/           # 1-parameter sweep
    ├── 2d/lambda-beta/      # 2–3 parameter sweep
    ├── 2d/lambda-beta-time/ # unsteady sweep
    └── 3d/lambda-beta-3d/   # 3D case
```

- **`romlab/`** is the library — every reusable class lives here.
- **`data/`** holds the training and test snapshots each example reads.
- **`examples/`** contains the notebooks that run the simple cases end to end.

### Dataset conventions

Each dataset folder provides plain-text matrices (rows = snapshots, columns =
degrees of freedom) with matching `*_test.txt` files for the held-out set, plus
a `mesh.vtk` for visualization:

| File                                 | Contents                                         |
| ------------------------------------ | ------------------------------------------------ |
| `snapshots.txt` / `snapshots_test.txt` | Velocity field per snapshot                    |
| `c-trace.txt` / `c-trace_test.txt`     | $\mathrm{tr}(\mathbf{c})$ per snapshot         |
| `mesh_coor2.txt` / `mesh_coor2_test.txt` | Deformed free-surface $y$-coordinates        |
| `parameters.txt` / `parameters_test.txt` | Parameter rows, e.g. `[λ, β, α, t, Δt]`      |
| `mesh.vtk`                             | Reference mesh for rendering fields             |

---

## Quickstart

Run any notebook in `examples/`, or reproduce the single-parameter case in a few
lines. The notebooks add the repository root to `sys.path`, so `romlab` imports
without a formal install:

```python
import sys
from pathlib import Path

ROOT = Path.cwd().parent            # repo root, from examples/
sys.path.insert(0, str(ROOT))
from romlab import FOM, POD, OfflinePhase, OnlinePhase, Plot

# File order: [velocity, second field, mesh (optional), parameters]
files_train = ["snapshots.txt",      "c-trace.txt",      "mesh_coor2.txt",      "parameters.txt"]
files_test  = ["snapshots_test.txt", "c-trace_test.txt", "mesh_coor2_test.txt", "parameters_test.txt"]

# 1) Full-order snapshots ------------------------------------------------------
fom = FOM(
    data_folder=ROOT / "data/2d/lambda",
    filenames_train=files_train,
    filenames_test=files_test,
    mesh_file="mesh.vtk",
    model="extrudate_swell",
    param_cols=[0],          # use column 0 (λ) as the active parameter
    seed=42,                 # reproducibility
)
fom.info()

# 2) POD compression -----------------------------------------------------------
pod = POD(fom, centering=False, eps=1e-6)
pod.compute_basis()
pod.info()

# 3) Offline: train one GPR per field -----------------------------------------
offline = OfflinePhase(fom, pod, x_scaler="minmax")
offline.train_rom()
offline.info()

# 4) Online: predict on the test parameters and score --------------------------
online = OnlinePhase(fom, pod, offline)
online.reconstruct_sol()     # returns predictions; pass return_std=True for GPR σ
online.info()                # prints per-field relative error

# 5) Visualize the relative error over parameter space -------------------------
plot = Plot(fom, online, comp_indices=[])
plot.plot_relative_error()
```

---

## The pipeline, class by class

| Stage        | Class          | What it does                                                                                     |
| ------------ | -------------- | ------------------------------------------------------------------------------------------------ |
| Data         | `FOM`          | Loads train/test snapshots, mesh, and parameters; `param_cols` selects the active inputs.        |
| Compression  | `POD`          | Per-field SVD, optional mean-centering, energy truncation at `eps`; exposes bases and coeffs.    |
| Offline      | `OfflinePhase` | Trains one `GaussianProcessRegressor` per field (constant × ARD-RBF), with `minmax`/`standard` input scaling. |
| Online       | `OnlinePhase`  | Predicts coefficients, reconstructs fields, and computes relative errors + predictive std.       |
| Postprocess  | `Plot`         | Singular-value decay, relative-error maps in parameter space, and deformed-mesh field figures.   |

Each class has an `.info()` method that prints a short summary (mode counts,
learned kernels, per-field errors) so you can sanity-check every stage as you go.

---

## Reproducibility

All stochastic steps are seeded so runs are deterministic. The `seed` passed to
`FOM` (default **42**) flows through to the GPR `random_state` used for the
kernel-optimizer restarts in `OfflinePhase`, so training the ROM twice on the
same data yields identical models. The example notebooks all pin `seed=42`.
