"""Adaptive sampling for the POD-GPR ROM (Section 3.4, Algorithm 3) and its LHS baseline."""

import os
import subprocess

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import qmc

from .pod import POD
from .offline_phase import OfflinePhase
from .online_phase import OnlinePhase


class AdaptiveSampling:

    def __init__(self, fom, solver, field="velocity", acq="var", n_candidates=10_000, n_ref=1_000,
                 seed=None, centering=True, eps=1e-6, x_scaler="minmax", alc_nugget=1e-6, plots=False,
                 plot_modes=(1, 3, 5), param_labels=(r"$\lambda$", r"$\beta$")):
        if acq not in ("var", "alc", "lhs"):
            raise ValueError('acq must be "var", "alc" or "lhs"')
        self.fom, self.solver, self.field, self.acq = fom, solver, field, acq
        # only the prescribed field is trained, predicted and stored
        fom.train, fom.test, fom.fields = {field: fom.train[field]}, {field: fom.test[field]}, (field,)
        self.n_candidates, self.n_ref, self.seed = n_candidates, n_ref, fom.seed if seed is None else seed
        self.centering, self.eps, self.x_scaler, self.alc_nugget = centering, eps, x_scaler, alc_nugget
        self.plots, self.plot_modes, self.param_labels = plots, tuple(plot_modes), param_labels
        P = fom.parameters_test[:, fom.param_cols]
        self.bounds = np.column_stack([P.min(axis=0), P.max(axis=0)])
        self.candidates, self.history = None, []
        self.reference = self.lhs(n_ref, seed=self.seed + 1) if acq == "alc" else None  # R, separate from the pool

    def lhs(self, n, seed=None):
        """n Latin hypercube points over the parameter domain, from a fixed seed."""
        seed = self.seed if seed is None else seed
        return qmc.scale(qmc.LatinHypercube(d=len(self.bounds), seed=seed).random(n), *self.bounds.T)

    def build_rom(self):
        """Algorithm 3, lines 4-6: SVD, projection and GPR training for the chosen field."""
        self.pod = POD(self.fom, self.centering, self.eps)
        self.pod.compute_basis()
        self.offline = OfflinePhase(self.fom, self.pod, self.x_scaler)
        self.offline.train_rom()
        self.online = OnlinePhase(self.fom, self.pod, self.offline)
        self.online.reconstruct_sol()

    def std(self, mu):
        """Posterior std sigma(mu), common to every POD coefficient under the shared kernel."""
        X = np.atleast_2d(mu)
        if self.offline.scaler is not None:
            X = self.offline.scaler.transform(X)
        _, sd = self.offline.models[self.field].predict(X, return_std=True)
        return sd.reshape(len(X), -1)[:, 0]

    def variance_reduction(self, C):
        """Delta(c): mean drop of the posterior variance over the reference set R if c were added."""
        gp, scale = self.offline.models[self.field], self.offline.scaler
        C, R = (C, self.reference) if scale is None else (scale.transform(C), scale.transform(self.reference))
        k, X = gp.kernel_, gp.X_train_
        VR = solve_triangular(gp.L_, k(X, R), lower=True)
        out = np.empty(len(C))
        for s in range(0, len(C), 2_000):  # chunks keep the R x C covariance small
            Cs = C[s:s + 2_000]
            VC = solve_triangular(gp.L_, k(X, Cs), lower=True)
            cov = k(R, Cs) - VR.T @ VC                     # posterior cov(x, c)
            var_c = k.diag(Cs) - (VC ** 2).sum(axis=0)     # posterior sigma^2(c)
            out[s:s + 2_000] = (cov ** 2).mean(axis=0) / (var_c + gp.alpha + self.alc_nugget * k.diag(Cs))
        return out

    def select(self):
        """Next point, removed from the pool: arg max of sd or Delta, or the next LHS point."""
        if self.acq == "var":
            j = int(np.argmax(self.std(self.candidates)))
        elif self.acq == "alc":
            j = int(np.argmax(self.variance_reduction(self.candidates)))
        else:
            j = 0
        mu, self.candidates = self.candidates[j], np.delete(self.candidates, j, axis=0)
        return mu

    def gp_predict(self, X, n_samples=0, mode=0):
        """GP mean and std of POD coefficient a_{mode+1} at X, plus n_samples posterior realizations."""
        gp, scale = self.offline.models[self.field], self.offline.scaler
        Xs = X if scale is None else scale.transform(X)
        mean, sd = gp.predict(Xs, return_std=True)
        mean, sd = mean.reshape(len(X), -1)[:, mode], sd.reshape(len(X), -1)[:, mode]
        if not n_samples:
            return mean, sd
        Y = gp.sample_y(Xs, n_samples, random_state=self.seed).reshape(len(X), -1, n_samples)[:, mode, :]
        return mean, sd, Y

    def plot_iteration(self, mu_next=None, n_grid=60, n_samples=3):
        """GP of the POD coefficients in plot_modes at the current iteration, one row per mode.

        1D parameter: (left) mean of a_k with the +-2 sigma band and the training data,
        (middle) the band and posterior realizations relative to the mean, at true scale,
        (right) the variance sigma^2.
        2D parameter: 3D surfaces over the box, (left) the mean of a_k with translucent
        mean +- 2 sigma surfaces and the training data, (right) the variance sigma^2.
        Modes beyond the current number of POD modes r are skipped.
        """
        import matplotlib.pyplot as plt
        lo, hi = self.bounds.T
        P = self.fom.parameters_train[:, self.fom.param_cols]
        coeffs = self.pod.data[self.field]["coeffs_train"]
        modes = [k for k in self.plot_modes if k <= coeffs.shape[1]]
        labels = list(self.param_labels) + [rf"$\mu_{{{i + 1}}}$" for i in range(len(self.param_labels), len(lo))]
        style = {"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13, "xtick.labelsize": 11,
                 "ytick.labelsize": 11, "legend.fontsize": 10, "text.usetex": False}

        with plt.rc_context(style):
            if len(lo) == 1:
                fig, axes = plt.subplots(len(modes), 3, figsize=(20, 4.6 * len(modes)), squeeze=False)
                x = np.linspace(lo[0], hi[0], 400)
                for row, k in zip(axes, modes):
                    a = rf"$a_{{{k}}}$"
                    mean, sd, Y = self.gp_predict(x[:, None], n_samples, mode=k - 1)
                    ak = coeffs[:, k - 1]

                    # left: GP mean with the +-2 sigma band and the training data
                    ax = row[0]
                    ax.fill_between(x, mean - 2 * sd, mean + 2 * sd, color="C0", alpha=0.25, label=r"$\pm 2\sigma$")
                    ax.plot(x, Y, color="C0", lw=0.8, alpha=0.7)
                    ax.plot(x, mean, color="C0", lw=2, label="GP mean")
                    ax.plot(P[:, 0], ak, "ko", ms=5, label="training")
                    ax.set(xlabel=labels[0], ylabel=a, title=rf"GP mean of {a} with $\pm 2\sigma$ band")

                    # middle: the same band and realizations relative to the mean, at their true size
                    ax = row[1]
                    ax.fill_between(x, -2 * sd, 2 * sd, color="C0", alpha=0.25, label=r"$\pm 2\sigma$")
                    ax.plot(x, Y - mean[:, None], color="C0", lw=0.9, alpha=0.8)
                    ax.plot([], [], color="C0", lw=0.9, label=f"{n_samples} realizations")
                    ax.axhline(0, color="C0", lw=2, label="GP mean")
                    ax.plot(P[:, 0], ak - self.gp_predict(P, mode=k - 1)[0], "ko", ms=5, label="training")
                    ax.set(xlabel=labels[0], ylabel=f"{a} $-$ GP mean", title="uncertainty band around the GP mean")

                    # right: the variance sigma^2(mu); alc minimizes the area under it
                    ax = row[2]
                    ax.plot(x, sd ** 2, color="C2", lw=2)
                    ax.plot(P[:, 0], np.zeros(len(P)), "k+", ms=10, mew=1.5, label="training")
                    if mu_next is not None:
                        s_next = self.gp_predict(np.atleast_2d(mu_next), mode=k - 1)[1][0]
                        ax.plot(mu_next[0], s_next ** 2, "r*", ms=15, label="next")
                    ax.set(xlabel=labels[0], ylabel=r"$\sigma^2$", title=rf"GP variance $\sigma^2$ of {a}")

                    for ax in row[:2]:
                        if mu_next is not None:
                            ax.axvline(mu_next[0], color="r", ls="--", label="next")
                    for ax in row:
                        ax.legend(loc="best")
            else:
                g0, g1 = np.meshgrid(np.linspace(lo[0], hi[0], n_grid), np.linspace(lo[1], hi[1], n_grid))
                grid = np.column_stack([g0.ravel(), g1.ravel()] +
                                       [np.full(g0.size, (l + h) / 2) for l, h in zip(lo[2:], hi[2:])])
                fig = plt.figure(figsize=(15, 6 * len(modes)))
                for i, k in enumerate(modes):
                    a = rf"$a_{{{k}}}$"
                    mean, sd = (v.reshape(g0.shape) for v in self.gp_predict(grid, mode=k - 1))

                    # left: GP mean surface with the +-2 sigma uncertainty surfaces and the training data
                    ax = fig.add_subplot(len(modes), 2, 2 * i + 1, projection="3d")
                    ax.plot_surface(g0, g1, mean, cmap="coolwarm", alpha=0.85, linewidth=0)
                    for s in (-2, 2):
                        ax.plot_surface(g0, g1, mean + s * sd, color="C0", alpha=0.18, linewidth=0)
                    ax.scatter(P[:, 0], P[:, 1], coeffs[:, k - 1], c="k", s=25, depthshade=False, label="training")
                    if mu_next is not None:
                        m_next, s_next = (v[0] for v in self.gp_predict(np.atleast_2d(mu_next), mode=k - 1))
                        ax.scatter(*mu_next[:2], m_next, c="r", marker="*", s=250, depthshade=False, label="next")
                    ax.set(xlabel=labels[0], ylabel=labels[1], zlabel=a,
                           title=rf"GP mean of {a} with $\pm 2\sigma$ surfaces")
                    ax.legend(loc="upper left")

                    # right: the variance surface sigma^2(mu); alc minimizes the volume under it
                    ax = fig.add_subplot(len(modes), 2, 2 * i + 2, projection="3d")
                    surf = ax.plot_surface(g0, g1, sd ** 2, cmap="viridis", linewidth=0)
                    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
                    ax.scatter(P[:, 0], P[:, 1], np.zeros(len(P)), c="k", marker="+", s=40, depthshade=False,
                               label="training")
                    if mu_next is not None:
                        ax.scatter(*mu_next[:2], s_next ** 2, c="r", marker="*", s=250, depthshade=False, label="next")
                    ax.set(xlabel=labels[0], ylabel=labels[1], zlabel=r"$\sigma^2$",
                           title=rf"GP variance $\sigma^2$ of {a}")
                    ax.legend(loc="upper left")
            fig.suptitle(f"{self.field}, acq = {self.acq}, M = {len(P)}")
            fig.tight_layout()
            plt.show()

    def enrich(self, mu):
        """Algorithm 3, line 9: full-order solve at mu, appended to the training set."""
        params, snaps = self.solver(mu)
        self.fom.train[self.field] = np.vstack([self.fom.train[self.field], snaps[self.field]])
        self.fom.parameters_train = np.vstack([self.fom.parameters_train, params])

    def run(self, max_samples, verbose=True):
        """Enrich until the training set holds max_samples snapshots; return the per-iteration history.

        Each history entry holds the ROM on the current M samples (error, max test std)
        and the point mu chosen next with its predictive std.
        """
        if self.candidates is None:  # the pool is drawn once, never per iteration
            n_new = max(max_samples - len(self.fom.parameters_train), 0)  # points to add to the initial design
            self.candidates = self.lhs(n_new if self.acq == "lhs" else self.n_candidates)
            if verbose and self.acq == "lhs":
                print(f"LHS design ({n_new} points, seed = {self.seed}), added in this order:")
                for i, mu in enumerate(self.candidates, 1):
                    print(f"  {i:3d}: (" + ", ".join(f"{v:.4f}" for v in mu) + ")")
        while True:
            self.build_rom()
            M = len(self.fom.parameters_train)
            h = dict(n_train=M, nmodes=self.pod.data[self.field]["nmodes"],
                     error=self.online.mean_errors[self.field],
                     max_std=self.std(self.fom.parameters_test[:, self.fom.param_cols]).max())
            done = M >= max_samples or len(self.candidates) == 0
            if not done:
                h["mu"] = self.select()
                h["std"] = self.std(h["mu"])[0]
            self.history.append(h)
            if verbose:
                line = f"M = {M:3d} | r = {h['nmodes']:3d} | mean error = {h['error']:.3e} | max std = {h['max_std']:.3e}"
                if not done:
                    mu = ", ".join(f"{v:.4f}" for v in h["mu"])
                    line += f" | next mu = ({mu}) std = {h['std']:.3e}"
                print(line)
            if self.plots:
                self.plot_iteration(h.get("mu"))
            if done:
                return self.history
            self.enrich(h["mu"])


def docker_solver(folder, template_row, param_cols, param_names=("lambda", "betav", "mobility"),
                  outputs=(("velocity", "snapshots_add.txt"), ("c-trace", "c-trace_add.txt"),
                           ("mesh_coor2", "mesh_coor2_add.txt")),
                  fields=None, image="fedora-tfem-gfortran-gmsh:latest", executable="./extrudate_test",
                  workdir="/shared"):
    """Full-order solver for AdaptiveSampling: mu -> (parameter row, {field: snapshot}).

    Writes the Fortran namelist (the last two entries of template_row are tend and
    nsteps), runs the solver container, and reads the snapshot it appended to each
    output file.
    """
    outputs = [(f, name) for f, name in outputs if fields is None or f in fields]  # e.g. fields=("c-trace",)

    def solve(mu):
        row = np.array(template_row, dtype=float)
        row[param_cols] = mu
        tend, nsteps = row[-2], int(row[-1])
        body = "".join(f"  {name} = {value:.10f}\n" for name, value in zip(param_names, row[:-2]))
        with open(os.path.join(folder, "params.txt"), "w") as fh:
            fh.write(f"&comppar\n{body}  deltat = {tend / nsteps:.10f}\n  numtimesteps = {nsteps}\n/\n")
        subprocess.run(["docker", "run", "--rm", "--mount", f"type=bind,source={folder},target={workdir}",
                        "-w", workdir, image, "bash", "-c", f"{executable} < params.txt"],
                       check=True, capture_output=True)
        return row, {f: np.loadtxt(os.path.join(folder, name), ndmin=2)[-1] for f, name in outputs}
    return solve
