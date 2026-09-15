"""
Visualization Module for Reduced-Order Modeling
Generates PDF plots of FOM and ROM solutions.
"""

import numpy as np
import os
import vtk
import shutil
import pyvista as pv
from matplotlib.colors import LogNorm
from typing import List, Optional, Literal
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 35,
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.weight": "light",
    "lines.linewidth": 1.5,
    "figure.figsize": (10, 8),
    "legend.fontsize": 30,
    "axes.labelsize": 40,
    "legend.loc": "best",
    "lines.markersize": 8,
    "legend.frameon": True
})

class Plot:
    """
    Visualization class for FOM and ROM field comparisons.
    
    Generates vector PDF plots of velocity and viscosity fields
    with side-by-side FOM/ROM comparisons and parameter annotations.
    
    Attributes:
        fom: Full-order model object with mesh and data
        online: Online phase object with ROM predictions
        comp_indices: Snapshot indices to visualize (1-based)
    """
    
    def __init__(self, fom, online, comp_indices: Optional[List[int]] = None):
        """
        Initialize plot generator.

        Args:
            fom: Full-order model object
            online: Online phase object with predictions
            comp_indices: Optional list of snapshot indices to plot (1-based).
                Only needed by ``run_plots`` and ``plot_mesh_error``.
        """
        self.fom = fom
        self.online = online
        self.comp_indices = [] if comp_indices is None else list(comp_indices)

    def plot_singular_values(
        self,
        fields=None,
        max_modes=None,
        figsize=(8, 6),
        filename=None,
    ):
        if not self.data:
            raise ValueError("call compute_basis() first")

        labels = {
            "velocity": "velocity",
            "c-trace": r"tr($\mathbf{c}$)",
            "mesh_coor2": "height",
        }

        fields = self.computed_fields if fields is None else fields

        fig, ax = plt.subplots(figsize=figsize)

        for field in fields:
            svals = self.data[field]["svals"]

            n = len(svals) if max_modes is None else min(max_modes, len(svals))

            ax.semilogy(
                np.arange(1, n + 1),
                svals[:n],
                marker="o",
                label=labels.get(field, field),
            )

        ax.set_xlabel("index")
        ax.set_ylabel(r"$\sigma_i/\sigma_1$")
        ax.grid(alpha=0.3)
        ax.legend()

        if filename:
            fig.savefig(filename, bbox_inches="tight")

        plt.show()

    def plot_mesh_error(
        self,
        selected_indices=None,
        fields=("velocity", "viscosity"),
        colormap="coolwarm",
        n_colors=256,
        normalize_error=False,
        vmin_percentile=None,
        clip_percentile=None,
        log_scale=False,
        scalar_bar_n_labels=6,
        export_pdf=True,
        output_dir=None,
    ):
        """
        Plot pointwise relative error on the deformed (swollen) mesh.

        The mesh y-coordinates are updated from ``mesh_coor2_test.txt``
        so that the extrudate shape is visible.  Only the relative-error
        field is rendered (no FOM / ROM panels).

        Parameters
        ----------
        selected_indices : list of int, optional
            1-based snapshot indices to plot.  Defaults to ``comp_indices``.
        fields : tuple of str
            Which fields to plot.  ``"velocity"`` and/or ``"viscosity"``.
        colormap : str
            Colormap name (default ``"coolwarm"``).
        n_colors : int
            Number of colour levels (use 256 for smooth gradients).
        normalize_error : bool
            If True, normalize each snapshot error to [0, 1] using
            ``err_norm = err / max(err)``. If False, use relative error
            ``|FOM - ROM| / (|FOM| + eps)``.
        vmin_percentile : float or None
            Optional lower-percentile floor for the color scale (e.g. 2.0).
            If set, ``vmin`` is taken as this percentile of strictly positive
            errors, improving contrast away from near-zero regions.
        clip_percentile : float or None
            Optional upper-percentile clipping for the color scale (e.g. 99.0).
            If set, ``vmax`` is taken as this percentile of the error field,
            which helps visualize low-to-mid error structure when a sharp
            localized peak dominates the range.
        log_scale : bool
            If True, use logarithmic color mapping (recommended for highly
            localized error fields spanning several orders of magnitude).
        scalar_bar_n_labels : int
            Number of tick labels shown on the scalar bar.
        export_pdf : bool
            If True, export vector PDF (cropped).
        output_dir : str or None
            Output directory.  Default: ``<data_folder>/mesh_error``.
        """
        from pathlib import Path

        if selected_indices is None:
            selected_indices = self.comp_indices

        data_path = Path(self.fom.data_folder)
        mesh_path = data_path / self.fom.mesh_file
        meshcoor2_test = np.loadtxt(data_path / "mesh_coor2_test.txt")

        if output_dir is None:
            output_dir = data_path / "mesh_error"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        mesh0 = pv.read(str(mesh_path))
        n_nodes = mesh0.number_of_points
        eps = 1e-12

        fom_vel  = self.fom.velocity_test
        rom_vel  = np.array(self.online.reduced_vel)
        fom_visc = self.fom.viscosity_test
        rom_visc = np.array(self.online.reduced_visc)

        visc_label = self.fom.second_field_label   # "tr(c)" or "Viscosity"

        for idx in selected_indices:
            if idx < 1 or idx > fom_vel.shape[0]:
                print(f"  Snapshot index {idx} out of range, skipping.")
                continue
            si = idx - 1  # 0-based

            # ---------- deform mesh ----------
            mesh = mesh0.copy()
            mesh.points[:, 1] = meshcoor2_test[si, :]

            for field in fields:
                if field == "velocity":
                    fom_s = self._extract_scalar_field(fom_vel[si], "velocity")
                    rom_s = self._extract_scalar_field(rom_vel[si], "velocity")
                    if normalize_error:
                        bar_title = r"Normalized $\|\mathbf{u}_{\mathrm{FOM}} - \mathbf{u}_{\mathrm{ROM}}\|$"
                    else:
                        bar_title = r"$\|\mathbf{u}_{\mathrm{FOM}} - \mathbf{u}_{\mathrm{ROM}}\|$"
                    fname = f"vel_error_{idx:04d}"
                else:
                    fom_s = fom_visc[si]
                    rom_s = rom_visc[si]
                    if visc_label == "tr(c)":
                        if normalize_error:
                            bar_title = (r"Normalized $\|\mathrm{tr}(\mathbf{c})_{\mathrm{FOM}}"
                                         r" - \mathrm{tr}(\mathbf{c})_{\mathrm{ROM}}\|$")
                        else:
                            bar_title = (r"$\|\mathrm{tr}(\mathbf{c})_{\mathrm{FOM}}"
                                         r" - \mathrm{tr}(\mathbf{c})_{\mathrm{ROM}}\|$")
                    else:
                        if normalize_error:
                            bar_title = r"Normalized $\|\eta_{\mathrm{FOM}} - \eta_{\mathrm{ROM}}\|$"
                        else:
                            bar_title = r"$\|\eta_{\mathrm{FOM}} - \eta_{\mathrm{ROM}}\|$"
                    fname = f"visc_error_{idx:04d}"

                err_abs = np.abs(fom_s - rom_s)
                rel_err = err_abs / (np.abs(fom_s) + eps)
                if normalize_error:
                    rel_max = float(np.nanmax(rel_err))
                    rel_err = rel_err / (rel_max + eps)

                mesh_err = mesh.copy()
                mesh_err.point_data["error"] = rel_err
                positive = rel_err[rel_err > 0]
                vmin = float(np.nanmin(positive)) if positive.size else 1e-16
                if vmin_percentile is not None and positive.size:
                    p_low = float(vmin_percentile)
                    p_low = min(max(p_low, 0.0), 100.0)
                    vmin_p = float(np.nanpercentile(positive, p_low))
                    if np.isfinite(vmin_p) and vmin_p > 0.0:
                        vmin = max(vmin, vmin_p)
                vmax = float(np.nanmax(rel_err))
                if clip_percentile is not None:
                    p = float(clip_percentile)
                    p = min(max(p, 0.0), 100.0)
                    vmax_p = float(np.nanpercentile(rel_err, p))
                    if np.isfinite(vmax_p) and vmax_p > 0.0:
                        vmax = min(vmax, vmax_p)
                if not np.isfinite(vmax) or vmax <= vmin:
                    vmax = max(vmin * 10.0, 1e-15)
                clim = [vmin, vmax]

                # ---- plotter ----
                plotter = pv.Plotter(
                    off_screen=export_pdf,
                    window_size=(1600, 600),
                )
                plotter.add_mesh(
                    mesh_err,
                    scalars="error",
                    cmap=colormap,
                    clim=clim,
                    n_colors=n_colors,
                    log_scale=log_scale,
                    show_edges=False,
                    show_scalar_bar=False,
                )

                # Compute scalar bar position from mesh bounds so it
                # aligns with the visible mesh width.
                plotter.view_xy()
                plotter.reset_camera()
                plotter.camera.zoom(1.3)
                plotter.render()

                xmin, xmax, ymin, ymax, _, _ = mesh_err.bounds
                renderer = plotter.renderer
                coord = vtk.vtkCoordinate()
                coord.SetCoordinateSystemToWorld()

                coord.SetValue(xmin, ymin, 0)
                disp_left = coord.GetComputedDisplayValue(renderer)
                coord.SetValue(xmax, ymin, 0)
                disp_right = coord.GetComputedDisplayValue(renderer)

                w, h = plotter.window_size
                bar_x = disp_left[0] / w
                bar_w = (disp_right[0] - disp_left[0]) / w
                # Clamp to safe range
                bar_x = max(0.02, min(bar_x, 0.4))
                bar_w = max(0.2, min(bar_w, 0.96 - bar_x))

                scalar_bar = plotter.add_scalar_bar(
                    title=bar_title,
                    title_font_size=28,
                    label_font_size=22,
                    position_x=bar_x,
                    position_y=0.2,
                    width=bar_w,
                    height=0.06,
                    vertical=False,
                    n_labels=max(2, int(scalar_bar_n_labels)),
                    fmt="%.1e" if (log_scale or vmax < 1e-2) else "%.3f",
                )
                scalar_bar.GetTitleTextProperty().SetFontFamilyToArial()
                scalar_bar.GetTitleTextProperty().SetBold(True)
                scalar_bar.GetLabelTextProperty().SetFontFamilyToArial()

                # ---- save cropped PNG ----
                png_path = output_path / f"{fname}.png"
                plotter.screenshot(str(png_path))
                self._crop_png(png_path, margin=10)

                # ---- export PDF ----
                if export_pdf:
                    pdf_path = output_path / f"{fname}.pdf"
                    render_window = plotter.ren_win
                    pdf_exporter = vtk.vtkGL2PSExporter()
                    pdf_exporter.SetRenderWindow(render_window)
                    pdf_exporter.SetFileFormatToPDF()
                    pdf_exporter.SetFilePrefix(str(pdf_path).replace(".pdf", ""))
                    pdf_exporter.CompressOff()
                    pdf_exporter.SetSortToBSP()
                    pdf_exporter.SetTextAsPath(False)
                    pdf_exporter.SetPS3Shading(True)
                    pdf_exporter.SetBestRoot(True)
                    pdf_exporter.DrawBackgroundOff()
                    render_window.SetMultiSamples(0)
                    plotter.render()
                    pdf_exporter.Write()
                    # PDF crop may fail on VTK-exported PDFs with transparency;
                    # fall back silently and rely on the cropped PNG.
                    try:
                        self.crop_pdf(pdf_path, pdf_path, margin=5, top_margin=0)
                    except Exception:
                        pass

                plotter.close()
                # Display the cropped PNG in the notebook
                from IPython.display import display, Image as IPImage
                display(IPImage(filename=str(png_path)))

    def plot_relative_error(
            self,
            param_labels=(r"$\lambda$", r"$\beta$"),
            fields=("velocity", "c-trace"),
            titles=(r"$\varepsilon_u$", r"$\varepsilon_c$"),
            vmin=1e-6,
            vmax=1e-2,
            n_levels=8,
            colormap="coolwarm",
            panel_size=(8, 6),
            filename=None,
            show=True,
        ):
            """Relative error of each test sample, one panel per field.

            1 active parameter:  error against the parameter on a log axis,
                                training samples as ``+`` markers along the bottom.
            2 active parameters: test samples coloured by error on a shared discrete
                                log colour scale (vmin..vmax, n_levels bins),
                                training samples as ``+`` markers.
            """
            from matplotlib.colors import BoundaryNorm

            cols = list(self.fom.param_cols)

            X_train = self.fom.parameters_train[:, cols]
            X_test = self.fom.parameters_test[:, cols]
            errors = self._relative_errors(fields)
            n = len(fields)
            pw, ph = panel_size
            train_style = dict(color="black", marker="+", markersize=14,
                            markeredgewidth=2.0, linestyle="none", zorder=1)

            if len(cols) == 1:
                fig, axes = plt.subplots(1, n, figsize=(n * pw, ph), sharey=True,
                                        squeeze=False)
                for k, (ax, field) in enumerate(zip(axes[0], fields)):
                    ax.semilogy(X_test[:, 0], errors[field], "o", markersize=10, zorder=2)
                    # training samples: x in data units, y fixed near the bottom
                    ax.plot(X_train[:, 0], np.full(len(X_train), 0.03),
                            transform=ax.get_xaxis_transform(), **train_style)
                    ax.set_xlabel(param_labels[0])
                    ax.set_title(titles[k] if k < len(titles) else field)
                    ax.grid(True, which="major", linewidth=0.5, alpha=0.3)
                axes[0, 0].set_ylabel("relative error")

            else:
                levels = np.logspace(np.log10(vmin), np.log10(vmax), n_levels + 1)
                cmap = plt.get_cmap(colormap, n_levels)
                norm = BoundaryNorm(levels, ncolors=n_levels)

                # absolute margins in inches so every panel is exactly panel_size
                m_left, m_right, m_top, m_bottom, gap = 2.0, 0.5, 1.0, 2.8, 1.0
                cbar_h, cbar_y = 0.4, 1.1
                fig_w = m_left + n * pw + (n - 1) * gap + m_right
                fig_h = m_bottom + ph + m_top
                fig = plt.figure(figsize=(fig_w, fig_h))

                for k, field in enumerate(fields):
                    x0 = (m_left + k * (pw + gap)) / fig_w
                    ax = fig.add_axes([x0, m_bottom / fig_h, pw / fig_w, ph / fig_h])
                    err = np.clip(np.asarray(errors[field], dtype=float), vmin, vmax)

                    ax.plot(X_train[:, 0], X_train[:, 1], **train_style)
                    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=err, cmap=cmap,
                                        norm=norm, s=180, edgecolors="none", zorder=2)
                    ax.set_xlabel(param_labels[0])
                    if k == 0:
                        ax.set_ylabel(param_labels[1])
                    else:
                        ax.tick_params(labelleft=False)
                    ax.set_title(titles[k] if k < len(titles) else field)
                    ax.grid(True, which="major", linewidth=0.5, alpha=0.3)

                # colorbar spanning the full width of the panel block
                cax = fig.add_axes([m_left / fig_w, cbar_y / fig_h,
                                    (n * pw + (n - 1) * gap) / fig_w, cbar_h / fig_h])
                cbar = fig.colorbar(
                    scatter, cax=cax, orientation="horizontal", boundaries=levels,
                    ticks=np.logspace(np.log10(vmin), np.log10(vmax),
                                    int(np.log10(vmax / vmin)) + 1))
                cbar.ax.set_xscale("log")
                cbar.ax.tick_params(labelsize=plt.rcParams["font.size"])

            if filename:
                fig.savefig(filename, bbox_inches="tight")
            if show:
                plt.show()
                plt.close(fig)
                return None
            return fig

    def _relative_errors(self, fields):
        """Per-test-sample relative error for each requested field."""
        short = {"velocity": "vel", "c-trace": "visc", "viscosity": "visc",
                 "mesh_coor2": "mesh"}
        out = {}
        for field in fields:
            err = None
            stored = getattr(self.online, "errors", None)
            if isinstance(stored, dict):
                err = stored.get(field)
            if err is None:
                err = getattr(self.online, f"relative_error_{short.get(field, field)}",
                              None)
            if err is None:
                raise ValueError(
                    f"no relative error available for '{field}'; run "
                    f"online.predict_non_intrusive(field={field!r}) first")
            out[field] = np.asarray(err, dtype=float)
        return out

    def _save_comparison(
        self,
        fom_data: np.ndarray,
        rom_data: np.ndarray,
        output_folder: str,
        field: Literal["velocity", "viscosity"],
        title_fom: str,
        title_rom: str,
        colormap: str = "coolwarm",
        lognorm: bool = False
    ):
        """
        Generate side-by-side FOM / ROM / Relative Error comparison plots.
        """

        # Load mesh
        mesh = pv.read(os.path.join(self.fom.data_folder, self.fom.mesh_file))
        eps = 1e-12   # to avoid division by zero

        # Loop over selected snapshots
        for idx in self.comp_indices:
            if idx < 1 or idx > fom_data.shape[0]:
                print(f"  ⚠ Snapshot index {idx} out of range. Skipping.")
                continue

            snapshot_index = idx - 1

            # Extract parameters for this test case
            params = self.fom.parameters_test[snapshot_index]
            param_str = self._format_parameters(params)

            # ============================================================
            # Compute Scalars: FOM, ROM, Relative Error
            # ============================================================
            fom_scalar = self._extract_scalar_field(fom_data[snapshot_index], field)
            rom_scalar = self._extract_scalar_field(rom_data[snapshot_index], field)

            # ------------------------------------------------------------
            # Relative error field:
            #   e(x) = |fom - rom| / (|fom| + eps)
            # ------------------------------------------------------------
            rel_err = np.abs(fom_scalar - rom_scalar) / (np.abs(fom_scalar) + eps)

            # FOM/ROM share same clim; error has separate scale
            clim_fom_rom = [
                min(fom_scalar.min(), rom_scalar.min()),
                max(fom_scalar.max(), rom_scalar.max())
            ]
            clim_error = [rel_err.min(), rel_err.max()]

            # ============================================================
            # Create Plotter: 3 panels (FOM, ROM, Error)
            # ============================================================
            plotter = pv.Plotter(
                shape=(1, 3),
                off_screen=True,
                window_size=[2400, 600]
            )

            # ============================================================
            # Panel 1: FOM
            # ============================================================
            plotter.subplot(0, 0)
            mesh_fom = mesh.copy()
            mesh_fom.point_data[field] = fom_scalar

            plotter.add_mesh(
                mesh_fom,
                scalars=field,
                cmap=colormap,
                clim=clim_fom_rom,
                show_edges=False,
                show_scalar_bar=False,
                n_colors=8
            )
            self._add_scalar_bar(plotter, title_fom, lognorm, clim_fom_rom)

            plotter.add_text("FOM", position="upper_left", font_size=14, color="black")
            plotter.add_text(param_str, position=(0.55, 0.5), font_size=10, color="black")
            plotter.view_xy()

            # ============================================================
            # Panel 2: ROM
            # ============================================================
            plotter.subplot(0, 1)
            mesh_rom = mesh.copy()
            mesh_rom.point_data[field] = rom_scalar

            plotter.add_mesh(
                mesh_rom,
                scalars=field,
                cmap=colormap,
                clim=clim_fom_rom,
                show_edges=False,
                show_scalar_bar=False,
                n_colors=8
            )
            self._add_scalar_bar(plotter, title_rom, lognorm, clim_fom_rom)

            plotter.add_text("ROM", position="upper_left", font_size=14, color="black")
            plotter.add_text(param_str, position=(0.55, 0.5), font_size=10, color="black")
            plotter.view_xy()

            # ============================================================
            # Panel 3: Relative Error
            # ============================================================
            plotter.subplot(0, 2)
            mesh_err = mesh.copy()
            mesh_err.point_data["relative_error"] = rel_err

            plotter.add_mesh(
                mesh_err,
                scalars="relative_error",
                cmap="coolwarm",
                clim=clim_error,
                show_edges=False,
                show_scalar_bar=False,
                n_colors=8
            )

            self._add_scalar_bar(
                plotter,
                "Relative error (pointwise)",
                True,
                clim_error
            )

            plotter.add_text("Pointwise error", position="upper_left",
                            font_size=14, color="black")
            plotter.add_text(param_str, position=(0.55, 0.5), font_size=10, color="black")
            plotter.view_xy()

            # ============================================================
            # Render and Save
            # ============================================================
            os.makedirs(output_folder, exist_ok=True)

            save_path = os.path.join(output_folder, f"comparison_{idx}.png")
            plotter.screenshot(save_path)
            
            plotter.show(jupyter_backend="static")            
    
    def _extract_scalar_field(
        self,
        snapshot: np.ndarray,
        field: str
    ) -> np.ndarray:
        """
        Extract scalar field from snapshot data.
        
        Args:
            snapshot: Snapshot vector
            field: Field type ('velocity' or 'viscosity')
        
        Returns:
            Scalar field array for visualization
        """
        if field == "velocity":
            vx = snapshot[::2]
            vy = snapshot[1::2]
            return np.sqrt(vx**2 + vy**2)
        else:
            return snapshot
    
    def _format_parameters(self, params: np.ndarray) -> str:
        """
        Format parameter vector as readable string.
        
        Args:
            params: Parameter vector
        
        Returns:
            Formatted string for display
        """
        if self.fom.model == "extrudate_swell":
            # params: [lambda, betav, mobility, time, timestep]
            lam, betav, mobility = params[0], params[1], params[2]
            return f"λ={lam:.3f}, β={betav:.3f}"
        
        elif self.fom.model == "lid_cavity":
            _, _, n, lambda1 = params[:4]
            return (
                f"n={n:.3f}, λ={lambda1:.3f}"
            )
        
        elif self.fom.model == "axi":
            _, _, lambda1, n, force = params[:5]
            return (f"n={n:.3f}, F={force:.3f}")
        
        else:
            # Generic format
            return ", ".join([f"p{i}={p:.3f}" for i, p in enumerate(params)])
    
    def _add_scalar_bar(
        self,
        plotter: pv.Plotter,
        title: str,
        lognorm: bool,
        clim: List[float]
    ):
        """
        Add scalar bar to plotter with consistent formatting.
        
        Args:
            plotter: PyVista plotter object
            title: Colorbar title
            lognorm: Whether to use logarithmic scale
            clim: Color limits [min, max]
        """
        # Position based on model type
        if self.fom.model in ("lid_cavity", "extrudate_swell"):
            position_x, position_y, width = 0.3, 0.05, 0.4
        else:
            position_x, position_y, width = 0.228, 0.20, 0.55
        
        scalar_bar = plotter.add_scalar_bar(
            title=title,
            title_font_size=20,
            label_font_size=20,
            position_x=position_x,
            position_y=position_y,
            width=width,
            height=0.04,
            vertical=False,
            fmt="%.1f" if not lognorm else "%.0e",
            n_labels=5
        )
        
        # Set font to Arial
        scalar_bar.GetTitleTextProperty().SetFontFamilyToArial()
        scalar_bar.GetLabelTextProperty().SetFontFamilyToArial()

    def plot_meshcoor2(self, data_dir,
                    hhat_file="rom_meshcoor2.txt",
                    field_array=None,
                    mesh_file= "mesh.vtk",
                    output_dir="rom_velocity_pdf",
                    selected_indices=[1, 3, 5],
                    export_pdf=True,
                    save_vtk=True,        
                    logscale=True,
                    n_colors=5,
                    field="velocity",           # "velocity" or "b-trace"
                    field_type="rom"):          # "fom", "rom", or "diff"
        """
        Plot mesh coordinates with scalar/vector fields and optionally save .VTK files.

        Parameters
        ----------
        field : str
            "velocity" → plots |u|
            "b-trace"  → plots trace(b)
        field_type : str
            One of {"fom", "rom", "diff"} → used for titles/filenames
        save_vtk : bool
            If True, saves .vtk files alongside PDF plots.
        """

        import os
        import numpy as np
        import pyvista as pv
        import shutil
        from pathlib import Path
        import vtk

        # --- Prepare output directory
        output_path = Path(output_dir)
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # --- Load mesh coordinates
        data_path = Path(data_dir)
        meshcoor2 = np.loadtxt(data_path / hhat_file)

        if field_array is None:
            raise ValueError("❌ You must provide field_array (shape: n_snapshots × n_values_per_snapshot).")
        field_data = np.asarray(field_array)


        mesh0 = pv.read(os.path.join(data_path, mesh_file))
        n_nodes = mesh0.number_of_points

        # --- Label setup
        if field.lower() == "velocity":
            if field_type == "fom":
                base_name, plot_title = "u_FOM", r"$|\mathbf{u}_{\mathrm{FOM}}|$"
            elif field_type == "rom":
                base_name, plot_title = "u_ROM", r"$|\mathbf{u}_{\mathrm{ROM}}|$"
            elif field_type == "diff":
                base_name, plot_title = "u_diff", r"$|\mathbf{u}_{\mathrm{FOM}} - \mathbf{u}_{\mathrm{ROM}}|$"
            elif field_type == "full":
                base_name, plot_title = "u", r"$|\mathbf{u}|$"
            else: 
                ValueError(f"Unknown field_type: {field_type}")
        elif field.lower() in ["b-trace", "b_trace"]:
            if field_type == "fom":
                base_name, plot_title = "btrace_FOM", r"$|\mathrm{tr}(\mathbf{b})_{\mathrm{FOM}}|$"
            elif field_type == "rom":
                base_name, plot_title = "btrace_ROM", r"$|\mathrm{tr}(\mathbf{b})_{\mathrm{ROM}}|$"
            elif field_type == "diff":
                base_name, plot_title = "btrace_diff", r"$|\mathrm{tr}(\mathbf{b})_{\mathrm{FOM}} - \mathrm{tr}(\mathbf{b})_{\mathrm{ROM}}|$"
            elif field_type == "full":
                base_name, plot_title = "btrace", r"$|\mathrm{tr}(\mathbf{b})|$"
            else:
                ValueError(f"Unknown field_type: {field_type}")
                base_name, plot_title = "btrace", r"$|\mathrm{tr}(b)|$"
        elif field.lower() in ["c-trace", "c_trace"]:
            if field_type == "fom":
                base_name, plot_title = "ctrace_FOM", r"$|\mathrm{tr}(\mathbf{c})_{\mathrm{FOM}}|$"
            elif field_type == "rom":
                base_name, plot_title = "ctrace_ROM", r"$|\mathrm{tr}(\mathbf{c})_{\mathrm{ROM}}|$"
            elif field_type == "diff":
                base_name, plot_title = "ctrace_diff", r"$|\mathrm{tr}(\mathbf{c})_{\mathrm{FOM}} - \mathrm{tr}(\mathbf{c})_{\mathrm{ROM}}|$"
            elif field_type == "full":
                base_name, plot_title = "ctrace", r"$|\mathrm{tr}(\mathbf{c})|$"
            else:
                ValueError(f"Unknown field_type: {field_type}")
        elif field.lower() in ["pressure"]:
            if field_type == "fom":
                base_name, plot_title = "pressure_FOM", r"$|p_{\mathrm{FOM}}|$"
            elif field_type == "rom":
                base_name, plot_title = "pressure_ROM", r"$|p_{\mathrm{ROM}}|$"
            elif field_type == "diff":
                base_name, plot_title = "pressure_diff", r"$|p_{\mathrm{FOM}} - p_{\mathrm{ROM}}|$"
            elif field_type == "full":
                base_name, plot_title = "pressure", r"$|p|$"
            else:
                ValueError(f"Unknown field_type: {field_type}")
        else:
            base_name, plot_title = field, field

        # --- Loop over selected indices
        for row_idx in selected_indices:
            mesh = pv.read(data_path / mesh_file)
            mesh.points[:, 1] = meshcoor2[row_idx, :]

            # --- Auto detect vector vs scalar field
            if field_data.shape[1] == 2 * n_nodes:
                vx = field_data[row_idx, 0::2]
                vy = field_data[row_idx, 1::2]
                scalar_field = np.sqrt(vx**2 + vy**2)
            elif field_data.shape[1] == n_nodes:
                scalar_field = np.abs(field_data[row_idx, :])
            else:
                raise ValueError(
                    f"Cannot determine field type: got {field_data.shape[1]} columns, "
                    f"expected {n_nodes} or {2*n_nodes}."
                )

            # --- Assign scalar field
            mesh.point_data[base_name] = scalar_field

            # --- ✅ Save VTK if requested
            if save_vtk:
                vtk_path = output_path / f"{base_name}_{row_idx:04d}.vtk"
                mesh.save(vtk_path)

                    # --- Plot
            plotter = pv.Plotter(off_screen=export_pdf, window_size=(1200, 800))

            # Compute scalar range
            vmin = np.nanmin(scalar_field[scalar_field > 0]) if logscale else scalar_field.min()
            vmax = scalar_field.max()

            # Optional: adjust color range to powers of 10 for cleaner tick spacing
            if logscale:
                clim = [10 ** np.floor(np.log10(vmin)), 10 ** np.ceil(np.log10(vmax))]
            else:
                clim = [vmin, vmax]

            # --- Add mesh (only once!)
            actor = plotter.add_mesh(
                mesh,
                scalars=base_name,
                cmap="coolwarm",
                clim=clim,
                n_colors=n_colors,
                show_edges=False,
                show_scalar_bar=False,
            )

            # --- Add scalar bar *after* configuring the actor
            scalar_bar = plotter.add_scalar_bar(
                title=plot_title,
                title_font_size=40,
                label_font_size=40,
                position_x=0.15,
                position_y=0.2,
                width=0.7,
                height=0.04,
                vertical=False,
                n_labels=n_colors,
                fmt="%.1e" if logscale else "%.2f",
            )

            # --- Styling
            scalar_bar.GetTitleTextProperty().SetFontFamilyToArial()
            scalar_bar.GetTitleTextProperty().SetBold(False)
            scalar_bar.GetLabelTextProperty().SetFontFamilyToArial()

            plotter.view_xy()
            plotter.camera.zoom(1.1)

            # --- ✅ Export as PDF
            if export_pdf:
                pdf_filename = output_path / f"{base_name}_{row_idx:04d}.pdf"
                render_window = plotter.ren_win

                # Create the PDF exporter
                pdf_exporter = vtk.vtkGL2PSExporter()
                pdf_exporter.SetRenderWindow(render_window)
                pdf_exporter.SetFileFormatToPDF()

                # File name prefix (without ".pdf")
                pdf_exporter.SetFilePrefix(str(pdf_filename).replace(".pdf", ""))

                # --- Rendering options ---
                pdf_exporter.CompressOff()             # keep file uncompressed (better compatibility)
                pdf_exporter.SetSortToBSP()            # correct z-order for transparency
                pdf_exporter.SetTextAsPath(False)      # keep text selectable
                pdf_exporter.SetPS3Shading(True)       # smoother shading
                pdf_exporter.SetBestRoot(True)
                pdf_exporter.DrawBackgroundOff()       # transparent background
                render_window.SetMultiSamples(0)       # disable anti-aliasing (vector output anyway)

                # --- Optional: add small delay to ensure render update ---
                plotter.render()

                # --- Write the PDF ---
                pdf_exporter.Write()

                self.crop_pdf(pdf_filename, pdf_filename, margin=5, top_margin=0, debug=False)

    def plot_meshcoor3d(self,
        data_dir,
        mesh_file="mesh.vtk",
        mesh_y_file="mesh_coor2.txt",
        mesh_z_file="mesh_coor3.txt",
        field_array=None,
        output_dir="rom_field_pdf",
        selected_indices=[1, 3, 5],
        export_pdf=True,
        save_vtk=True,
        logscale=False,
        n_colors=5,
        field="velocity",
        field_type="rom"
    ):
        """
        Plot 3D deformed meshes with scalar/vector fields and optionally save .VTK files.

        This version updates both Y and Z coordinates from mesh_coor2.txt and mesh_coor3.txt.

        Parameters
        ----------
        mesh_y_file : str
            File containing Y displacements (shape: n_snapshots × n_nodes)
        mesh_z_file : str
            File containing Z displacements (shape: n_snapshots × n_nodes)
        field_array : np.ndarray
            Field data for each snapshot (shape: n_snapshots × n_field_values)
        """

        import os
        import numpy as np
        import pyvista as pv
        import shutil
        from pathlib import Path
        import vtk

        # --- Prepare output directory
        output_path = Path(output_dir)
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        # --- Label setup
        if field.lower() == "velocity":
            if field_type == "fom":
                base_name, plot_title = "u_FOM", r"$|\mathbf{u}_{\mathrm{FOM}}|$"
            elif field_type == "rom":
                base_name, plot_title = "u_ROM", r"$|\mathbf{u}_{\mathrm{ROM}}|$"
            elif field_type == "diff":
                base_name, plot_title = "u_diff", r"$|\mathbf{u}_{\mathrm{FOM}} - \mathbf{u}_{\mathrm{ROM}}|$"
            elif field_type == "full":
                base_name, plot_title = "u", r"$|\mathbf{u}|$"
            else: 
                ValueError(f"Unknown field_type: {field_type}")
        elif field.lower() in ["b-trace", "b_trace"]:
            if field_type == "fom":
                base_name, plot_title = "btrace_FOM", r"$|\mathrm{tr}(\mathbf{b})_{\mathrm{FOM}}|$"
            elif field_type == "rom":
                base_name, plot_title = "btrace_ROM", r"$|\mathrm{tr}(\mathbf{b})_{\mathrm{ROM}}|$"
            elif field_type == "diff":
                base_name, plot_title = "btrace_diff", r"$|\mathrm{tr}(\mathbf{b})_{\mathrm{FOM}} - \mathrm{tr}(\mathbf{b})_{\mathrm{ROM}}|$"
            elif field_type == "full":
                base_name, plot_title = "btrace", r"$|\mathrm{tr}(\mathbf{b})|$"
            else:
                ValueError(f"Unknown field_type: {field_type}")
                base_name, plot_title = "btrace", r"$|\mathrm{tr}(b)|$"
        elif field.lower() in ["c-trace", "c_trace"]:
            if field_type == "fom":
                base_name, plot_title = "ctrace_FOM", r"$|\mathrm{tr}(\mathbf{c})_{\mathrm{FOM}}|$"
            elif field_type == "rom":
                base_name, plot_title = "ctrace_ROM", r"$|\mathrm{tr}(\mathbf{c})_{\mathrm{ROM}}|$"
            elif field_type == "diff":
                base_name, plot_title = "ctrace_diff", r"$|\mathrm{tr}(\mathbf{c})_{\mathrm{FOM}} - \mathrm{tr}(\mathbf{c})_{\mathrm{ROM}}|$"
            elif field_type == "full":
                base_name, plot_title = "ctrace", r"$|\mathrm{tr}(\mathbf{c})|$"
            else:
                ValueError(f"Unknown field_type: {field_type}")
        elif field.lower() in ["pressure"]:
            if field_type == "fom":
                base_name, plot_title = "pressure_FOM", r"$|p_{\mathrm{FOM}}|$"
            elif field_type == "rom":
                base_name, plot_title = "pressure_ROM", r"$|p_{\mathrm{ROM}}|$"
            elif field_type == "diff":
                base_name, plot_title = "pressure_diff", r"$|p_{\mathrm{FOM}} - p_{\mathrm{ROM}}|$"
            elif field_type == "full":
                base_name, plot_title = "pressure", r"$|p|$"
            else:
                ValueError(f"Unknown field_type: {field_type}")
        else:
            base_name, plot_title = field, field

        data_path = Path(data_dir)

        # --- Load coordinate deformation data
        mesh_y = np.loadtxt(data_path / mesh_y_file)
        mesh_z = np.loadtxt(data_path / mesh_z_file)
        mesh0 = pv.read(data_path / mesh_file)
        n_nodes = mesh0.number_of_points

        # Validate shape
        if mesh_y.shape[1] != n_nodes or mesh_z.shape[1] != n_nodes:
            raise ValueError(
                f"❌ Shape mismatch: mesh_coor2={mesh_y.shape}, mesh_coor3={mesh_z.shape}, expected (n_snapshots, {n_nodes})"
            )

        # --- Load field data
        if field_array is None:
            raise ValueError("❌ You must provide field_array (n_snapshots × n_values_per_snapshot).")

        field_data = np.asarray(field_array)

        # --- Loop over selected snapshots
        for row_idx in selected_indices:
            mesh = mesh0.copy()

            # === Update Y, Z coordinates ===
            # Keep X from the base mesh
            mesh.points[:, 1] = mesh_y[row_idx, :]
            mesh.points[:, 2] = mesh_z[row_idx, :]

            # === Detect field type ===
            if field_data.shape[1] == 3 * n_nodes:
                # Vector field (vx, vy, vz)
                vx = field_data[row_idx, 0::3]
                vy = field_data[row_idx, 1::3]
                vz = field_data[row_idx, 2::3]
                scalar_field = np.sqrt(vx**2 + vy**2 + vz**2)
            elif field_data.shape[1] == n_nodes:
                # Scalar field
                scalar_field = np.abs(field_data[row_idx, :])
            else:
                raise ValueError(
                    f"Cannot determine field type: got {field_data.shape[1]} columns, expected {n_nodes} or {3*n_nodes}."
                )

            mesh.point_data[base_name] = scalar_field

            # === Save VTK if requested ===
            if save_vtk:
                vtk_path = output_path / f"{base_name}_{row_idx:04d}.vtk"
                mesh.save(vtk_path)

            # === Visualization ===
            plotter = pv.Plotter(off_screen=export_pdf, window_size=(1200, 900))
            clim = [scalar_field.min(), scalar_field.max()]

            plotter.add_mesh(
                mesh,
                scalars=base_name,
                cmap="coolwarm",
                clim=clim,
                n_colors=n_colors,
                show_edges=False,
                show_scalar_bar=False,
            )

            scalar_bar = plotter.add_scalar_bar(
                title=plot_title,
                title_font_size=36,
                label_font_size=28,
                position_x=0.1,
                position_y=0.05,
                width=0.8,
                height=0.08,
                vertical=False,
                n_colors=n_colors,
                fmt="%.2e" if logscale else "%.2f",
            )
            scalar_bar.GetTitleTextProperty().SetFontFamilyToArial()
            scalar_bar.GetTitleTextProperty().SetBold(False)
            scalar_bar.GetLabelTextProperty().SetFontFamilyToArial()

            if field.lower() == "snapshots":
                plotter.camera.azimuth += 90   # rotate 60° around vertical axis
                plotter.camera.elevation += 100 # tilt slightly
                plotter.camera.roll += 40      # Rotate camera around the X-axis (flip upside down)
                plotter.camera.zoom(1.1)
            else:
                plotter.view_isometric()
                plotter.camera.azimuth += 90   # rotate 60° around vertical axis
                plotter.camera.elevation += 10 # tilt slightly
                plotter.camera.roll += 220      # Rotate camera around the X-axis (flip upside down)
                plotter.camera.zoom(1.2)

            # --- Export as PDF
            if export_pdf:
                pdf_filename = output_path / f"{base_name}_{row_idx:04d}.pdf"
                render_window = plotter.ren_win

                pdfExporter = vtk.vtkGL2PSExporter()
                pdfExporter.SetRenderWindow(render_window)
                pdfExporter.SetFileFormatToPDF()
                pdfExporter.SetFilePrefix(str(pdf_filename).replace(".pdf", ""))
                pdfExporter.CompressOff()
                pdfExporter.SetSortToBSP()
                pdfExporter.SetTextAsPath(False)
                pdfExporter.SetPS3Shading(True)
                pdfExporter.SetBestRoot(True)
                pdfExporter.DrawBackgroundOff()
                render_window.SetMultiSamples(0)

                pdfExporter.Write()

            plotter.close()