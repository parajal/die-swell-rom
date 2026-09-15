"""Full-order snapshot container."""

import os
import numpy as np

class FOM:
    """Load and store training/test snapshots and parameters."""

    def __init__(self, data_folder=None, filenames_train=None, filenames_test=None,
                 model=None, param_cols=None, docker_folder=None,
                 mesh_file=None, seed = 42):
        self.model = model.lower() if model else None
        self.data_folder = data_folder
        self.docker_folder = docker_folder
        self.mesh_file = mesh_file

        (self.velocity_train, self.viscosity_train,
         self.mesh_train, self.parameters_train) = self._load_snapshots(data_folder, filenames_train)
        (self.velocity_test, self.viscosity_test,
         self.mesh_test, self.parameters_test) = self._load_snapshots(data_folder, filenames_test)

        self.param_cols = (list(param_cols) if param_cols is not None
                           else list(range(self.parameters_train.shape[1])))
        self.seed = seed
    def _load_snapshots(self, folder, filenames):
        """Load [velocity, second field, (mesh), parameters]; mesh is optional."""
        velocity, second, *mesh, params = filenames
        load = lambda name: np.atleast_2d(np.loadtxt(os.path.join(folder, name)))
        return load(velocity), load(second), load(mesh[0]) if mesh else None, load(params)

    def info(self):
        label = "tr(c)" if self.model == "extrudate_swell" else "viscosity"
        fields = [("velocity", self.velocity_train), (label, self.viscosity_train),
                  ("mesh_coor2", self.mesh_train)]
        dofs = ", ".join(f"{name} {a.shape[1]}" for name, a in fields if a is not None)
        print(f"{self.model} | {len(self.parameters_train)} train, "
              f"{len(self.parameters_test)} test")
        print(f"dofs: {dofs}")