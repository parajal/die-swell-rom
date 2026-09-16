"""Full-order snapshot container."""

import os
import numpy as np


class FOM:
    """Load and store training/test snapshots and parameters."""

    FIELDS = ("velocity", "c-trace", "mesh_coor2")

    def __init__(self, data_folder, filenames_train, filenames_test,
                 param_cols, mesh_file=None, seed=42):
        self.data_folder, self.mesh_file, self.seed = data_folder, mesh_file, seed
        self.param_cols = list(param_cols)
        self.train, self.parameters_train = self._load(filenames_train)
        self.test, self.parameters_test = self._load(filenames_test)

    def _load(self, filenames):
        *fields, params = [np.atleast_2d(np.loadtxt(os.path.join(self.data_folder, f)))
                           for f in filenames]
        return dict(zip(self.FIELDS, fields, strict=True)), params

    def info(self):
        print(f"{len(self.parameters_train)} train, {len(self.parameters_test)} test")
        print("dofs:", ", ".join(f"{f} {a.shape[1]}" for f, a in self.train.items()))