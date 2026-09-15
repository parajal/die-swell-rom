"""POD basis and coefficient computation."""

import numpy as np

class POD:
    """Compute POD bases and coefficients from snapshots."""

    FIELDS = {
        "velocity": "velocity",
        "c-trace": "viscosity",
        "mesh_coor2": "mesh"}

    def __init__(self, fom, eps=1e-6):
        self.fom = fom
        self.eps = eps
        self.data = {}

    def compute_basis(self):
        for field, name in self.FIELDS.items():
            train = getattr(self.fom, f"{name}_train", None)

            basis, sigma, _ = np.linalg.svd(train.T, full_matrices=False)

            nmodes = max(1, np.sum(sigma / sigma[0] >= self.eps))

            self.data[field] = {
                "basis": basis[:, :nmodes],
                "svals": sigma / sigma[0],
                "coeffs_train": train @ basis[:, :nmodes],
                "nmodes": nmodes,
            }

        self.computed_fields = list(self.data.keys())

        if "velocity" in self.data:
            self.degfd_vel = self.data["velocity"]["basis"].shape[0]

        return self

    def info(self):
        for field, d in self.data.items():
            print(f"{field}: {d['nmodes']}/{len(d['svals'])} modes")