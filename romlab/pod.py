"""POD basis and coefficient computation."""

import numpy as np

class POD:
    """Compute POD bases and coefficients from snapshots."""

    def __init__(self, fom, centering=True, eps=1e-6):
        self.fom, self.centering, self.eps = fom, centering, eps
        self.data = {}

    def compute_basis(self, nmodes = None):
        for field, train in self.fom.train.items():
            lifting = train.mean(axis=0) * self.centering
            X = train - lifting
            U, s, _ = np.linalg.svd(X.T, full_matrices=False)
            n = max(1, int(np.sum(s / s[0] >= self.eps)))
            if nmodes is not None:
                n = nmodes
            self.data[field] = dict(basis=U[:, :n], lifting=lifting, svals=s/s[0],
                                    coeffs_train=X @ U[:, :n], nmodes=n)
        self.computed_fields = list(self.data)

    def info(self):
        for field, d in self.data.items():
            print(f"{field}: {d['nmodes']}/{len(d['svals'])} modes")