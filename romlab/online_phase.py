"""Online POD reconstruction and error evaluation."""

import numpy as np

class OnlinePhase:
    """Predict POD coefficients, reconstruct snapshots, and compute errors."""

    def __init__(self, fom, pod, offline):
        self.fom = fom
        self.pod = pod
        self.offline = offline
        self.predictions = {}
        self.errors = {}
        self.mean_errors = {}

    def reconstruct_sol(self):        
        X = self.fom.parameters_test[:, self.fom.param_cols]
        if self.offline.scaler is not None:
            X = self.offline.scaler.transform(X)

        for fld in self.pod.computed_fields:
            coeffs = self.offline.models[fld].predict(X, return_std=False)
            pred = coeffs @ self.pod.data[fld]["basis"].T + self.pod.data[fld]["lifting"]
            truth = getattr(self.fom, f"{self.pod.FIELDS[fld]}_test")
            rel = np.linalg.norm(pred - truth, axis=1) / np.linalg.norm(truth, axis=1)
            self.predictions[fld] = pred
            self.errors[fld] = rel
            self.mean_errors[fld] = rel.mean()

        preds = {f: self.predictions[f] for f in self.pod.computed_fields}

        return preds

    def info(self):
        for fld, rel in self.errors.items():
            print(f"{fld} (relative error) = {rel.mean():.2e}")