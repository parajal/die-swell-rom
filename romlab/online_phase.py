"""Online POD reconstruction and error evaluation."""

import numpy as np

class OnlinePhase:
    """Predict POD coefficients, reconstruct snapshots, and compute errors."""

    def __init__(self, fom, pod, offline):
        self.fom, self.pod, self.offline = fom, pod, offline
        self.predictions, self.errors, self.mean_errors = {}, {}, {}

    def reconstruct_sol(self):
        X = self.fom.parameters_test[:, self.fom.param_cols]
        if self.offline.scaler is not None:
            X = self.offline.scaler.transform(X)

        for fld, d in self.pod.data.items():
            coeffs = self.offline.models[fld].predict(X).reshape(len(X), -1)
            rom_pred = coeffs @ d["basis"].T + d["lifting"]
            truth = self.fom.test[fld]
            self.predictions[fld] = rom_pred
            self.errors[fld] = np.linalg.norm(rom_pred - truth, axis=1) / np.linalg.norm(truth, axis=1)
            self.mean_errors[fld] = self.errors[fld].mean()

    def info(self):
        for fld, err in self.mean_errors.items():
            print(f"{fld} (relative error) = {err:.2e}")