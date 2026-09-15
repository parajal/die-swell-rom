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
        self.stds = {}

    def reconstruct_sol(self, field=None, X=None, return_std=False):
        """Reconstruct every trained field, or only `field` (str or list)."""
        if field is None:
            field = list(self.offline.models)
        fields = [field] if isinstance(field, str) else list(field)

        if X is None:
            X = self.fom.parameters_test[:, self.fom.param_cols]
        X = np.atleast_2d(X)
        if self.offline.scaler is not None:
            X = self.offline.scaler.transform(X)

        for fld in fields:
            coeffs, std = self.offline.models[fld].predict(X, return_std=True)

            # sklearn drops the mode axis when there is a single mode
            coeffs = np.reshape(coeffs, (len(X), -1))
            std = np.broadcast_to(np.reshape(std, (len(X), -1)), coeffs.shape).copy()
            pred = coeffs @ self.pod.data[fld]["basis"].T

            truth = getattr(self.fom, f"{self.pod.FIELDS[fld]}_test")
            rel = (np.linalg.norm(pred - truth, axis=1) / np.linalg.norm(truth, axis=1)
                   if pred.shape == truth.shape else None)

            self.predictions[fld] = pred
            self.errors[fld] = rel
            self.mean_errors[fld] = None if rel is None else float(rel.mean())
            self.stds[fld] = std

        if isinstance(field, str):
            preds, stds = self.predictions[field], self.stds[field]
        else:
            preds = {f: self.predictions[f] for f in fields}
            stds = {f: self.stds[f] for f in fields}
        return (preds, stds) if return_std else preds

    def info(self):
        for fld, rel in self.errors.items():
            print(f"{fld} (relative error) = {rel.mean():.2e}")