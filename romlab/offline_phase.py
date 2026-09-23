"""Offline GPR training from parameters to POD coefficients."""

import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SCALERS = {"minmax": MinMaxScaler, "standard": StandardScaler}

class OfflinePhase:
    """Train one GPR per POD field, from parameters to POD coefficients."""

    def __init__(self, fom, pod, x_scaler="minmax"):
        self.fom, self.pod, self.x_scaler = fom, pod, x_scaler
        self.scaler, self.models = None, {}

    def train_rom(self):
        X = self.fom.parameters_train[:, self.fom.param_cols]
        self.scaler = SCALERS.get(self.x_scaler, lambda: None)()
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)

        kernel = ConstantKernel(1.0, (1e-6, 1e6)) * RBF(np.ones(X.shape[1]), (5e-2, 1e3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.models = {field: GaussianProcessRegressor(kernel, n_restarts_optimizer=20,
                random_state=self.fom.seed).fit(X, d["coeffs_train"])
                for field, d in self.pod.data.items()}

    def info(self):
        for field, model in self.models.items():
            print(f"{field}: {model.kernel_}")