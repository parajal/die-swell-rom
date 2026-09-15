import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class OfflinePhase:
    """Train one GPR per POD field, from parameters to POD coefficients."""

    def __init__(self, fom, pod, x_scaler="minmax", verbose=True):
        self.fom = fom
        self.pod = pod
        self.x_scaler = x_scaler
        self.verbose = verbose
        self.scaler = None
        self.models = {}

    def train_rom(self):
        """Fit one GPR per POD field."""
        X = self.fom.parameters_train[:, self.fom.param_cols]

        if self.x_scaler == "minmax":
            self.scaler = MinMaxScaler()
        elif self.x_scaler == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = None
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)

        kernel = ConstantKernel(1.0, (1e-6, 1e6)) * RBF(
            length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-3, 1e3))

        self.models = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for field, data in self.pod.data.items():
                self.models[field] = GaussianProcessRegressor(
                    kernel=kernel, alpha=1e-10, normalize_y=False,
                    n_restarts_optimizer=100, random_state=self.fom.seed,
                ).fit(X, data["coeffs_train"])
        return self

    def info(self):
        for field, model in self.models.items():
            print(f"{field}: {model.kernel_}")