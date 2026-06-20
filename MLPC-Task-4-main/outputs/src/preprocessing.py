"""Training-only preprocessing for the MLPC Task 4 features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class FittedPreprocessor:
    imputer: SimpleImputer
    scaler: StandardScaler

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply imputation and standardization learned from training data."""
        x_imp = self.imputer.transform(x)
        x_scaled = self.scaler.transform(x_imp)
        return x_scaled.astype(np.float32)


def fit_preprocessor(x_train: np.ndarray) -> FittedPreprocessor:
    """Fit median imputation and standard scaling on training data only."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_imp = imputer.fit_transform(x_train)
    scaler.fit(x_imp)
    return FittedPreprocessor(imputer=imputer, scaler=scaler)
