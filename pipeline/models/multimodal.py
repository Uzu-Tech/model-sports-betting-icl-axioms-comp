import numpy as np
import polars as pl
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

class MultiModalLogitWrapper:
    def __init__(self, lasso: bool):
        self.model_res = None
        self.scaler = StandardScaler().set_output(transform="pandas")
        self.lasso = lasso

    def _prepare_features(
        self, X: pl.DataFrame, fit_scaler: bool = False
    ) -> np.ndarray:
        X = X.drop("match_id", "Datetime", "Season")
        # Scale the features
        if fit_scaler:
            scaled_data = self.scaler.fit_transform(X)  # type: ignore
        else:
            scaled_data = self.scaler.transform(X)  # type: ignore

        # Add the intercept AFTER scaling
        return sm.add_constant(scaled_data, has_constant="add")

    def fit(self, X: pl.DataFrame, y: pl.DataFrame) -> None:
        y = y.drop("match_id", "Datetime", "Season")
        features = self._prepare_features(X, fit_scaler=True)
        model = sm.MNLogit(y.to_pandas(), features)
        if self.lasso:
            self.model_res = model.fit_regularized(
                method="l1", alpha=2, L1_wt=0.9  # type: ignore
            )
        else:
            self.model_res = model.fit(method="bfgs", maxiter=1000, disp=False)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        if self.model_res is None:
            raise ValueError("Model must be fitted before prediction.")

        features = self._prepare_features(X, fit_scaler=False)
        return self.model_res.predict(features)
