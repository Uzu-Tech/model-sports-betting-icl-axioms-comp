import polars as pl
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

class MultiModalLogitWrapper:
    def __init__(self, lasso: bool):
        self.model_res = None
        self.scaler = StandardScaler().set_output(transform="pandas")
        self.lasso = lasso

    def _prepare_features(self, X: pl.DataFrame, market_probs: pl.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        # Calculate market logits
        market_features = market_probs.with_columns([
            (pl.col("home_prob") / pl.col("away_prob")).log().alias("mkt_logit_home"),
            (pl.col("draw_prob") / pl.col("away_prob")).log().alias("mkt_logit_draw")
        ]).select(["mkt_logit_home", "mkt_logit_draw"])
        
        combined = pl.concat([X, market_features], how="horizontal").to_pandas()
        # Scale the features
        if fit_scaler:
            scaled_data = self.scaler.fit_transform(combined) # type: ignore
        else:
            scaled_data = self.scaler.transform(combined) # type: ignore
            
        # Add the intercept AFTER scaling
        return sm.add_constant(scaled_data, has_constant='add')

    def fit(self, X: pl.DataFrame, y: pl.Series, market_probs: pl.DataFrame) -> None:
        features = self._prepare_features(X, market_probs, fit_scaler=True)
        model = sm.MNLogit(y.to_pandas(), features)
        if self.lasso:
            self.model_res = model.fit_regularized(
                    method='l1', 
                    alpha=2, # type: ignore
                    L1_wt=0.9
                )
        else:
            self.model_res = model.fit(method='bfgs', maxiter=1000, disp=False)

    def predict(self, X: pl.DataFrame, market_probs: pl.DataFrame) -> np.ndarray:
        if self.model_res is None:
            raise ValueError("Model must be fitted before prediction.")
            
        features = self._prepare_features(X, market_probs, fit_scaler=False)
        return self.model_res.predict(features)