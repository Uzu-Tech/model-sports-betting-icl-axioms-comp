import polars as pl
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel

class OrdinalBettingWrapper:
    def __init__(self, predict_residuals):
        self.predict_residuals = predict_residuals

    def fit(self, X: pl.DataFrame, y: pl.Series, market_probs: pl.DataFrame) -> None:
        # For Ordinal, we use the Log-Probability of the Home Win as the baseline shift
        # This pushes the latent variable toward the 'Home' end of the scale
        home_offset = market_probs.select(
            (pl.col("home_prob") / (1 - pl.col("home_prob"))).log()
        ).to_numpy().flatten()
        # statsmodels requires Pandas for the internal formula handling
        model = OrderedModel(y, X.to_pandas(), offset=home_offset, distr='logit')
        # 'bfgs' is generally more stable for ordered models
        self.model_res = model.fit(method='bfgs', disp=False)

    def predict(self, X: pl.DataFrame, market_probs: pl.DataFrame) -> np.ndarray:
        home_offset = market_probs.select(
            (pl.col("home_prob") / (1 - pl.col("home_prob"))).log()
        ).to_numpy().flatten()
        return self.model_res.predict(X.to_pandas(), offset=home_offset).to_numpy()