import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class BayesianLassoLogitWrapper:
    def __init__(self, C=2):
        # In a Bayesian context, 'C' is the inverse of the prior variance.
        # penalty='l1' with a SAGA solver approximates a Laplacian Prior.
        self.model = LogisticRegression(
            solver='saga',
            penalty='elasticnet',  # Use this instead of 'l1'
            l1_ratio=0.0,         # 1.0 means 100% L1 (Lasso)
            C=C,           
            max_iter=10000,
            tol=1e-4
        )
        self.scaler = StandardScaler().set_output(transform="pandas")

    def _prepare_features(self, X: pl.DataFrame, market_probs: pl.DataFrame, fit_scaler: bool = False):
        market_features = market_probs.with_columns([
            (pl.col("home_prob") / pl.col("draw_prob")).log().alias("mkt_logit_home"),
            (pl.col("away_prob") / pl.col("draw_prob")).log().alias("mkt_logit_away")
        ]).select(["mkt_logit_home", "mkt_logit_away"])
        
        # Combine your residuals (X) with these market features
        combined = pl.concat([X, market_features], how="horizontal")
        
        # Convert to numpy for the model
        data_array = combined.to_pandas()
        if fit_scaler:
            return self.scaler.fit_transform(data_array) # type: ignore
        else:
            return self.scaler.transform(data_array) # type: ignore

    def fit(self, X: pl.DataFrame, y: pl.Series, market_probs: pl.DataFrame):
        features = self._prepare_features(X, market_probs, fit_scaler=True)
        self.model.fit(features, y.to_numpy().ravel())

    def predict(self, X: pl.DataFrame, market_probs: pl.DataFrame):
        features = self._prepare_features(X, market_probs, fit_scaler=False)
        # Bayesian models give you "Soft" probabilities
        return self.model.predict_proba(features)