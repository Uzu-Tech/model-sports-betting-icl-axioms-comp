import polars as pl
import xgboost as xgb

class MultiClassXGBoostWrapper:
    def __init__(self, params: dict, predict_residuals: bool):
        self.predict_residuals = predict_residuals
        self.params = params
        self.params['objective'] = 'multi:softprob'
        self.params['num_class'] = 3

    def fit(self, X: pl.DataFrame, y: pl.Series, market_probs: pl.DataFrame) -> None:
        # Convert probabilities p to log(p) to use as guess as a inverse softmax
        market_logits = market_probs.select(pl.all().log())

        if self.predict_residuals:
            data_train = xgb.DMatrix(
                X.to_pandas(), 
                label=y.to_numpy(), 
                base_margin=market_logits.to_numpy()
            )
        else:
            data_train = xgb.DMatrix(
                X.to_pandas(), 
                label=y.to_numpy()
            )
        
        self.model = xgb.train(self.params, data_train)

    def predict(self, X: pl.DataFrame, market_probs: pl.DataFrame):
        market_logits = market_probs.select(pl.all().log())

        data_test = xgb.DMatrix(X.to_pandas(), base_margin=market_logits.to_numpy())
        predictions = self.model.predict(data_test)
        return predictions