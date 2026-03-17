from typing import Protocol
import polars as pl
import numpy as np
from sklearn.metrics import log_loss
from pipeline.models.possion import ExponentialPoissonModel

"""
market probabilities:
- implied_home
- implied_away
- implied_draw 

"""

class BettingModel(Protocol):
    def fit(self, X: pl.DataFrame, y: pl.DataFrame, market_probs: pl.DataFrame) -> None:
        ...

    def predict(self, X: pl.DataFrame, market_probs: pl.DataFrame) -> pl.Series:
        ...

class SequentialSeasonValidator:
    def __init__(
            self, 
            model_wrapper, 
            features: list, 
            start_season: str, 
            num_seasons: int, 
            target_col: str
        ):
        self.model = model_wrapper
        self.features = features
        self.start_season = start_season
        self.num_seasons = num_seasons
        self.target_col = target_col
        self.market_cols = ["away_prob", "draw_prob", "home_prob"]

    def _filter_seasons(self, df: pl.DataFrame):
        """Extracts the relevant seasons based on start_season and num_seasons."""
        all_seasons = df["Season"].unique(maintain_order=True).to_list()
        try:
            start_idx = all_seasons.index(self.start_season)
        except ValueError:
            raise ValueError(f"Start season {self.start_season} not found in data.")
            
        return all_seasons[start_idx : start_idx + self.num_seasons]

    def fit_all_training_data(self, df: pl.DataFrame):
        target_seasons = self._filter_seasons(df)
        train_df = df.filter(pl.col("Season").is_in(target_seasons))
        self.model.fit(
            train_df.select(self.features), 
            train_df.select(self.target_col), 
            train_df.select(self.market_cols)
        )

    def _execute_iteration(self, train_df: pl.DataFrame, val_df: pl.DataFrame, season_name: str):
        """Fits on training data and evaluates on the validation season."""
        # 1. Fit
        self.model.fit(
            train_df.select(self.features), 
            train_df.select(self.target_col), 
            train_df.select(self.market_cols)
        )
        
        # 2. Predict
        pred_probs = self.model.predict(
            val_df.select(self.features), 
            val_df.select(self.market_cols)
        )

        # 3. Score
        val_y = val_df.select(self.target_col)
        bookie_probs = val_df.select(self.market_cols).to_numpy()
        
        m_loss = log_loss(val_y, pred_probs, labels=[0, 1, 2])
        b_loss = log_loss(val_y, bookie_probs, labels=[0, 1, 2])

        return {
            "validation_season": season_name,
            "train_size": train_df.height,
            "model_loss": m_loss,
            "bookie_loss": b_loss,
            "improvement": b_loss - m_loss
        }

    def run(self, df: pl.DataFrame):
        df = df.sort("Datetime")
        target_seasons = self._filter_seasons(df)
        results = []

        # Iterate through seasons starting from the second one in our list
        for i in range(1, len(target_seasons)):
            current_val_season = target_seasons[i]
            completed_seasons = target_seasons[:i]

            # Train on all history up to the start of the validation season
            train_df = df.filter(pl.col("Season").is_in(completed_seasons))
            val_df = df.filter(pl.col("Season") == current_val_season)

            res = self._execute_iteration(train_df, val_df, current_val_season)
            results.append(res)

        return pl.DataFrame(results)


class PoissonSeasonValidator:
    def __init__(
            self, 
            model_wrapper: ExponentialPoissonModel, 
            start_season: str, 
            num_seasons: int, 
            target_col: str
        ):
        self.model = model_wrapper
        self.start_season = start_season
        self.num_seasons = num_seasons
        self.target_col = target_col
        self.market_cols = ["away_prob", "draw_prob", "home_prob"]

    def _filter_seasons(self, df: pl.DataFrame):
        all_seasons = df["season"].unique(maintain_order=True).to_list()
        try:
            start_idx = all_seasons.index(self.start_season)
        except ValueError:
            raise ValueError(f"Start season {self.start_season} not found.")
        return all_seasons[start_idx : start_idx + self.num_seasons]

    def _predict_season_match_by_match(self, train_history: pl.DataFrame, val_season_df: pl.DataFrame):
        all_preds = []
        
        # 'Current History' grows as we move through the validation season
        current_history = train_history.clone()
        
        for match in val_season_df.iter_rows(named=True):
            # 1. Prepare current match metadata
            # 2. Predict using history up to this second
            # (Assuming your Poisson wrapper has a predict_single_match method)
            pred = self.model.predict_match(current_history, match)
            all_preds.append(pred)
            
            # 3. Add this match to history so next matches can use this result
            # This mimics real-life where we know the result of game 1 before game 10
            # Convert dict match back to a small dataframe to concat
            match_row = pl.DataFrame([match])
            current_history = pl.concat([current_history, match_row])
            
        return np.array(all_preds)

    def run(self, df: pl.DataFrame):
        df = df.sort("date")
        target_seasons = self._filter_seasons(df)
        results = []

        for i in range(1, len(target_seasons)):
            current_val_season = target_seasons[i]
            completed_seasons = target_seasons[:i]

            train_df = df.filter(pl.col("season").is_in(completed_seasons))
            val_df = df.filter(pl.col("season") == current_val_season)

            # Sequential prediction within the season
            predictions = self._predict_season_match_by_match(train_df, val_df)

            # Scoring
            y_true = val_df.select(self.market_cols).to_numpy().flatten()
            bookie_probs = val_df.select(self.market_cols).to_numpy()
            
            m_loss = log_loss(y_true, predictions, labels=[0, 1, 2])
            b_loss = log_loss(y_true, bookie_probs, labels=[0, 1, 2])

            results.append({
                "validation_season": current_val_season,
                "model_loss": m_loss,
                "bookie_loss": b_loss,
                "improvement": b_loss - m_loss
            })

        return pl.DataFrame(results)