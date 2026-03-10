import polars as pl
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson

class ExponentialPoissonModel:
    def __init__(self, time_decay: float = 0.0015):
        """
        xi: The time decay constant. 
        0.0015 is a common default for football (roughly 50% weight after 1.5 seasons).
        """
        self.time_decay = time_decay

    def _prepare_data(self, history_df: pl.DataFrame, current_date):
        """Converts match rows into two observation rows with time weights."""
        # We need a 'days_ago' column to calculate exponential decay
        df = history_df.with_columns(
            days_ago = (current_date - pl.col("date")).dt.total_days()
        )

        # Home team perspective (Attack = home_team, Defense = away_team)
        home_obs = df.select([
            pl.col("League").alias("league"),
            pl.col("HomeTeam").alias("team"),
            pl.col("AwayTeam").alias("opponent"),
            pl.col("FTHG").alias("goals"),
            pl.lit(1).alias("is_home"),
            pl.col("days_ago")
        ])

        # Away team perspective (Attack = away_team, Defense = home_team)
        away_obs = df.select([
            pl.col("League").alias("league"),
            pl.col("AwayTeam").alias("team"),
            pl.col("HomeTeam").alias("opponent"),
            pl.col("FTAG").alias("goals"),
            pl.lit(0).alias("is_home"),
            pl.col("days_ago")
        ])

        long_df = pl.concat([home_obs, away_obs]).to_pandas()
        
        # Apply the exponential weight
        long_df["weight"] = np.exp(-self.time_decay * long_df["days_ago"])
        return long_df

    def predict_match(self, history_df: pl.DataFrame, match: dict) -> np.ndarray:
        """
        Fits a GLM on the fly for a specific match.
        match: dict with 'home_team', 'away_team', and 'date'.
        """
        # 1. Prepare data with weights relative to the match date
        train_data = self._prepare_data(history_df, match["date"])

        # 2. Fit the Poisson GLM
        # Goals ~ Home_Advantage + Attacking_Strength + Opponent_Defensive_Weakness
        model = smf.glm(
            formula="goals ~ is_home + league + team + opponent",
            data=train_data,
            family=sm.families.Poisson(),
            freq_weights=train_data["weight"]
        ).fit()

        # 3. Predict Expected Goals (Lambdas)
        home_in = pd.DataFrame({
            "is_home": [1], 
            "league": [match["League"]],
            "team": [match["HomeTeam"]], 
            "opponent": [match["AwayTeam"]]
        })
        away_in = pd.DataFrame({
            "is_home": [1], 
            "league": [match["League"]],
            "team": [match["AwayTeam"]], 
            "opponent": [match["HomeTeam"]]
        })

        # If a team is new to the dataset, statsmodels might raise a ValueError
        l_home = model.predict(home_in).iloc[0]
        l_away = model.predict(away_in).iloc[0]
        return self._lambdas_to_probs(l_home, l_away)

    def _lambdas_to_probs(self, l_home, l_away, max_goals=10):
        goals = np.arange(0, max_goals + 1)
        prob_home_goals = poisson.pmf(goals, l_home)
        prob_away_goals = poisson.pmf(goals, l_away)

        # Grid of all possible scorelines (0-0 to 10-10)
        score_matrix = np.outer(prob_home_goals, prob_away_goals)

        # Sum probabilities for each outcome
        prob_draw = np.sum(np.diag(score_matrix))
        prob_home = np.sum(np.triu(score_matrix, k=1))
        prob_away = np.sum(np.tril(score_matrix, k=-1))

        # Normalize in case max_goals wasn't high enough
        total = prob_home + prob_draw + prob_away
        return np.array([prob_away/total, prob_draw/total, prob_home/total])