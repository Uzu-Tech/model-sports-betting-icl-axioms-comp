from typing import Literal, Protocol

import numpy as np
import polars as pl

FormatType = Literal["number", "percent", "currency"]

SEASON_MATCH_DAYS = 120 # Rough estimate of number of match days in a season

class Metric(Protocol):
    __name__: str
    format: FormatType
    decimals: int
    signed: bool
    suffix: str

    def __call__(self, equity_series: pl.DataFrame) -> float: ...


def metric_config(
    fmt: FormatType = "number",
    decimals: int = 0,
    signed: bool = False,
    suffix: str = "",
):
    def decorator(func) -> Metric:
        func.format = fmt
        func.decimals = decimals
        func.signed = signed
        func.suffix = suffix
        return func

    return decorator


def get_log_returns(equity_series: pl.DataFrame) -> np.ndarray:
    return (
        equity_series.select(
            (pl.col("new_bankroll") / pl.col("prev_bankroll"))
            .log()
            .alias("log_returns")
        )
        .get_column("log_returns")
        .to_numpy()
    )


def season_adjust(value: float, equity_series: pl.DataFrame) -> float:
    mean_bets_per_season = get_mean_beats_per_season(equity_series)
    return value * np.sqrt(mean_bets_per_season) # type: ignore

def calmer_season_adjust(value: float, equity_series: pl.DataFrame) -> float:
    mean_bets_per_season = get_mean_beats_per_season(equity_series)
    return value * mean_bets_per_season # type: ignore

def get_mean_beats_per_season(equity_series: pl.DataFrame):
    return (
        equity_series
        .group_by(pl.col("Season"))
        .agg(pl.len().alias("bets_per_season"))
        .get_column("bets_per_season")
        .mean()
    )