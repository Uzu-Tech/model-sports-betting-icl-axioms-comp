import numpy as np
import polars as pl

from pipeline.metrics.utils import (
    calmer_season_adjust,
    get_log_returns,
    metric_config,
    season_adjust,
)
from pipeline.metrics.drawdown import max_drawdown

@metric_config(decimals=3, suffix="Season-Adjusted")
def sharpe_ratio(equity_series: pl.DataFrame):
    log_returns = get_log_returns(equity_series)
    sharpe = log_returns.mean() / log_returns.std(ddof=1)
    return season_adjust(sharpe, equity_series)


@metric_config(decimals=3, suffix="Season-Adjusted")
def sortino_ratio(equity_series: pl.DataFrame):
    log_returns = get_log_returns(equity_series)
    downside_returns = np.clip(log_returns, a_min=None, a_max=0)
    raw_downside_dev = np.sqrt(np.mean(np.square(downside_returns)))

    sortino = log_returns.mean() / raw_downside_dev
    return season_adjust(sortino, equity_series)


@metric_config(decimals=3, suffix="Season-Adjusted")
def calmer_ratio(equity_series: pl.DataFrame):
    mean_per_returns = (
        equity_series.select(
            (
                (pl.col("new_bankroll") - pl.col("prev_bankroll"))
                / pl.col("prev_bankroll")
            ).alias("per_returns")
        )
        .get_column("per_returns")
        .mean()
    )

    season_return = calmer_season_adjust(float(mean_per_returns), equity_series)  # type: ignore
    return season_return / max_drawdown(equity_series)