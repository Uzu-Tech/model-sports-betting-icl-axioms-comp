import numpy as np
import polars as pl
import scipy.stats as stats

from pipeline.metrics.utils import (
    get_log_returns,
    season_adjust,
    metric_config,
)


@metric_config(fmt="percent", decimals=1, suffix="Season-Adjusted")
def volatility(equity_series: pl.DataFrame):
    log_returns = get_log_returns(equity_series)
    return season_adjust(log_returns.std(ddof=1), equity_series)


@metric_config(fmt="percent", decimals=1, suffix="Season-Adjusted")
def downside_volatility(equity_series: pl.DataFrame):
    log_returns = get_log_returns(equity_series)
    downside_returns = np.clip(log_returns, a_min=None, a_max=0)
    downside_vol = np.sqrt(np.mean(downside_returns ** 2))
    return season_adjust(downside_vol, equity_series)


@metric_config(fmt="currency", decimals=2, signed=True, suffix="PNL")
def best_bet(equity_series: pl.DataFrame):
    return equity_series.get_column("pnl").max()


@metric_config(fmt="currency", decimals=2, signed=True, suffix="PNL")
def worst_bet(equity_series: pl.DataFrame):
    return equity_series.get_column("pnl").min()


@metric_config(decimals=3)
def skewness(equity_series: pl.DataFrame) -> float:
    log_returns = get_log_returns(equity_series)
    return float(stats.skew(log_returns, bias=False))


@metric_config(decimals=3, suffix="Excess")
def kurtosis(equity_series: pl.DataFrame) -> float:
    log_returns = get_log_returns(equity_series)
    return float(stats.kurtosis(log_returns, fisher=True, bias=False))
