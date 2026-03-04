import polars as pl

from pipeline.metrics.utils import (
    metric_config,
)


@metric_config(fmt="percent", decimals=1)
def max_drawdown(equity_series: pl.DataFrame):
    return get_draw_downs(equity_series).max()


@metric_config(fmt="percent", decimals=1)
def average_drawdown(equity_series: pl.DataFrame):
    return get_draw_downs(equity_series).mean()


@metric_config(decimals=0, suffix="Bets")
def max_time_underwater(equity_series: pl.DataFrame) -> int:
    durations = get_draw_down_durations(equity_series)
    return int(durations.max()) if not durations.is_empty() else 0 # type: ignore


@metric_config(decimals=1, suffix="Bets")
def avg_time_underwater(equity_series: pl.DataFrame) -> float:
    durations = get_draw_down_durations(equity_series)
    return float(durations.mean()) if not durations.is_empty() else 0.0 # type: ignore


def get_draw_downs(equity_series: pl.DataFrame):
    return equity_series.select(
        (
            (pl.col("new_bankroll").cum_max() - pl.col("new_bankroll"))
            / pl.col("new_bankroll").cum_max()
        ).alias("drawdown")
    ).get_column("drawdown")

def get_draw_down_durations(equity_series: pl.DataFrame) -> pl.Series:
    return (
        equity_series
        .with_columns(
            (pl.col("new_bankroll") >= pl.col("new_bankroll").cum_max()).alias("is_hwm")
        )
        .with_columns(
            pl.col("is_hwm").cast(pl.Int32).cum_sum().alias("hwm_group")
        )
        .filter(~pl.col("is_hwm")) # Filter for time under water
        .group_by("hwm_group").agg(pl.len().alias("duration"))
        .get_column("duration")
    )