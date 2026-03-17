import polars as pl

from pipeline.metrics.utils import metric_config


@metric_config(fmt="currency", decimals=2, signed=True)
def total_PNL(equity_series: pl.DataFrame) -> float:
    equity_series = equity_series.sort("Datetime")
    initial_bankroll = equity_series.get_column("prev_bankroll").first()
    final_bankroll = equity_series.get_column("new_bankroll").last()
    return final_bankroll - initial_bankroll  # type: ignore


@metric_config()
def number_of_bets(equity_series: pl.DataFrame) -> float:
    return len(equity_series)


@metric_config(fmt="currency", decimals=2, signed=True, suffix="Per Bet")
def mean_PNL(equity_series: pl.DataFrame) -> float:
    return equity_series.get_column("pnl").mean()  # type: ignore


@metric_config(fmt="currency", decimals=2, signed=True, suffix="Per Bet")
def median_PNL(equity_series: pl.DataFrame) -> float:
    return equity_series.get_column("pnl").median()  # type: ignore


@metric_config(fmt="percent", decimals=1)
def win_rate(equity_series: pl.DataFrame) -> float:
    return len(equity_series.filter(pl.col("pnl") > 0)) / len(equity_series)


@metric_config(fmt="percent", decimals=1)
def loss_rate(equity_series: pl.DataFrame) -> float:
    return len(equity_series.filter(pl.col("pnl") < 0)) / len(equity_series)


@metric_config(fmt="percent", decimals=1, signed=True, suffix="Per Season")
def compound_return(equity_series: pl.DataFrame) -> float:
    equity_series = equity_series.sort("Datetime")
    initial_bankroll = equity_series.get_column("prev_bankroll").first()
    final_bankroll = equity_series.get_column("new_bankroll").last()
    num_seasons = equity_series.get_column("Season").unique().count()

    return (final_bankroll - initial_bankroll) / (initial_bankroll * num_seasons)  # type: ignore
