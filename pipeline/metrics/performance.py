import polars as pl

def total_PNL(equity_series: pl.DataFrame) -> float:
    equity_series = equity_series.sort("date")
    initial_bankroll = equity_series.get_column("prev_bankroll").first()
    final_bankroll = equity_series.get_column("new_bankroll").last()
    return final_bankroll - initial_bankroll  # type: ignore


def number_of_bets(equity_series: pl.DataFrame) -> float:
    return len(equity_series)


def mean_PNL(equity_series: pl.DataFrame) -> float:
    return equity_series.get_column("pnl").mean()  # type: ignore


def median_PNL(equity_series: pl.DataFrame) -> float:
    return equity_series.get_column("pnl").median()  # type: ignore


def win_rate(equity_series: pl.DataFrame) -> float:
    return len(equity_series.filter(pl.col("pnl") > 0)) / len(equity_series)


def loss_rate(equity_series: pl.DataFrame) -> float:
    return len(equity_series.filter(pl.col("pnl") < 0)) / len(equity_series)
