"""
Required Data Structure for Performance Metrics:

- equity_series (pl.DataFrame):
    - match_id: str/int (Unique identifier)
    - date: datetime (For time-series growth)
    - stake: float (The amount risked on this bet)
    - pnl: float (Profit or Loss from this specific bet)
    - prev_bankroll: float (The running balance BEFORE this bet settled)
    - new_bankroll: float (The running balance AFTER this bet settled)
"""

from typing import Iterable, Protocol

import great_tables as gt
import polars as pl


class Metric(Protocol):
    __name__: str  # Explicitly tell the type checker this exists

    def __call__(self, equity_series: pl.DataFrame) -> float: ...


def get_table(results: dict, title: str) -> gt.GT:
    df = pl.DataFrame([results]).transpose(
        include_header=True, header_name="Metric", column_names=["Value"]
    )

    return (
        gt.GT(df)
        .tab_header(
            title=title,
        )
        .fmt_number(columns="Value", decimals=2)
    )


def get_metrics(equity_series: pl.DataFrame, metrics: Iterable[Metric]) -> dict:
    return {
        metric.__name__.replace("_", " ").capitalize(): metric(equity_series)
        for metric in metrics
    }
