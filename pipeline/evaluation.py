"""
From risk management stage this script needs:
- The initial bankroll
- A polars Dataframe with two columns:
    - date: Every date where at least one match happened from the a given starting date
    - bankroll: Updated Bankroll after all the betting done that day
"""

from typing import Iterable, Protocol
import great_tables as gt
import polars as pl


class Metric(Protocol):
    __name__: str  # Explicitly tell the type checker this exists

    def __call__(
        self, initial_bankroll: float, equity_series: pl.DataFrame
    ) -> float: ...


def get_table(results: dict, title: str) -> gt.GT:
    df = pl.DataFrame([results]).transpose(
        include_header=True, 
        header_name="Metric", 
        column_names=["Value"]
    )

    return (
        gt.GT(df)
        .tab_header(
            title=title,
        )
        .fmt_number(columns="Value", decimals=2)
    )

def get_metrics(
    initial_bankroll: float, equity_series: pl.DataFrame, metrics: Iterable[Metric]
) -> dict:
    return {
        metric.__name__.replace("_", " ").capitalize(): metric(
            initial_bankroll, equity_series
        )
        for metric in metrics
    }
