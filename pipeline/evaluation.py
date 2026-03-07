"""
Required Data Structure for Performance Metrics:

- equity_series (pl.DataFrame):
    - match_id: str/int (Unique identifier)
    - date: datetime (For time-series growth)
    - stake: float (The amount risked on this bet)
    - pnl: float (Profit or Loss from this specific bet)
    - prev_bankroll: float (The running balance BEFORE this bet settled)
    - new_bankroll: float (The running balance AFTER this bet settled)

- bet_series (pl.DataFrame):
    - match_id: str/int (Unique identifier)
    - date: datetime (For time-series growth)
    - model_probability: float
    - odds: float (Profit or Loss from this specific bet)
    - outcome: str (win or loss)
"""

from typing import Iterable, Mapping

import great_tables as gt
import polars as pl

from pipeline.metrics.utils import Metric


def evaluate_returns(
    equity_series: pl.DataFrame, metrics: Mapping[str, Iterable[Metric]]
):
    rows = [
        {
            "Type": metric_type,
            "Metric": metric.__name__.replace("_", " ").title()
            + (f" ({metric.suffix})" if metric.suffix != "" else metric.suffix),
            "Value": metric(equity_series),
            "fmt": metric.format,
            "dec": metric.decimals,
            "sign": metric.signed,
        }
        for metric_type, metric_list in metrics.items()
        for metric in metric_list
    ]

    df = pl.DataFrame(rows)
    return df


def get_evaluation_table(df: pl.DataFrame, title: str, subtitle: str):
    table = (
        gt.GT(df, groupname_col="Type", rowname_col="Metric")
        .cols_hide(columns=["fmt", "dec", "sign"])
        .tab_header(
            title=title,
            subtitle=subtitle,
        )
    )
    # Loop through unique combinations of format, decimals, and sign
    formats_to_apply = df.select(["fmt", "dec", "sign"]).unique()

    for row in formats_to_apply.to_dicts():
        selector = (
            (pl.col("fmt") == row["fmt"])
            & (pl.col("dec") == row["dec"])
            & (pl.col("sign") == row["sign"])
        )

        if row["fmt"] == "percent":
            table = table.fmt_percent(
                columns="Value",
                rows=selector,
                decimals=row["dec"],
                force_sign=row["sign"],
            )
        elif row["fmt"] == "currency":
            table = table.fmt_currency(
                columns="Value",
                rows=selector,
                decimals=row["dec"],
                force_sign=row["sign"],
                currency="USD",
            )
        else:
            table = table.fmt_number(
                columns="Value",
                rows=selector,
                decimals=row["dec"],
                force_sign=row["sign"],
            )

    return table


def apply_clean_theme(gt_table: gt.GT):
    return (
        gt_table
        .tab_style(
            style=gt.style.fill(color="#2c3e50"),  # Deep Slate Blue
            locations=gt.loc.column_labels(),
        )
        .tab_style(
            style=gt.style.text(
                color="white", weight="bold", transform="uppercase", size="12px"
            ),
            locations=gt.loc.column_labels(),
        )
        .tab_style(
            style=[
                gt.style.fill(color="#f2f4f4"),  # Very light grey
                gt.style.text(weight="bold", color="#34495e"),
            ],
            locations=gt.loc.row_groups(),
        )
        .tab_style(
            style=gt.style.text(weight="normal", color="#2c3e50"),
            locations=gt.loc.stub(),
        )
        .tab_options(
            column_labels_border_bottom_color="#1a252f",
            table_font_names=[
                "Inter",
                "Segoe UI",
                "Roboto",
                "Helvetica Neue",
                "Arial",
                "sans-serif",
            ],
        )
        .tab_style(
            style=gt.style.text(
                font="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
                size="15px",
            ),
            locations=gt.loc.body(columns="Value"),
        )
    )
