"""
Required Data Structure:

bet series (pl.DataFrame):
    - match_id: str/int (Unique identifier)
    - date: Ddatetime (for time-series growth)
    - model_probability: float
    - odds: float (Profit or loss from this specific bet)
    - outcome: str (win or loss)

Outputted Data Structure:

equity series (pl.DataFrame):
    - match_id: str/int (Unique identifier)
    - date: datetime (For time-series growth)
    - stake: float (The amount risked on this bet)
    - pnl: float (Profit or Loss from this specific bet)
    - prev_bankroll: float (The running balance BEFORE this bet settled)
    - new_bankroll: float (The running balance AFTER this bet settled)

"""

import polars as pl


def create_bet_series(ev_series: pl.DataFrame):
    bet_series = (
        ev_series.with_columns(EV=pl.max_horizontal("EV_A", "EV_H", "EV_D"))
        .with_columns(
            odds=(
                pl.when(pl.col("EV_A") == pl.col("EV"))
                .then(pl.col("MaxA"))
                .when(pl.col("EV_H") == pl.col("EV"))
                .then(pl.col("MaxH"))
                .otherwise(pl.col("MaxD"))
            ),
            model_probability=(
                pl.when(pl.col("EV_A") == pl.col("EV"))
                .then(pl.col("m_away_probs"))
                .when(pl.col("EV_H") == pl.col("EV"))
                .then(pl.col("m_home_probs"))
                .otherwise(pl.col("m_draw_probs"))
            ),
        )
        .with_columns(
            outcome=pl.col("FTR")
            .cast(pl.String)  # Move this here
            .replace({"0": "A", "1": "D", "2": "H"})  # Map strings to strings
        )
        .with_columns(
            model_predictions=(
                pl.when(pl.col("EV_A") == pl.col("EV"))
                .then(pl.lit("A"))
                .when(pl.col("EV_H") == pl.col("EV"))
                .then(pl.lit("H"))
                .otherwise(pl.lit("D"))
            )
        )
        .with_columns(
            outcome=pl.when(pl.col("outcome") == pl.col("model_predictions"))
            .then(pl.lit("win"))
            .otherwise(pl.lit("loss"))
        )
    )

    return bet_series.select(
        "match_id", "Datetime", "Season", "outcome", "odds", "model_probability"
    )


def create_ev_series(
    probs: pl.DataFrame,
    outcomes: pl.DataFrame,
    test_seasons: list[str],
    training_df: pl.DataFrame,
):
    ev_series = (
        outcomes.join(probs, on="match_id")
        .join(
            training_df.filter(pl.col("Season").is_in(test_seasons)).select(
                "match_id", "MaxH", "MaxA", "MaxD"
            ),
            on="match_id",
        )
        .with_columns(
            (
                pl.col("m_away_probs") * (pl.col("MaxA") - 1)
                - (1 - pl.col("m_away_probs"))
            ).alias("EV_A"),
            (
                pl.col("m_home_probs") * (pl.col("MaxH") - 1)
                - (1 - pl.col("m_home_probs"))
            ).alias("EV_H"),
            (
                pl.col("m_draw_probs") * (pl.col("MaxD") - 1)
                - (1 - pl.col("m_draw_probs"))
            ).alias("EV_D"),
        )
    )

    THRESHOLD = 0

    return ev_series.filter(
        pl.any_horizontal(
            pl.col("EV_A") >= THRESHOLD,
            pl.col("EV_H") >= THRESHOLD,
            pl.col("EV_D") >= THRESHOLD,
        )
    )


def kelly_criterion(bet_series, initial_bank_roll, fraction):
    bet_series = (
        bet_series.with_columns((pl.col("odds") - 1).alias("net_odds"))
        .with_columns(
            (
                (
                    (
                        pl.col("net_odds") * pl.col("model_probability")
                        - (1 - pl.col("model_probability"))
                    )
                    * fraction
                )
                / pl.col("net_odds")
            ).alias("kelly_fraction")
        )
        .with_columns(
            (
                pl.when(pl.col("outcome") == "win")
                .then(pl.col("kelly_fraction") * pl.col("net_odds"))
                .otherwise(-pl.col("kelly_fraction"))
            ).alias("return")
        )
        .with_columns(
            ((1 + pl.col("return")).cum_prod() * initial_bank_roll).alias(
                "new_bank_roll"
            )
        )
        .with_columns(
            (pl.col("new_bank_roll").shift().fill_null(initial_bank_roll)).alias(
                "prev_bank_roll"
            )
        )
        .with_columns(
            (pl.col("prev_bank_roll") * pl.col("kelly_fraction")).alias("stake"),
            (pl.col("return") * pl.col("prev_bank_roll")).alias("pnl"),
        )
        .drop(
            "model_probability",
            "odds",
            "outcome",
            "kelly_fraction",
            "net_odds",
            "return",
        )
    )

    return bet_series


def fixed_fraction(bet_series, initial_bank_roll, fraction):
    bet_series = (
        bet_series.with_columns((pl.col("odds") - 1).alias("net_odds"))
        .with_columns(
            (
                pl.when(pl.col("outcome") == "win")
                .then(fraction * pl.col("net_odds"))
                .otherwise(-fraction)
            ).alias("return")
        )
        .with_columns(
            ((1 + pl.col("return")).cum_prod() * initial_bank_roll).alias(
                "new_bank_roll"
            )
        )
        .with_columns(
            (pl.col("new_bank_roll").shift().fill_null(initial_bank_roll)).alias(
                "prev_bank_roll"
            )
        )
        .with_columns(
            (pl.col("prev_bank_roll") * fraction).alias("stake"),
            (pl.col("return") * pl.col("prev_bank_roll")).alias("pnl"),
        )
        .drop("model_probability", "odds", "outcome", "net_odds", "return")
    )

    return bet_series


def flat_bet(bet_series, initial_bank_roll, flat_bet):
    # 1. Calculate potential PnL (as if we never went bust)
    bet_series = bet_series.with_columns(
        pnl_potential=pl.when(pl.col("outcome") == "win")
        .then(flat_bet * (pl.col("odds") - 1))
        .otherwise(-flat_bet)
    )

    # 2. Identify the Bankruptcy Point
    # Once hypothetical_bank <= 0, 'is_bankrupt' becomes True and stays True
    bet_series = bet_series.with_columns(
        hypothetical_bank=initial_bank_roll + pl.col("pnl_potential").cum_sum()
    ).with_columns(
        is_bankrupt=(pl.col("hypothetical_bank") <= 0)
        .cast(pl.Int8)
        .cum_max()
        .cast(pl.Boolean)
    )

    # 3. Apply the "Kill Switch"
    # We shift the bankrupt status because you only stop betting AFTER you've hit zero
    bet_series = (
        bet_series.with_columns(
            currently_bust=pl.col("is_bankrupt").shift().fill_null(False)
        )
        .with_columns(
            stake=pl.when(pl.col("currently_bust")).then(0).otherwise(flat_bet),
            pnl=pl.when(pl.col("currently_bust"))
            .then(0)
            .otherwise(pl.col("pnl_potential")),
        )
        # 4. Final Bankroll calculation based on active bets only
        .with_columns(new_bank_roll=initial_bank_roll + pl.col("pnl").cum_sum())
        .with_columns(
            prev_bank_roll=pl.col("new_bank_roll").shift().fill_null(initial_bank_roll)
        )
        .drop("pnl_potential", "hypothetical_bank", "is_bankrupt", "currently_bust")
    )

    return bet_series
