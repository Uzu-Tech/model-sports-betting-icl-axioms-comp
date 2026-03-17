import polars as pl
import polars.selectors as cs

def add_seasons(df: pl.DataFrame, season_start_month: int):
    return df.with_columns(
        Season = (
            pl.when(pl.col("Datetime").dt.month() >= season_start_month)
            .then(
                pl.concat_str([
                    pl.col("Datetime").dt.year(),
                    pl.lit("/"),
                    pl.col("Datetime").dt.year() + 1
                ])
            )
            .otherwise(
                pl.concat_str([
                    pl.col("Datetime").dt.year() - 1,
                    pl.lit("/"),
                    pl.col("Datetime").dt.year()
                ])
            )
        )
    )


def add_custom_ewm_features(
    df: pl.DataFrame, windows: list[int], feature_definitions: dict[str, dict]
):
    home_side = df.select(
        [
            pl.col("match_id"),
            pl.col("Datetime"),
            pl.col("HomeTeam").alias("team"),
            *[
                (
                    val["home"].alias(name)
                    if isinstance(val["home"], pl.Expr)
                    else pl.col(val["home"]).alias(name)
                )
                for name, val in feature_definitions.items()
            ],
        ]
    )

    away_side = df.select(
        [
            pl.col("match_id"),
            pl.col("Datetime"),
            pl.col("AwayTeam").alias("team"),
            *[
                (
                    val["away"].alias(name)
                    if isinstance(val["away"], pl.Expr)
                    else pl.col(val["away"]).alias(name)
                )
                for name, val in feature_definitions.items()
            ],
        ]
    )

    stacked = pl.concat([home_side, away_side]).sort("Datetime", "team")
    feature_names = list(feature_definitions.keys())

    stacked = stacked.with_columns(
        [
            pl.col(name)
            .ewm_mean(span=window)
            .over("team")
            .name.suffix(f"_ewm_{window}")
            for name in feature_names
            for window in windows
        ]
    )

    latest_stats = (
        stacked.sort("Datetime")
        .group_by("team")
        .last()
    )

    stacked = stacked.with_columns([cs.contains("ewm").shift().over("team")])

    ewm_cols = [c for c in stacked.columns if "_ewm_" in c]

    for side in ("Home", "Away"):
        team_col = f"{side}Team"
        df = df.join(
            stacked.select("match_id", "team", cs.contains("ewm")),
            left_on=["match_id", team_col],
            right_on=["match_id", "team"],
            how="left",
        ).rename({c: f"{side.lower()}_{c}" for c in ewm_cols})

    return df, latest_stats
