import logging
from pathlib import Path

import polars as pl

from utils.constants import TRANSFER_DATA_COUNTRY_MAP

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_clubs(leagues: list[str], clubs_path: Path):
    logger.info(f"Loading clubs from file: {clubs_path}")
    df = pl.read_csv(clubs_path)

    transfer_data_leagues = get_transfer_data_leagues(leagues)

    df = (
        df.filter(pl.col("domestic_competition_id").is_in(transfer_data_leagues))
        .select("club_id", "domestic_competition_id", "club_code")
        .rename({"domestic_competition_id": "league"})
    )

    logger.info("Clubs loaded successfully")
    return df


def fix_incorrect_mappings(mapping: dict):
    manual_fixes = {
        "Man City": "manchester-city",
        "Porto": "fc-porto",
        "Sp Lisbon": "sporting-lissabon",
        "Napoli": "ssc-neapel",
        "Torino": "fc-turin",
        "Nice": "ogc-nizza",
        "Aris": "aris-thessaloniki",
        "Verona": "hellas-verona",
        "Man United": "manchester-united",
        "Angers": "sco-angers",
        "Aves": "desportivo-aves",
        "Fiorentina": "ac-florenz",
        "Buyuksehyr": "istanbul-basaksehir"
    }
    mapping.update(manual_fixes)
    return mapping


def get_games(
    leagues: list[str],
    club_df: pl.DataFrame,
    num_years: int,
    games_path: Path,
):
    transfer_data_leagues = get_transfer_data_leagues(leagues)

    logger.info(f"Loading games from file: {games_path}")

    df = pl.read_csv(games_path, try_parse_dates=True)

    df = df.filter(
        pl.col("competition_id").is_in(transfer_data_leagues)
        & (pl.col("date").dt.year() >= 2025 - num_years)
        & (pl.col("date").dt.year() <= 2025)
    ).select("game_id", "date", "competition_id", "home_club_id", "away_club_id")

    lookup = club_df.select(pl.col("club_id"), pl.col("club_code"))

    df = (
        df.join(lookup, left_on="home_club_id", right_on="club_id", how="left")
        .rename({"club_code": "home_team"})
        .join(lookup, left_on="away_club_id", right_on="club_id", how="left")
        .rename({"club_code": "away_team"})
    )

    logger.info("Games loaded successfully")
    return df


def get_team_transfer_values(
    game_df: pl.DataFrame,
    lineups_path: Path,
    valuations_path: Path,
    appearances_path: Path,
):
    logger.info(f"Loading lineups from file: {lineups_path}")
    lineups_df = pl.read_csv(lineups_path)
    logger.info("Lineups Read Successfully")

    logger.info(f"Loading valuations from file: {valuations_path}")
    vals_df = pl.read_csv(valuations_path)
    logger.info("Valuations Read Successfully")

    logger.info(f"Loading appearances from file: {appearances_path}")
    appearances_df = pl.read_csv(appearances_path)
    logger.info("Appearances Read Successfully")

    game_ids = game_df.get_column("game_id").unique().to_list()

    lineups_df = lineups_df.filter(
        pl.col("game_id").is_in(game_ids) & (pl.col("type") == "starting_lineup")
    ).select("date", "game_id", "player_id")

    lineups_df = lineups_df.sort("date")
    vals_df = vals_df.sort("date")

    market_value_df = lineups_df.join_asof(
        vals_df, on="date", by="player_id", strategy="backward"
    ).select("game_id", "date", "player_id", "market_value_in_eur")

    market_value_df = market_value_df.join(
        appearances_df.select("game_id", "player_id", "player_club_id"),
        on=["game_id", "player_id"],
        how="left",
    )

    market_value_df = market_value_df.group_by(["game_id", "player_club_id"]).agg(
        pl.col("market_value_in_eur").mean().alias("mean_market_val")
    )

    return market_value_df


def get_game_market_values(
    game_df: pl.DataFrame,
    team_values: pl.DataFrame
):
    game_market_vals_df = game_df.join(
        team_values.select("game_id", "player_club_id", "mean_market_val"),
        left_on=["game_id", "home_club_id"],
        right_on=["game_id", "player_club_id"],
        how="left"
    ).rename({"mean_market_val": "home_mean_market_val"})

    game_market_vals_df = game_market_vals_df.join(
        team_values.select("game_id", "player_club_id", "mean_market_val"),
        left_on=["game_id", "away_club_id"],
        right_on=["game_id", "player_club_id"],
        how="left"
    ).rename({"mean_market_val": "away_mean_market_val"})

    return game_market_vals_df


def get_transfer_data_leagues(leagues: list[str]):
    return [TRANSFER_DATA_COUNTRY_MAP.inverse[league] for league in leagues]
