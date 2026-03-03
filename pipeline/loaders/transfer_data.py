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

def get_transfer_data_leagues(leagues: list[str]):
    return [
        TRANSFER_DATA_COUNTRY_MAP.inverse[league] for league in leagues
    ]