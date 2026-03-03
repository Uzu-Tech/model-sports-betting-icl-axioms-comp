import json
import logging
from pathlib import Path

import polars as pl
from fuzzywuzzy import process

from utils.constants import FOOTBALL_DATA_COUNTRY_MAP

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "http://api.clubelo.com"

def load_football_data(
    leagues: list[str], columns: list[str], num_years: int, save_path: Path
) -> pl.DataFrame:
    if save_path.exists():
        df = pl.read_parquet(save_path)
        return df

    num_years = num_years
    final_season = 25
    seasons = [
        f"{season - 1}{season}"
        for season in range(final_season, final_season - num_years, -1)
    ]

    BASE_URL = "https://www.football-data.co.uk/mmz4281/"
    all_dfs = [
        pl.read_csv(
            f"{BASE_URL}{season}/{league}.csv",
            null_values=["#", "1xBet"],
            try_parse_dates=True,
        ).select(columns)
        for league in leagues
        for season in seasons
    ]

    df = pl.concat(all_dfs)
    df = df.with_columns(pl.col("Div").replace(FOOTBALL_DATA_COUNTRY_MAP)).rename(
        {"Div": "League"}
    )
    df.write_parquet(save_path)
    return df


def load_club_elo(teams: list[str], num_years: int, mapping: dict, save_path: Path):
    if save_path.exists():
        df = pl.read_parquet(save_path)
        return df

    all_dfs = []
    for team in teams:
        logger.info(f"Pulling data for team: {team}")
        club_elo_team = mapping[team].replace(" ", "")
        url = f"{BASE_URL}/{club_elo_team}"

        try:
            df = pl.read_csv(url, null_values=["None"], try_parse_dates=True)
            df = (
                df
                .filter(
                    (pl.col("From").dt.year() >= 2025 - num_years) &
                    (pl.col("From").dt.year() <= 2025)
                )
                .drop("Rank", "Level")
            )

            if df.is_empty():
                raise FileNotFoundError(f"{team} data returns empty dataframe")

            logger.info(
                f"Successfully processed {team} data, len: {len(df)}"
            )
            all_dfs.append(df)

        except Exception as e:
                logger.error(f"Failed to pull data for {team}: {e}")
                continue
    
    df = pl.concat(all_dfs)
    df.write_parquet(save_path)
    return df


def get_club_elo_names(num_yrs: int, leagues: list[str]):
    teams = set()

    for league in leagues:
        logger.info(f"Pulling data for league: {league}")

        for year in range(2024, 2024 - num_yrs, -1):
            # Getting all teams in the league for that year by getting data for one date in that year
            date_str = f"{year}-12-31"
            url = f"{BASE_URL}/{date_str}"

            try:
                logger.debug(f"Fetching URL: {url}")

                df = pl.read_csv(url, null_values=["None"])
                league_teams = df.filter(
                    (pl.col("Country") == league) & (pl.col("Level") == 1)
                )
                new_teams = league_teams.get_column("Club").to_list()
                teams.update(new_teams)

                logger.info(
                    f"Successfully processed {league} for {year}. {len(teams)} unique teams so far."
                )

            except Exception as e:
                logger.error(f"Failed to pull data for {league} in {year}: {e}")
                continue

    logger.info(f"Extraction complete. Total unique teams found: {len(teams)}")
    return teams


def create_team_mapping(
    football_data_names: list[str], elo_names: set[str], save_path: Path
):
    logger.info(f"--- Num Teams: {len(football_data_names)} ---")
    mapping = {}
    for name in football_data_names:
        # Find the best match in the Elo list
        pair = process.extractOne(name, elo_names)
        if pair:
            best_match, score = pair  # type: ignore
            mapping[name] = best_match
            logger.info(f"Mapped: {name:20} -> {best_match} (Score: {score})")

    save_team_mapping(mapping, save_path)
    logger.info(f"--- Saved mapping to {save_path} ---")
    return mapping


def fix_incorrect_mappings(mapping: dict):
    mapping["Ath Madrid"] = "Atletico"
    mapping["Sp Lisbon"] = "Sporting"
    return mapping


def save_team_mapping(mapping: dict, save_path: Path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)


def load_team_mapping(save_path: Path):
    with open(save_path, "r", encoding="utf-8") as f:
        return json.load(f)
