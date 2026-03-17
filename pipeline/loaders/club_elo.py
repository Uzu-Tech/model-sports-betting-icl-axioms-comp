import logging
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CLUB_ELO_URL = "http://api.clubelo.com"


def load_club_elo(
    teams: list[str], num_years: int, mapping: dict[str, str], save_path: Path
):
    if save_path.exists():
        df = pl.read_parquet(save_path)
        return df

    all_dfs = []
    for team in teams:
        logger.info(f"Pulling data for team: {team}")
        club_elo_team = mapping[team].replace(" ", "")
        url = f"{CLUB_ELO_URL}/{club_elo_team}"

        try:
            df = pl.read_csv(url, null_values=["None"], try_parse_dates=True)
            df = df.filter(
                (pl.col("From").dt.year() >= 2026 - num_years)
                & (pl.col("From").dt.year() <= 2026)
            ).drop("Rank", "Level")

            if df.is_empty():
                raise FileNotFoundError(f"{team} data returns empty dataframe")

            logger.info(f"Successfully processed {team} data, len: {len(df)}")
            all_dfs.append(df)

        except Exception as e:
            logger.error(f"Failed to pull data for {team}: {e}")
            continue

    df = pl.concat(all_dfs)
    df.write_parquet(save_path)
    return df


def get_club_elo_names(num_yrs: int, leagues: list[str]) -> list[str]:
    teams = set()

    for league in leagues:
        logger.info(f"Pulling data for league: {league}")

        for year in range(2025, 2025 - num_yrs, -1):
            # Getting all teams in the league for that year by getting data for one date in that year
            date_str = f"{year}-12-31"
            url = f"{CLUB_ELO_URL}/{date_str}"

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
    return list(teams)


def fix_incorrect_mappings(mapping: dict):
    mapping["Ath Madrid"] = "Atletico"
    mapping["Sp Lisbon"] = "Sporting"
    return mapping
