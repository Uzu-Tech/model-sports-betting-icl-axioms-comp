from pathlib import Path
import polars as pl
import logging
from utils.constants import FOOTBALL_DATA_COUNTRY_MAP

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_football_data(
    leagues: list[str],
    columns: list[str],
    num_years: int,
    save_path: Path,
) -> pl.DataFrame:
    fd_leagues = [FOOTBALL_DATA_COUNTRY_MAP.inverse[league] for league in leagues]

    if save_path.exists():
        if logger:
            logger.info(f"Loading data from file {save_path}")

        df = pl.read_parquet(save_path)
        return df

    num_years = num_years
    final_season = 25
    seasons = [
        f"{season - 1}{season}"
        for season in range(final_season, final_season - num_years, -1)
    ]

    BASE_URL = "https://www.football-data.co.uk/mmz4281/"

    if logger:
        logger.info(f"Fetching {num_years} of data from football data")

    all_dfs = [
        pl.read_csv(
            f"{BASE_URL}{season}/{league}.csv",
            null_values=["#", "1xBet"],
            try_parse_dates=True,
        ).select(columns)
        for league in fd_leagues
        for season in seasons
    ]

    df = pl.concat(all_dfs)
    df = (
        df.with_columns(
        pl.col("Div")
        .replace(FOOTBALL_DATA_COUNTRY_MAP)
        ).rename({"Div": "League"})
    )
    df.write_parquet(save_path)
    return df
