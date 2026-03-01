import json
from pathlib import Path

import polars as pl
from fuzzywuzzy import process

from utils.constants import FOOTBALL_DATA_COUNTRY_MAP, MAIN_EUROPEAN_LEAGUES


def load_football_data(
    leagues: list[str], columns: list[str], num_years: int, save_path: Path
) -> pl.DataFrame:
    file_path = save_path

    if file_path.exists():
        df = pl.read_parquet(file_path)
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


BASE_URL = "http://api.clubelo.com"


def load_club_elo(leagues: list[str], num_years: int, save_path: Path): ...


def get_club_elo_names(num_yrs: int, leagues: list[str]):
    teams = set()
    for league in leagues:
        for year in range(2025, 2025 - num_yrs, -1):
            date_str = f"{year}-12-31"
            url = f"{BASE_URL}/{date_str}"
            df = pl.read_csv(url, null_values=["None"])
            league_teams = df.filter(
                (pl.col("Country") == league) & (pl.col("Level") == 1)
            )
            teams = teams.union(league_teams.get_column("Club").to_list())

    return teams


def create_team_mapping(
    football_data_names: list[str], elo_names: set[str], save_path: Path
):
    print(f"--- Num Teams: {len(football_data_names)} ---")
    mapping = {}
    for name in football_data_names:
        # Find the best match in the Elo list
        pair = process.extractOne(name, elo_names, score_cutoff=80)
        if pair:
            best_match, score = pair  # type: ignore
            mapping[name] = best_match
            print(f"Mapped: {name:20} -> {best_match} (Score: {score})")

    print(f"--- Num teams mapped: {len(mapping)} ---")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)

    print(f"--- Saved mapping to {save_path} ---")
    print("Mapping:")
    print(mapping)
    return mapping
