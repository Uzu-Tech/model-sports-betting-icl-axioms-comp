from pathlib import Path

import polars as pl

import pipeline.loaders.club_elo as elo
import pipeline.loaders.transfer_data as transfer
from pipeline.loaders.football_data import load_football_data
from pipeline.loaders.utils import (
    create_team_mapping,
    load_team_mapping,
    save_team_mapping,
)
from utils.constants import (
    FOOTBALL_DATA_COLUMNS,
    MAIN_EUROPEAN_LEAGUES,
)

DATA_FOLDER = Path("data")
FOOTBALL_DATA_FILE = "football_data.parquet"
CLUB_ELO_DATA_FILE = "club_elo_data.parquet"

TRANSFER_FOLDER = "transfer_market"
CLUB_FILE = "clubs.csv"
GAMES_FILE = "games.csv"
LINEUPS_FILE = "game_lineups.csv"
VALUATIONS_FILE = "player_valuations.csv"
APPEARANCES_FILE = "appearances.csv"

FD_ELO_MAPPING_FILE = "FD_to_ClubElo_team_map.json"
FD_TM_MAPPING_FILE = "FD_to_TM_team_map.json"


def load_data(leagues: list[str], num_years: int):
    fd_df = load_football_data(
        leagues=leagues,
        columns=FOOTBALL_DATA_COLUMNS,
        num_years=num_years,
        save_path=DATA_FOLDER / FOOTBALL_DATA_FILE,
    )
    fd_teams = fd_df.get_column("HomeTeam").unique().sort().to_list()
    fd_elo_mapping, fd_tm_mapping = get_mappings(fd_teams, num_years, leagues)

    club_elo_df = elo.load_club_elo(
        teams=fd_teams,
        num_years=num_years,
        mapping=fd_elo_mapping,
        save_path=DATA_FOLDER / CLUB_ELO_DATA_FILE,
    )

    inverse_mapping = {val: key for key, val in fd_elo_mapping.items()}
    club_elo_df = club_elo_df.with_columns(pl.col("Club").replace(inverse_mapping))

    market_val_df = get_transfer_market_df(leagues, num_years)
    inverse_mapping = {val: key for key, val in fd_tm_mapping.items()}
    market_val_df = market_val_df.with_columns(
        pl.col("home_team").replace(inverse_mapping),
        pl.col("away_team").replace(inverse_mapping),
    )

    return add_elo_to_df(
        add_market_data_to_df(fd_df, market_val_df), club_elo_df
    ).drop_nulls()


def get_mappings(fd_teams: list[str], num_years: int, leagues: list[str]):
    mapping_file = DATA_FOLDER / FD_ELO_MAPPING_FILE
    if mapping_file.exists():
        fd_elo_mapping = load_team_mapping(mapping_file)
    else:
        search_teams = elo.get_club_elo_names(num_yrs=num_years, leagues=leagues)
        fd_elo_mapping = create_team_mapping(
            teams=fd_teams,
            search_teams=search_teams,
        )
        fd_elo_mapping = elo.fix_incorrect_mappings(fd_elo_mapping)
        save_team_mapping(fd_elo_mapping, mapping_file)

    mapping_file = DATA_FOLDER / FD_TM_MAPPING_FILE
    if mapping_file.exists():
        fd_tm_mapping = load_team_mapping(mapping_file)
    else:
        club_df = transfer.get_clubs(
            leagues, clubs_path=DATA_FOLDER / TRANSFER_FOLDER / CLUB_FILE
        )
        search_teams = club_df.get_column("club_code").unique().sort().to_list()
        fd_tm_mapping = create_team_mapping(
            teams=fd_teams,
            search_teams=search_teams,
        )
        fd_tm_mapping = transfer.fix_incorrect_mappings(fd_tm_mapping)
        save_team_mapping(fd_tm_mapping, mapping_file)

    return fd_elo_mapping, fd_tm_mapping


def add_elo_to_df(df: pl.DataFrame, club_elo_df: pl.DataFrame):
    df = df.sort("Date")
    club_elo_df = club_elo_df.sort("From")

    combined_df = df.join_asof(
        club_elo_df.select("Club", "From", "Elo"),
        left_on="Date",
        right_on="From",
        by_left="HomeTeam",
        by_right="Club",
        strategy="backward",
    ).rename({"Elo": "home_team_elo"})

    combined_df = combined_df.join_asof(
        club_elo_df.select("Club", "From", "Elo"),
        left_on="Date",
        right_on="From",
        by_left="AwayTeam",
        by_right="Club",
        strategy="backward",
    ).rename({"Elo": "away_team_elo"})

    return combined_df.drop("From", "From_right")


def get_transfer_market_df(leagues: list[str], num_years: int):
    club_df = transfer.get_clubs(
        MAIN_EUROPEAN_LEAGUES, clubs_path=DATA_FOLDER / TRANSFER_FOLDER / CLUB_FILE
    )

    game_df = transfer.get_games(
        leagues,
        club_df,
        num_years=num_years,
        games_path=DATA_FOLDER / TRANSFER_FOLDER / GAMES_FILE,
    )

    teams_values = transfer.get_team_transfer_values(
        game_df,
        lineups_path=DATA_FOLDER / TRANSFER_FOLDER / LINEUPS_FILE,
        valuations_path=DATA_FOLDER / TRANSFER_FOLDER / VALUATIONS_FILE,
        appearances_path=DATA_FOLDER / TRANSFER_FOLDER / APPEARANCES_FILE,
    )

    market_val_df = transfer.get_game_market_values(game_df, teams_values).sort("date")
    return market_val_df


def add_market_data_to_df(df: pl.DataFrame, market_val_df: pl.DataFrame):
    df.sort("Date")
    market_val_df.sort("date")

    combined_df = df.join(
        market_val_df.select(
            "date",
            "home_team",
            "home_mean_market_val",
            "away_team",
            "away_mean_market_val",
        ),
        left_on=["Date", "HomeTeam", "AwayTeam"],
        right_on=["date", "home_team", "away_team"],
        how="left",
    ).sort("Date")

    return combined_df.with_columns(
        pl.col("home_mean_market_val").fill_null(strategy="forward").over("HomeTeam"),
        pl.col("away_mean_market_val").fill_null(strategy="forward").over("AwayTeam"),
    )
