import polars as pl
from pathlib import Path

DATA_FOLDER = Path('data')
DF_FILENAME = 'football_data.parquet'

LEAGUES = [
    "E0", "E1", "SP1", "D1", "I1", "F1", "N1", "B1", "SC1", "T1"
]

COLUMNS = [
    "Date",
    "Div",
    "HomeTeam",
    "AwayTeam",
    "FTHG",    
    "FTAG",
    "HS", 
    "AS",
    "HST", 
    "AST",
    "MaxH", 
    "MaxD", 
    "MaxA",
    "AvgH", 
    "AvgD", 
    "AvgA",
]

def load_football_data(leagues: list[str], columns: list[str], num_years: int, save_path: Path) -> pl.DataFrame:
    file_path = save_path

    if file_path.exists():
        df = pl.read_parquet(file_path)
        return df

    num_years = 6
    final_season = 25
    seasons = [
        f"{season - 1}{season}"
        for season in range(final_season, final_season - num_years, -1)
    ]

    BASE_URL = "https://www.football-data.co.uk/mmz4281/"
    all_dfs = [
        pl.read_csv(f"{BASE_URL}{season}/{league}.csv", null_values=["#"], try_parse_dates=True).select(columns)
        for league in leagues
        for season in seasons
    ]

    df = pl.concat(all_dfs)
    df.write_parquet(save_path)
    return df

print(load_football_data(LEAGUES, COLUMNS, num_years=6, save_path=DATA_FOLDER / DF_FILENAME))