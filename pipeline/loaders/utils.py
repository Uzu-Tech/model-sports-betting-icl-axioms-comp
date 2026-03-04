import json
from pathlib import Path
import logging
from typing import Sequence
from fuzzywuzzy import process

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def save_team_mapping(mapping: dict, save_path: Path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)
    logger.info(f"--- Saved mapping to {save_path} ---")


def load_team_mapping(save_path: Path):
    with open(save_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def create_team_mapping(
    teams: Sequence[str], search_teams: Sequence[str]
):
    mapping = {}
    for name in teams:
        # Find the best match in the Elo list
        pair = process.extractOne(name, search_teams)
        if pair:
            best_match, score = pair  # type: ignore
            mapping[name] = best_match
            logger.info(f"Mapped: {name:20} -> {best_match} (Score: {score})")

    return mapping