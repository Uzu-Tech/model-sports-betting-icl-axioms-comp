'''
Required Data Structure:

bet series (pl.DataFrame):
    - match_id: str/int (Unique identifier)
    - date: Ddatetime (for time-series growth)
    - model_probability: float
    - odds: float (Profit or loss from this specific bet)
    - outcome: str (win or loss)

Outputted Data Structure:

equity series (pl.DataFrame):
    - match_id: str/int (Unique identifier)
    - date: datetime (For time-series growth)
    - stake: float (The amount risked on this bet)
    - pnl: float (Profit or Loss from this specific bet)
    - prev_bankroll: float (The running balance BEFORE this bet settled)
    - new_bankroll: float (The running balance AFTER this bet settled)

'''

import polars as pl
from datetime import datetime

def kelly_criterion(bet_series, initial_bank_roll, fraction):
    
    bet_series = (
        bet_series
        .with_columns(
            (pl.col('odds') - 1).alias('net_odds')
        )
        .with_columns(
            (((pl.col('net_odds')*pl.col('model_probability') - (1 - pl.col('model_probability'))) * fraction)
            /pl.col('net_odds'))
            .alias('kelly_fraction')
        )
        .with_columns(
            (pl.when(pl.col('outcome') == 'win')
             .then(pl.col('kelly_fraction')*pl.col('net_odds'))
             .otherwise(-pl.col('kelly_fraction')))
             .alias('return')
        )
        .with_columns(
            ((1 + pl.col('return')).cum_prod()  * initial_bank_roll)
            .alias('new_bank_roll')
        )
        .with_columns(
            (pl.col('new_bank_roll').shift().fill_null(initial_bank_roll))
            .alias('prev_bank_roll')
        )
        .with_columns(
            (pl.col('prev_bank_roll')*pl.col('kelly_fraction'))
            .alias('stake'),
            (pl.col('new_bank_roll') - pl.col('prev_bank_roll'))
            .alias('pnl')
        )
    )

    return bet_series

        
        
