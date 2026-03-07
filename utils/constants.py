from bidict import bidict

FOOTBALL_DATA_COLUMNS = [
    "Date",
    "Div",
    "HomeTeam",
    "AwayTeam",
    "FTR",
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

MAIN_EUROPEAN_LEAGUES = [
    "ENG",  # England Premier League
    "POR",  # Portugal Liga I
    "ESP",  # Spain La Liga
    "GER",  # Germany Bundesliga 1
    "ITA",  # Italy Serie A
    "FRA",  # France Ligue 1
    "NED",  # Netherlands Eredivisie
    "BEL",  # Belgium Pro League
    "TUR",  # Turkey Super Lig
    "GRE",  # Greece Super League
    "SCO"   # Scottish Prem
]

FOOTBALL_DATA_COUNTRY_MAP = bidict({
    "E0": "ENG",  
    "P1": "POR",   
    "SP1": "ESP",  
    "D1": "GER",  
    "I1": "ITA",  
    "F1": "FRA",  
    "N1": "NED",   
    "B1": "BEL",  
    "T1": "TUR",   
    "G1": "GRE",
    "SC0": "SCO" 
})

TRANSFER_DATA_COUNTRY_MAP = bidict({
    "GB1": "ENG",
    "PO1": "POR",
    "ES1": "ESP",
    "L1":  "GER",
    "IT1": "ITA",
    "FR1": "FRA",
    "NL1": "NED",
    "BE1": "BEL",
    "TR1": "TUR",
    "GR1": "GRE",
    "SC1": "SCO"
})