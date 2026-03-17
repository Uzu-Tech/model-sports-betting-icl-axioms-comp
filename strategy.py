# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # KZA: Building Statistical Models for Football Betting

# %% [markdown]
# ## Imports

# %%
import polars as pl
import importlib

# %% [markdown]
# ## Introduction
#
# ### Objective
# Our primary goal is to generate a profitable trading model betting on European Football matches by identifying discrepancies between our calculated "True Probability" and the odds or "Implied Probabilities" offered by bookmakers.
#
# ### The Pipeline
# Our approach follows a structured data science lifecycle:
#
# - **Research**: Investigate if there are market inefficiencies and look at potiental models.
# - **Data Collection**: Sourcing online match and market data.
# - **Exploratory Data Analysis**: Check if our data is valid and look for insights 
# - **Feature Engineering**: Transforming raw stats into predictive signals.
# - **Modelling**: Training and validating via time-series cross-validation.
# - **Signal Generation**: Converting probabilities into bets.
# - **Risk Management**: Deciding how much to stake on each bet.
# - **Evaluation**: Assessing performance of our strategy the Closing Line.
# - **Launch**: Deployment and monitoring.

# %% [markdown]
# ## Research
#
# ### Evidence of Market Inefficiency
#
# The Efficient Market Hypothesis suggests that in a highly liquid market, like European Football, the odds offered by bookmakers should reflect all available information, rendering it incredibly difficult to achieve a consistent edge. However when lopsided betting occurs, driven by fan bias toward "Big Six" clubs (e.g., Real Madrid, Manchester United) or "longshot" underdogs, bookmakers often adjust their odds away from the "True Probability" to encourage betting on the other side and reduce their risk.
#
# We hypothesize that value exists in the "drift" between the market price and statistical reality, when the odds drift too far from the "True Probabilities". Some studies that support this idea are:
#
# **TODO** Add papers
#
# ### Model Selection
#
# To capture the "drift" between market prices and statistical reality, we chose a diverse set of three distinct modeling approaches.
#
# **Simplest Model: Ordered Logistic Regression**
#
# **Linear Baseline: Multimodal Logistic Regression**
#
# **Non-Linear Logic: Gradient Boosted Trees (XGBoost)**
#
# **TODO** Add papers
#
# ### Predicting Bookmaker Errors Directly
#
# **TODO** Add papers

# %% [markdown]
# ## Data Collection
#
# To build a robust representation of European football, we integrated three distinct data dimensions: performance stats, historical rankings, and financial valuation.
#
# ### Data Sources
# We aggregated our dataset from three primary pillars found on Kaggle and specialized football databases:
#
# - Football-Data.co.uk: Our primary source for historical match results, betting odds, and match-level statistics (shots, corners, fouls).
# - ClubELO: Used to integrate long-term team strength ratings. ELO provides a "memory" of a team's quality that persists beyond a single result.
# - Transfermarkt (Market Values): We pulled squad valuations to act as a proxy for raw talent and depth. This helps the model distinguish between a "lucky" mid-table team and a powerhouse underperforming its budget.
#
# ### League Selection: The "Main 10"
# To ensure data consistency and high liquidity for our betting strategy, we focused on the top 10 European leagues. This selection provides a massive sample size of roughly 3,800 matches per season, ensuring our model has enough "experience" to learn league-specific nuances.
#
# Leagues Included:
# - Premier League (England)
# - La Liga (Spain), 
# - Bundesliga (Germany)
# - Serie A (Italy)
# - Ligue 1 (France) 
# - Primeira Liga (Portugal)
# - Super League (Greece)
# - Süper Lig (Turkey)
# - Premiership (Scotland)
# - Jupiler Pro League (Belgium)
#
# ### Time Frame
# We opted for a 7-year historical window. This timeframe represents a good middle ground for sports modeling:
#
# Relevance: Data older than 7 years may reflect a different era of tactical play (pre-heavy pressing/VAR), which can "pollute" modern predictions.
# Volume: Six years provides over 22,000 matches, offering a great sample size to train and test our models without immediate overfitting.
#
# ### Implementation
# To ensure reproducibility and clean execution, the entire ingestion pipeline—including cleaning, joining, and handling missing ELO/Value data—is encapsulated in a custom modular function. This pipeline handles the merging of disparate CSV files into a single, analysis-ready DataFrame.
#
# Note: The full implementation of our data pipeline, including the load_data() logic, is available in our GitHub repository linked in the appendix.
#

# %%
import logging
from pipeline.data_loader import load_data
from utils.constants import MAIN_EUROPEAN_LEAGUES


logging.getLogger("pipeline").setLevel(logging.ERROR)

NUM_YEARS = 7
df = load_data(leagues=MAIN_EUROPEAN_LEAGUES, num_years=NUM_YEARS)

# %%
# Remove non numerical columns
df.drop("Datetime", "League", "HomeTeam", "AwayTeam", "FTR").describe()

# %% [markdown]
#

# %% [markdown]
# ## Feature Engineering
#
# ### Creation

# %%
SEASON_START_MONTH = 7
df = df.with_columns(
    Season = (
        pl.when(pl.col("Datetime").dt.month() >= SEASON_START_MONTH)
        .then(
            pl.concat_str([
                pl.col("Datetime").dt.year(),
                pl.lit("/"),
                pl.col("Datetime").dt.year() + 1
            ])
        )
        .otherwise(
            pl.concat_str([
                pl.col("Datetime").dt.year() - 1,
                pl.lit("/"),
                pl.col("Datetime").dt.year()
            ])
        )
    )
)

df = df.with_row_index(name="match_id")

# %%
df = df.with_columns(
    (pl.col("home_team_elo") - pl.col("away_team_elo")).alias("elo_diff"),
    pl.col("home_mean_market_val").log().alias("HMV_log_val"),
    pl.col("away_mean_market_val").log().alias("AMV_log_val")
)

df = df.with_columns(
    (pl.col("HMV_log_val") - pl.col("AMV_log_val")).alias("MV_log_diff"),
    (
        pl.col("home_team_elo") / pl.col("HMV_log_val")
        - pl.col("away_team_elo") / pl.col("AMV_log_val")
    ).alias("elo_per_value_diff")
).drop("HMV_log_val", "AMV_log_val")

df.select(["Datetime", "elo_diff", "MV_log_diff", "elo_per_value_diff"])

# %%
from pipeline import feature_engineering
importlib.reload(feature_engineering)
from pipeline.feature_engineering import add_custom_ewm_features

SHORT_WINDOW = 4
MEDIUM_WINDOW = 15
LONG_WINDOW = 30
windows = [SHORT_WINDOW, MEDIUM_WINDOW, LONG_WINDOW]

# Calculate Global Priors
# SOT Ratio (Total SOT / Total Shots)
stats = df.select([
    (pl.col("HST").sum() + pl.col("AST").sum()).alias("total_sot"),
    (pl.col("HS").sum() + pl.col("AS").sum()).alias("total_shots"),
    (pl.col("FTHG").sum() + pl.col("FTAG").sum()).alias("total_goals")
])
prior_sot = stats.get_column("total_sot")[0] / stats.get_column("total_shots")[0]
prior_conv = stats.get_column("total_goals")[0] / stats.get_column("total_sot")[0]
prior_save = 1 - prior_conv  # Simplified: Saves are just non-goals on target

# Bayesian Smoothing Constants
ALPHA_SHOTS = 5
ALPHA_GOALS = 5

ewm_features = {
    "goals": {
        "home": "FTHG",
        "away": "FTAG"
    },
    # Shots on Target Ratio (Smoothed)
    "sot_ratio": {
        "home": (pl.col("HST") + (ALPHA_SHOTS * prior_sot)) / (pl.col("HS") + ALPHA_SHOTS),
        "away": (pl.col("AST") + (ALPHA_SHOTS * prior_sot)) / (pl.col("AS") + ALPHA_SHOTS)
    },
    # SOT Against Ratio (Smoothed)
    "sot_a_ratio": {
        "home": (pl.col("AST") + (ALPHA_SHOTS * prior_sot)) / (pl.col("AS") + ALPHA_SHOTS),
        "away": (pl.col("HST") + (ALPHA_SHOTS * prior_sot)) / (pl.col("HS") + ALPHA_SHOTS)
    },
    # Conversion Rate (Smoothed)
    "conversion_rate": {
        "home": (pl.col("FTHG") + (ALPHA_GOALS * prior_conv)) / (pl.col("HST") + ALPHA_GOALS),
        "away": (pl.col("FTAG") + (ALPHA_GOALS * prior_conv)) / (pl.col("AST") + ALPHA_GOALS)
    },
    # Save Rate (Smoothed)
    "save_rate": {
        "home": ((pl.col("AST") - pl.col("FTAG")) + (ALPHA_GOALS * prior_save)) / (pl.col("AST") + ALPHA_GOALS),
        "away": ((pl.col("HST") - pl.col("FTHG")) + (ALPHA_GOALS * prior_save)) / (pl.col("HST") + ALPHA_GOALS)
    },
    "elo_adv": {
        "home": pl.col("elo_diff"),
        "away": -pl.col("elo_diff")
    }
}

df, latest_stats = add_custom_ewm_features(df, windows, ewm_features)

df = df.drop(
    f"home_elo_adv_ewm_{MEDIUM_WINDOW}", f"home_elo_adv_ewm_{LONG_WINDOW}",
    f"away_elo_adv_ewm_{MEDIUM_WINDOW}", f"away_elo_adv_ewm_{LONG_WINDOW}"
).drop_nulls()

latest_stats.filter(pl.col("team") == "Man City")

# %%
momentum_features = ["goals", "sot_ratio", "sot_a_ratio", "conversion_rate", "save_rate"]

for feature in momentum_features:
    for side in ("home", "away"):
        df = df.with_columns(
            (pl.col(f"{side}_{feature}_ewm_{SHORT_WINDOW}")
            - pl.col(f"{side}_{feature}_ewm_{LONG_WINDOW}"))
            .alias(f"{side}_{feature}_momentum")
        )

df

# %%
for side in ("home", "away"):
    df = df.with_columns(
        (
            pl.col(f"{side}_conversion_rate_ewm_{SHORT_WINDOW}") 
            + pl.col(f"{side}_save_rate_ewm_{SHORT_WINDOW}")
        ).alias(f"{side}_luck_factor")
    )

# %%
import polars.selectors as cs
# machine learning featurs
training_df = df.filter(pl.col("Season") != "2019/2020")

ml_features = training_df.drop(
    "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA"
)
possion_features = training_df.select(["match_id", "Datetime", "Season", "HomeTeam", "AwayTeam"])

ml_results = training_df.select(["match_id", "Season", "Datetime", "FTR"])
possion_results = training_df.select(["match_id", "Datetime", "FTHG", "FTAG"])

print("Machine Learning Features:")
for col in ml_features.columns:
    if col not in ("match_id", "Datetime"):
        print(f"- {col}")

print("\nPossion Features:")
for col in ml_features.columns:
    print(f"- {col}")

# %%
import polars as pl
import plotly.express as px

def plot_corr_matrix(features: pl.DataFrame, size):
    numeric_df = features.select(cs.numeric())

    corr_matrix = numeric_df.corr()

    # 3. Create the Plotly Heatmap
    fig = px.imshow(
        corr_matrix.to_numpy(),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu_r', # Red-Blue scale (standard for corr)
        zmin=-1, zmax=1,                 # Correlation is always -1 to 1
        title="Feature Correlation Matrix",
        labels=dict(color="Pearson Corr"),
        aspect="auto"
    )

    fig.update_layout(
        width=size[0],
        height=size[1],
        xaxis_tickangle=-45
    )
    fig.show()

plot_corr_matrix(ml_features, size=(1200, 1200))

# %%

ml_features = (
    ml_features
    .rename({
        f"home_elo_adv_ewm_{SHORT_WINDOW}": "home_elo_adv",
        f"away_elo_adv_ewm_{SHORT_WINDOW}": "away_elo_adv"
    })
)

ml_features = ml_features.drop([
    feature 
    for feature in ml_features.columns 
    if (f"ewm_{MEDIUM_WINDOW}" in feature) or (f"ewm_{SHORT_WINDOW}" in feature)
])

for side in ("home", "away"):
    ml_features = ml_features.drop(
        f"{side}_luck_factor", 
        f"{side}_mean_market_val",
        f"{side}_team_elo"
    )

ml_features = ml_features.drop("MV_log_diff")
plot_corr_matrix(ml_features, size=(900, 900))

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor
import polars as pl
import pandas as pd

def calculate_vif(features: pl.DataFrame):
    numeric_df = features.select(cs.numeric()).drop("match_id")
    # 2. Convert to Pandas (statsmodels requires it)
    X = numeric_df.to_pandas()
    
    # 3. Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) 
        for i in range(len(X.columns))
    ]
    
    return vif_data.sort_values("VIF", ascending=False)

# Run it
vif_results = pl.DataFrame(calculate_vif(ml_features))
for row in vif_results.iter_rows(named=True):
    print(f"Feature: {row['feature']}, VIF: {row['VIF']}")

# %%
residuals = training_df.select(
    "match_id", "Datetime", "Season", "FTR", "AvgH", "AvgA", "AvgD"
)

residuals = residuals.with_columns(
    AvgH= 1 / pl.col("AvgH"),
    AvgA= 1 / pl.col("AvgA"),
    AvgD= 1 / pl.col("AvgD")
)

residuals = residuals.with_columns(implied_probs_sum=pl.col("AvgH") + pl.col("AvgD") + pl.col("AvgA"))
residuals = residuals.with_columns(
    pl.col(["AvgH", "AvgD", "AvgA"]) / pl.col("implied_probs_sum")
).rename({
    "AvgH": "home_prob",
    "AvgA": "away_prob",
    "AvgD": "draw_prob"
}).drop("implied_probs_sum")

residuals = residuals.to_dummies("FTR")
residuals = residuals.with_columns(
    home_residual=pl.col("home_prob") - pl.col("FTR_H"),
    away_residual=pl.col("away_prob") - pl.col("FTR_A"),
    draw_residual=pl.col("draw_prob") - pl.col("FTR_D"),
).drop(cs.matches(r"_prob$") | cs.matches("^FTR"))
residuals

# %%
from statsmodels.multivariate.manova import MANOVA

manov_df = residuals.join(
    ml_features,
    on=['match_id', 'Datetime'],
    how='left'
)

manov_df = manov_df.select(cs.numeric()).drop("match_id")

feature_list = ml_features.select(cs.numeric()).drop("match_id").columns
# 2. Join them with a plus sign
features_formula = " + ".join(feature_list)

ma = MANOVA.from_formula(f'home_residual + away_residual ~ {features_formula}', data=manov_df.to_pandas())
print(ma.mv_test())

# %%
import polars as pl
from scipy import stats


manov_df = ml_results.join(
    ml_features,
    on=['match_id', 'Datetime'],
    how='left'
)

features = manov_df.select(cs.numeric()).drop("match_id").columns
target_col = "FTR"

results = {}

for feature in features:
    # 2. Group data by category and collect the numerical values into lists
    groups = (
        manov_df.group_by(target_col)
        .agg(pl.col(feature))
        .get_column(feature)
        .to_list()
    )
    
    # 3. Perform One-Way ANOVA (*groups unpacks the list of arrays)
    f_stat, p_val = stats.f_oneway(*groups)
    
    results[feature] = {"F-Statistic": f_stat, "P-Value": p_val}

# 4. View results as a summary table
summary_df = pl.DataFrame([
    {"feature": k, "f_stat": v["F-Statistic"], "p_value": v["P-Value"]} 
    for k, v in results.items()
]).sort("p_value")

for row in summary_df.iter_rows(named=True):
    print(row["feature"], round(row["f_stat"], 2), round(row['p_value'], 2))

# %%
import polars as pl

manov_df = residuals.join(
    ml_features,
    on=['match_id', 'Datetime'],
    how='left'
)

# 1. Convert to pandas for Plotly compatibility (zero-copy in Polars)
plot_df = manov_df.to_pandas()

feature = "away_sot_ratio_momentum"
# 2. Create the scatter plot with trendlines
fig = px.scatter(
    plot_df, 
    x=feature, 
    y="home_residual", 
    trendline="ols",  # Adds the linear regression lines
    title="Interactive Residuals vs. SOT Ratio Momentum",
    template="plotly_white",
)

# 3. Add a horizontal line at Zero (The 'Perfect Prediction' line)
fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.7)

# 4. Improve layout
fig.update_layout(
    xaxis_title=feature.replace("_", " ").capitalize(),
    yaxis_title="Residuals (Actual - Predicted)",
    legend_title="Match Result"
)

fig.show()

# %%
import polars as pl
from sklearn.feature_selection import mutual_info_regression

residual_cols = ["home_residual", "away_residual", "draw_residual"]
features = ml_features.select(cs.numeric()).drop("match_id").columns
X = ml_features.select(features).to_pandas()

mi_data = []
for feat in features:
    # Prepare X (2D array required by sklearn)
    X = ml_features.select(feat).to_pandas()
    
    row = {"feature": feat}
    for res in residual_cols:
        y = residuals.get_column(res).to_pandas()
        # Calculate MI (returns an array, we take the first element [0])
        score = mutual_info_regression(X, y, random_state=42)[0]
        row[res] = score
        
    mi_data.append(row)

# 4. Convert to Polars for easy viewing/sorting
mi_results_df = pl.DataFrame(mi_data)

# %%
for row in mi_results_df.sort('home_residual', descending=True).iter_rows(named=True):
    print(
        row["feature"], 
        round(row["home_residual"], 3), 
        round(row['away_residual'], 3), 
        round(row['draw_residual'], 3)
    )

# %%
residual_selected_columns = [
    "elo_per_value_diff", "away_sot_ratio_momentum",
]

for side in ("home", "away"):
    residual_selected_columns.extend(
        [
           # f"{side}_elo_adv",
        ]
    )

outcome_selected_columns = [
    "elo_diff", "elo_per_value_diff"
]

for side in ("home", "away"):
    outcome_selected_columns.extend(
        [
            f"{side}_elo_adv",
        ]
    )

ml_residuals = (
    ml_features
    .select("match_id", "Datetime", "Season", "League", *residual_selected_columns,
            conversion_diff=(
                pl.col("home_conversion_rate_ewm_30") - pl.col("away_conversion_rate_ewm_30")
            ))
)
ml_outcomes = ml_features.select("match_id", "Datetime", "Season", *outcome_selected_columns, 
                                 goals_ewm_30_diff=(
                                     pl.col("home_goals_ewm_30") - pl.col("away_goals_ewm_30")
                                 ))

# %%
vif_results = pl.DataFrame(calculate_vif(ml_residuals))
for row in vif_results.iter_rows(named=True):
    print(f"Feature: {row['feature']}, VIF: {row['VIF']}")

# %%
probs = training_df.select(
    "match_id", "Datetime", "Season", "FTR", "AvgH", "AvgA", "AvgD"
)

probs = probs.with_columns(
    AvgH= 1 / pl.col("AvgH"),
    AvgA= 1 / pl.col("AvgA"),
    AvgD= 1 / pl.col("AvgD")
)

probs = probs.with_columns(implied_probs_sum=pl.col("AvgH") + pl.col("AvgD") + pl.col("AvgA"))
probs = probs.with_columns(
    pl.col(["AvgH", "AvgD", "AvgA"]) / pl.col("implied_probs_sum")
).rename({
    "AvgH": "home_prob",
    "AvgA": "away_prob",
    "AvgD": "draw_prob"
}).drop("implied_probs_sum", "FTR")

# %%
from pipeline.models.ordinal import OrdinalBettingWrapper
from pipeline.models.xgboost import MultiClassXGBoostWrapper
from pipeline import model
importlib.reload(model)
from pipeline.model import SequentialSeasonValidator
from pipeline.models import bayesian, multimodal
importlib.reload(bayesian)
importlib.reload(multimodal)


from pipeline.models.multimodal import MultiModalLogitWrapper
from pipeline.models.bayesian import BayesianLassoLogitWrapper

outcomes = training_df.select(
    "match_id", "Datetime", "Season", 
    outcome=pl.col("FTR").replace({"H": 2, "D": 1, "A": 0}).cast(pl.Int64)
)

model_df = outcomes.join(
    probs,
    on=("match_id", "Datetime", "Season"),
    how="left"
)

model_outcomes_df = model_df.join(
    ml_outcomes,
    on=("match_id", "Datetime", "Season"),
    how="left"
)

model_residual_df = model_df.join(
    ml_residuals,
    on=("match_id", "Datetime", "Season"),
    how="left"
)

params = {
    # The Essentials
    'max_depth': 3,             # Shallow trees to avoid overfitting noise
    'learning_rate': 0.1,      # High enough to learn, low enough to be stable
    
    # Regularization (Crucial for 2020 data)
    'min_child_weight': 5,      # Won't make a rule unless it hits ~5 matches
    'lambda': 1.5,              # L2 regularization (keeps weights small)
    'subsample': 0.8,           # Use 80% of matches per tree for variety
    
    # Technicals
    'tree_method': 'hist',      # Fast training
    'nthread': -1               # Use all CPU cores
}

outcomes_features = ml_outcomes.select(cs.numeric()).drop("match_id").columns
residuals_features = ml_residuals.select(cs.numeric()).drop("match_id").columns

outcomes_models = [
    OrdinalBettingWrapper(predict_residuals=False),
    MultiClassXGBoostWrapper(predict_residuals=False, params=params),
]

residual_models = [
    #OrdinalBettingWrapper(predict_residuals=True),
    #MultiClassXGBoostWrapper(predict_residuals=True, params=params),
    MultiModalLogitWrapper(lasso=False),
    MultiModalLogitWrapper(lasso=True),
    BayesianLassoLogitWrapper()
]

for res_model in residual_models:
    validator = SequentialSeasonValidator(
        model_wrapper=res_model,
        features=residuals_features,
        start_season="2020/2021",
        num_seasons=5,
        target_col="outcome"
    )


    print("Residual Model")
    print(validator.run(model_residual_df))

# %%
final_model = residual_models[1]

validator = SequentialSeasonValidator(
    model_wrapper=final_model,
    features=residuals_features,
    start_season="2020/2021",
    num_seasons=4,
    target_col="outcome"
)

validator.fit_all_training_data(model_residual_df)
final_model.model_res.summary()

# %%
USE_MODEL = True

test_seasons = [24, 25]
test_seasons = [f"20{season}/20{season + 1}" for season in test_seasons]

if USE_MODEL:
    # Your current model prediction logic
    preds = final_model.predict(
        model_residual_df.filter(pl.col("Season").is_in(test_seasons)).select(residuals_features), 
        market_probs=probs.filter(pl.col("Season").is_in(test_seasons))
    )
    val_probs = pl.DataFrame(preds, schema=["m_away_probs", "m_draw_probs", "m_home_probs"])
else:
    # Use the bookie's own probs (extracted from your 'probs' dataframe)
    # We rename them to match the schema used in the join
    val_probs = (
        probs.filter(pl.col("Season").is_in(test_seasons))
        .select([
            pl.col("away_prob").alias("m_away_probs"),
            pl.col("draw_prob").alias("m_draw_probs"),
            pl.col("home_prob").alias("m_home_probs"),
            pl.col("match_id")
        ])
    )

# Ensure match_id is present if it wasn't added in the 'else' block
if "match_id" not in val_probs.columns:
    val_probs = val_probs.with_columns(
        match_id = model_residual_df.filter(pl.col("Season").is_in(test_seasons)).get_column("match_id")
    )

val_probs

# %%
ev_series = (
    outcomes.filter(pl.col("Season").is_in(test_seasons))
    .join(
        val_probs,
        on='match_id'
    )
    .join(
        df.filter(pl.col("Season").is_in(test_seasons)).select("match_id", "MaxH", "MaxA", "MaxD"),
        on='match_id'
    )
    .with_columns(
        (pl.col("m_away_probs") * (pl.col("MaxA") - 1) - (1 - pl.col("m_away_probs"))).alias("EV_A"),
        (pl.col("m_home_probs") * (pl.col("MaxH") - 1) - (1 - pl.col("m_home_probs"))).alias("EV_H"),
        (pl.col("m_draw_probs") * (pl.col("MaxD") - 1) - (1 - pl.col("m_draw_probs"))).alias("EV_D")
    )
)

THRESHOLD = 0
MAX_THRESHOLD = 100

ev_series = ev_series.filter(
    pl.any_horizontal(
        pl.col("EV_A") >= THRESHOLD,
        pl.col("EV_H") >= THRESHOLD,
        pl.col("EV_D") >= THRESHOLD
    )
)

bet_series = (
    ev_series
    .with_columns(
        EV=pl.max_horizontal("EV_A", "EV_H", "EV_D")
    )
    .with_columns(
        odds=(pl.when(pl.col("EV_A") == pl.col("EV"))
            .then(pl.col("MaxA"))
            .when(pl.col("EV_H") == pl.col("EV"))
            .then(pl.col("MaxH"))
            .otherwise(pl.col("MaxD"))
        ),
        model_probability=(pl.when(pl.col("EV_A") == pl.col("EV"))
            .then(pl.col("m_away_probs"))
            .when(pl.col("EV_H") == pl.col("EV"))
            .then(pl.col("m_home_probs"))
            .otherwise(pl.col("m_draw_probs"))
        )
    )
    .with_columns(
        outcome = pl.col("outcome")
                .cast(pl.String)  # Move this here
                .replace({"0": "A", "1": "D", "2": "H"}) # Map strings to strings
    )
    .with_columns(
        model_predictions=(pl.when(pl.col("EV_A") == pl.col("EV"))
            .then(pl.lit("A"))
            .when(pl.col("EV_H") == pl.col("EV"))
            .then(pl.lit("H"))
            .otherwise(pl.lit("D"))
        )
    )
    .with_columns(
        outcome=pl.when(pl.col("outcome") == pl.col("model_predictions"))
        .then(pl.lit("win"))
        .otherwise(pl.lit("loss"))
    )
)

bet_series = bet_series.select("match_id", "Datetime", "Season", "outcome", "odds", "model_probability")
# Outlier
bet_series = bet_series.filter(pl.col("match_id") != 22149)
bet_series

# %%
from pipeline.risk import kelly_criterion, flat_bet, fixed_fraction
from pipeline.evaluation import evaluate_returns, get_evaluation_table, apply_clean_theme
from pipeline.metrics.performance import (
    median_PNL, mean_PNL, total_PNL, number_of_bets, win_rate, loss_rate, compound_return
)
from pipeline.metrics.risk_adjusted import sharpe_ratio, calmer_ratio, sortino_ratio
from pipeline.metrics.drawdown import (
    max_drawdown, max_time_underwater, average_drawdown, avg_time_underwater
)
from pipeline.metrics.volatility import volatility, downside_volatility, skewness, kurtosis, best_bet, worst_bet

KELLY_FRACTION = 0.2

metrics = {
    "Performance": [mean_PNL, median_PNL, total_PNL, number_of_bets, win_rate, loss_rate, compound_return],
    "Distribution": [volatility, downside_volatility, skewness, kurtosis, best_bet, worst_bet],
    "Risk Adjusted Returns": [sharpe_ratio, calmer_ratio, sortino_ratio],
    "Drawdown": [max_drawdown, max_time_underwater, average_drawdown, avg_time_underwater]
}

equity_series = kelly_criterion(bet_series, initial_bank_roll=500, fraction=KELLY_FRACTION)
px.line(
    equity_series,
    y="new_bank_roll",
    x="match_id",
)

# %%
match_id = equity_series.filter(pl.col("pnl") == pl.col("pnl").max()).get_column("match_id").item()
df.filter(pl.col("match_id") == match_id)

# %%
equity_series = equity_series.rename({"prev_bank_roll": "prev_bankroll", "new_bank_roll": "new_bankroll"})
returns = evaluate_returns(equity_series, metrics)
apply_clean_theme(get_evaluation_table(returns, title="Results", subtitle=""))


# %%
HOME_TEAM = "Fenerbahce"
AWAY_TEAM = "Gaziantep"

AVGH = 1.25
AVGD = 1 + (21/4)
AVGA = 1 + (11/1)

MAXH = 1.18
MAXA = 14.1
MAXD = 6.2

teams = df.select("match_id", "Datetime", "HomeTeam", "AwayTeam").unpivot(
    index=["match_id", "Datetime"], on=["HomeTeam", "AwayTeam"]
).sort("Datetime")

home_team_id = teams.filter(pl.col("value") == HOME_TEAM).tail(1).get_column("match_id").item()
away_team_id = teams.filter(pl.col("value") == AWAY_TEAM).tail(1).get_column("match_id").item()

home_df = df.filter(pl.col("match_id") == home_team_id)
away_df = df.filter(pl.col("match_id") == away_team_id)

home_features = latest_stats.filter(
    pl.col("match_id") == home_team_id,
    pl.col("team") == HOME_TEAM
)

away_features = latest_stats.filter(
    pl.col("match_id") == away_team_id,
    pl.col("team") == AWAY_TEAM
)

away_sot_ratio_momentum = away_features.select(
    pl.col(f"sot_ratio_ewm_{SHORT_WINDOW}") - pl.col(f"sot_ratio_ewm_{LONG_WINDOW}")
).item()

conversion_diff = (
    home_features.get_column("conversion_rate_ewm_30").item() 
    - away_features.get_column("conversion_rate_ewm_30").item() 
)

elo_per_value_diff = (
    home_df.select(pl.col("home_team_elo_after") / pl.col("home_mean_market_val").log()).item()
    - away_df.select(pl.col("away_team_elo_after") / pl.col("away_mean_market_val").log()).item()
)

pred_features = pl.DataFrame(
    {
        "elo_per_value_diff": (elo_per_value_diff, ),
        "conversion_diff": (conversion_diff, ),
        "away_sot_ratio_momentum": (away_sot_ratio_momentum, ),
        "AvgH": AVGH,
        "AvgD": AVGD,
        "AvgA": AVGA,
        "MaxH": MAXH,
        "MaxD": MAXD,
        "MaxA": MAXA,
    }
)

probs = pred_features.select(
    AvgH= 1 / pl.col("AvgH"),
    AvgA= 1 / pl.col("AvgA"),
    AvgD= 1 / pl.col("AvgD")
)

probs = probs.with_columns(implied_probs_sum=pl.col("AvgH") + pl.col("AvgD") + pl.col("AvgA"))
probs = probs.with_columns(
    pl.col(["AvgH", "AvgD", "AvgA"]) / pl.col("implied_probs_sum")
).rename({
    "AvgH": "home_prob",
    "AvgA": "away_prob",
    "AvgD": "draw_prob"
}).drop("implied_probs_sum")

pred_features

# %%
preds = final_model.predict(
    pred_features.select("elo_per_value_diff", "away_sot_ratio_momentum", "conversion_diff"), 
    market_probs=probs
)
val_probs = pl.DataFrame(preds, schema=["m_away_probs", "m_draw_probs", "m_home_probs"])
pred_df = pl.concat([pred_features, probs, val_probs], how="horizontal")
pred_df

# %%
ev_series = (
    pred_df
    .with_columns(
        (pl.col("m_away_probs") * (pl.col("MaxA") - 1) - (1 - pl.col("m_away_probs"))).alias("EV_A"),
        (pl.col("m_home_probs") * (pl.col("MaxH") - 1) - (1 - pl.col("m_home_probs"))).alias("EV_H"),
        (pl.col("m_draw_probs") * (pl.col("MaxD") - 1) - (1 - pl.col("m_draw_probs"))).alias("EV_D")
    )
)

ev_series

# %%
ev_series = ev_series.filter(
    pl.any_horizontal(
        pl.col("EV_A") >= THRESHOLD,
        pl.col("EV_H") >= THRESHOLD,
        pl.col("EV_D") >= THRESHOLD
    )
)

bet_series = (
    ev_series
    .with_columns(
        EV=pl.max_horizontal("EV_A", "EV_H", "EV_D")
    )
    .with_columns(
        odds=(pl.when(pl.col("EV_A") == pl.col("EV"))
            .then(pl.col("MaxA"))
            .when(pl.col("EV_H") == pl.col("EV"))
            .then(pl.col("MaxH"))
            .otherwise(pl.col("MaxD"))
        ),
        model_probability=(pl.when(pl.col("EV_A") == pl.col("EV"))
            .then(pl.col("m_away_probs"))
            .when(pl.col("EV_H") == pl.col("EV"))
            .then(pl.col("m_home_probs"))
            .otherwise(pl.col("m_draw_probs"))
        )
    )
    .with_columns(
        model_predictions=(pl.when(pl.col("EV_A") == pl.col("EV"))
            .then(pl.lit("A"))
            .when(pl.col("EV_H") == pl.col("EV"))
            .then(pl.lit("H"))
            .otherwise(pl.lit("D"))
        )
    ).select("odds", "model_probability", "model_predictions", "EV")
)

bet_series

# %%
bet_series = (
    bet_series
    .with_columns(
        (pl.col('odds') - 1).alias('net_odds')
    )
    .with_columns(
        kelly_fraction=(
            (((pl.col('net_odds')*pl.col('model_probability') - (1 - pl.col('model_probability')))*KELLY_FRACTION)
            /pl.col('net_odds'))
        )
    )
)

bet_series
