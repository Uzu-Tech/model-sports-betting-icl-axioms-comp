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
#       jupytext_version: 1.11.2
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
from pipeline.data_loader import load_data
from utils.constants import MAIN_EUROPEAN_LEAGUES

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
# **Linear Baseline: Ordered Logistic Regression**
#
#
# **Non-Linear Logic: Gradient Boosted Trees (XGBoost)**
#
#
# **Goal-Based Distribution: Dixon-Coles Poisson Model**
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
# We opted for a 6-year historical window. This timeframe represents a good middle ground for sports modeling:
#
# Relevance: Data older than 6-7 years may reflect a different era of tactical play (pre-heavy pressing/VAR), which can "pollute" modern predictions.
# Volume: Six years provides over 22,000 matches, offering a great sample size to train and test our models without immediate overfitting.
#
# ### Implementation
# To ensure reproducibility and clean execution, the entire ingestion pipeline—including cleaning, joining, and handling missing ELO/Value data—is encapsulated in a custom modular function. This pipeline handles the merging of disparate CSV files into a single, analysis-ready DataFrame.
#
# Note: The full implementation of our data pipeline, including the load_data() logic, is available in our GitHub repository linked in the appendix.
#

# %%
from pipeline.data_loader import load_data
from utils.constants import MAIN_EUROPEAN_LEAGUES

NUM_YEARS = 6
df = load_data(leagues=MAIN_EUROPEAN_LEAGUES, num_years=6)

# %% [markdown]
# ## Exploratory Data Analysis

# %%
# Remove non numerical columns
df.drop("Date", "League", "HomeTeam", "AwayTeam", "FTR").describe()

# %% [markdown]
# Talk about descibed stats blah blah....

# %% [markdown]
# ## Feature Engineering
