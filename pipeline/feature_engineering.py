import polars as pl
import pandas as pd
import polars.selectors as cs
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
import numpy as np
import statsmodels.api as sm

def run_stats_investigation(df, target_y, feature_cols):
    X = df.select(feature_cols).to_numpy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_scaled = sm.add_constant(X_scaled)
    
    model = sm.MNLogit(target_y, X_scaled)
    result = model.fit(method='lbfgs', maxiter=1000)
    
    col_names = ['const'] + feature_cols
    print(result.summary(xname=col_names))
    
    return result

def get_features_df(ml_features: pl.DataFrame):
    return ml_features.select(cs.numeric()).drop('match_id')

def get_model_eval(model, X, y, time_split):
    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', clone(model))
        ])
        
    losses, accuracies = [], []
    for train_idx, test_idx in time_split.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        pipeline.fit(X_train, y_train)
        
        probs = pipeline.predict_proba(X_test)
        preds = pipeline.predict(X_test)
        
        losses.append(log_loss(y_test, probs, labels=[0, 1, 2]))
        accuracies.append(accuracy_score(y_test, preds))

    return np.mean(losses), np.mean(accuracies), losses, accuracies



def forward_selection(model, initial_features, candidate_features, X_df, y, time_split):
    current_features = list(initial_features)
    best_overall_loss = float('inf')
    
    while True:
        best_feature_this_round = None
        
        for feature in candidate_features:
            if feature in current_features: 
                continue
            
            trial_features = current_features + [feature]
            X_trial = X_df.select(trial_features).to_numpy()
            
            mean_loss, _, _, _ = get_model_eval(model, X_trial, y, time_split)
            
            if mean_loss < best_overall_loss:
                best_overall_loss = mean_loss
                best_feature_this_round = feature

        if best_feature_this_round:
            current_features.append(best_feature_this_round)
            print(f"Added feature {best_feature_this_round}")
        else:
            break
            
    return current_features


def calculate_market_probs(df: pl.DataFrame):
    df = df.with_columns(
        raw_prob_h= 1 / pl.col("AvgH"),
        raw_prob_a= 1 / pl.col("AvgA"),
        raw_prob_d= 1 / pl.col("AvgD")
    )

    df = df.with_columns(
        implied_probs_sum=pl.col("raw_prob_h") + pl.col("raw_prob_a") + pl.col("raw_prob_d")
    )

    return df.with_columns(
        pl.col(["raw_prob_h", "raw_prob_d", "raw_prob_a"]) / pl.col("implied_probs_sum")
    ).rename({
        "raw_prob_h": "home_prob",
        "raw_prob_a": "away_prob",
        "raw_prob_d": "draw_prob"
    }).drop("implied_probs_sum")

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


def add_seasons(df: pl.DataFrame, season_start_month: int):
    return df.with_columns(
        Season = (
            pl.when(pl.col("Datetime").dt.month() >= season_start_month)
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


def add_custom_ewm_features(
    df: pl.DataFrame, windows: list[int], feature_definitions: dict[str, dict]
):
    home_side = df.select(
        [
            pl.col("match_id"),
            pl.col("Datetime"),
            pl.col("HomeTeam").alias("team"),
            *[
                (
                    val["home"].alias(name)
                    if isinstance(val["home"], pl.Expr)
                    else pl.col(val["home"]).alias(name)
                )
                for name, val in feature_definitions.items()
            ],
        ]
    )

    away_side = df.select(
        [
            pl.col("match_id"),
            pl.col("Datetime"),
            pl.col("AwayTeam").alias("team"),
            *[
                (
                    val["away"].alias(name)
                    if isinstance(val["away"], pl.Expr)
                    else pl.col(val["away"]).alias(name)
                )
                for name, val in feature_definitions.items()
            ],
        ]
    )

    stacked = pl.concat([home_side, away_side]).sort("Datetime", "team")
    feature_names = list(feature_definitions.keys())

    stacked = stacked.with_columns(
        [
            pl.col(name)
            .ewm_mean(span=window)
            .over("team")
            .name.suffix(f"_ewm_{window}")
            for name in feature_names
            for window in windows
        ]
    )

    latest_stats = (
        stacked.sort("Datetime")
        .group_by("team")
        .last()
    )

    stacked = stacked.with_columns([cs.contains("ewm").shift().over("team")])

    ewm_cols = [c for c in stacked.columns if "_ewm_" in c]

    for side in ("Home", "Away"):
        team_col = f"{side}Team"
        df = df.join(
            stacked.select("match_id", "team", cs.contains("ewm")),
            left_on=["match_id", team_col],
            right_on=["match_id", "team"],
            how="left",
        ).rename({c: f"{side.lower()}_{c}" for c in ewm_cols})

    return df, latest_stats
