import polars as pl
import plotly.graph_objects as go
import polars.selectors as cs
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pipeline.risk import kelly_criterion, flat_bet, fixed_fraction

import logging
logging.getLogger('kaleido').setLevel(logging.ERROR)
logging.getLogger('plotly').setLevel(logging.ERROR)

def plot_equity_comparison(
    type: str,
    model_bet_series, 
    initial_bank=500, 
    flat_size=10, 
    fixed_frac=0.005, 
    kelly_fractions=np.arange(0.1, 0.6, 0.1),
):
    fig = go.Figure()
    # Professional color scale for Kelly variations
    kelly_colors = ["#636EFA", "#19D3F3", "#FFA15A", "#FF6692", "#B6E880", "#AB63FA"]

    # 1. Plot Kelly Fractions
    for i, frac in enumerate(kelly_fractions):
        f_val = round(frac, 1)
        kelly_series = kelly_criterion(
            model_bet_series, 
            initial_bank_roll=initial_bank, 
            fraction=f_val
        )
        
        fig.add_trace(go.Scatter(
            x=kelly_series["match_id"], 
            y=kelly_series["new_bank_roll"], 
            name=f"{f_val} Kelly",
            line=dict(color=kelly_colors[i % len(kelly_colors)], width=2)
        ))

    # 2. Plot Flat Bet Baseline (Green)
    flat_equity = flat_bet(
        model_bet_series, 
        initial_bank_roll=initial_bank, 
        flat_bet=flat_size
    )
    fig.add_trace(go.Scatter(
        x=flat_equity["match_id"], 
        y=flat_equity["new_bank_roll"], 
        name=f"Flat Bet (${flat_size})",
        line=dict(color="#00CC96", width=3, dash='dot')
    ))

    # 3. Plot Fixed Fraction (Red)
    fixed_frac_equity = fixed_fraction(
        model_bet_series, 
        initial_bank_roll=initial_bank, 
        fraction=fixed_frac
    )
    fig.add_trace(go.Scatter(
        x=fixed_frac_equity["match_id"], 
        y=fixed_frac_equity["new_bank_roll"], 
        name=f"Fixed Fraction ({fixed_frac*100}%)",
        line=dict(color="#EF553B", width=2, dash='dash')
    ))

    # 4. Styling
    fig.update_layout(
        title=f"<b>{type} Equity Growth Comparison</b><br><sup>Risk Management Strategy Performance Over Time</sup>",
        xaxis_title="Match Index",
        yaxis_title="Bankroll Value ($)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=100, b=50)
    )

    fig.update_layout(
        width=1200,
        height=700,
        margin=dict(t=120), # Gives the title room
        legend=dict(
            orientation="h",
            y=-0.2,         # Moves the legend BELOW the chart instead of on top
            x=0.5,
            xanchor="center"
        )
    )
    
    fig.show(renderer='png', scale=1.5)


def plot_model_comparison(market_results, model_results):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    splits = [f"Split {i+1}" for i in range(len(market_results))]

    # --- 1. MARKET BASELINES (Black Lines) ---
    # Log Loss Baseline
    fig.add_trace(go.Scatter(
        x=splits, 
        y=[r['Log Loss'] for r in market_results], 
        name="Market Baseline (Loss)", 
        line=dict(color="black", width=3, dash='dot')
    ), secondary_y=False)

    # Accuracy Baseline
    fig.add_trace(go.Scatter(
        x=splits, 
        y=[r['Accuracy'] for r in market_results], 
        name="Market Baseline (Acc)", 
        line=dict(color="black", width=2, dash='longdash'),
        opacity=0.3
    ), secondary_y=True)


    colors = {
        "Logistic": "#636EFA", 
        "XGBoost": "#00CC96", 
        "LightGBM": "#EF553B",
    }

    for name, data in model_results.items():
        color = colors.get(name, "#888888") 
        # Plot Log Loss (Solid Line)
        fig.add_trace(go.Scatter(
            x=splits, 
            y=data["losses"], 
            name=f"{name} Loss",
            line=dict(color=color, width=2)
        ), secondary_y=False)
        
        # Plot Accuracy (Dashed Line)
        fig.add_trace(go.Scatter(
            x=splits, 
            y=data["accuracies"], 
            name=f"{name} Acc",
            line=dict(color=color, width=1, dash='dash'),
            opacity=0.6
        ), secondary_y=True)

    # --- 3. LAYOUT & FORMATTING ---
    fig.update_layout(
        title="Model Selection Performance vs Market Baseline", 
        template="plotly_white", 
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="Log Loss (Lower is Better)", secondary_y=False)
    fig.update_yaxes(title_text="Accuracy (Higher is Better)", secondary_y=True)

    fig.update_layout(
        width=1200,
        height=700,
        margin=dict(t=120), # Gives the title room
        legend=dict(
            orientation="h",
            y=-0.2,         # Moves the legend BELOW the chart instead of on top
            x=0.5,
            xanchor="center"
        )
    )
    
    fig.show(renderer='png', scale=1.5)

def plot_anova_importance(anova_df: pl.DataFrame, top_n: int = 20):
    plot_data = anova_df.sort("F-Statistic", descending=True).head(top_n)
    
    # Create the figure
    fig = px.bar(
        plot_data.to_pandas(), # Plotly works natively with pandas/polars
        x="F-Statistic",
        y="Feature",
        color="p-value",
        orientation='h',
        title=f"Top {top_n} Features by ANOVA Significance",
        labels={"F-Statistic": "F-Statistic (Signal Strength)", "p-value": "Significance (p-value)"},
        color_continuous_scale="Viridis_r", # Reversed Viridis so low p-values are bright/distinct
        template="plotly_white"
    )
    
    # Invert Y-axis so the highest F-stat is at the top
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=200 + (top_n * 20), # Dynamic height based on number of features
        coloraxis_colorbar=dict(title="p-value")
    )
    
    # Add a vertical line for a common significance threshold if p-values are visible
    fig.show(renderer='png', scale=1.5)

def plot_bookmaker_baseline(metrics_df: pl.DataFrame):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Log Loss trace
    fig.add_trace(
        go.Scatter(
            x=metrics_df["Validation Set"], 
            y=metrics_df["Log Loss"], 
            name="Log Loss",
            mode='lines+markers',
            line=dict(color="firebrick", width=3)
        ),
        secondary_y=False,
    )

    # Add Accuracy trace
    fig.add_trace(
        go.Scatter(
            x=metrics_df["Validation Set"], 
            y=metrics_df["Accuracy"], 
            name="Accuracy",
            mode='lines+markers',
            line=dict(color="royalblue", width=3, dash='dash')
        ),
        secondary_y=True,
    )

    # Formatting
    fig.update_layout(
        title="Baseline Market Performance: Log Loss & Accuracy per Validation Set",
        xaxis_title="Time Series Split",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=1200,
        height=600,
    )

    fig.update_yaxes(title_text="<b>Log Loss</b> (Lower is better)", secondary_y=False, color="firebrick")
    fig.update_yaxes(title_text="<b>Accuracy</b> (Higher is better)", secondary_y=True, color="royalblue")

    fig.show(renderer='png', scale=1.5)

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
    fig.show(renderer='png', scale=1.5)

def plot_vif(vif_df, threshold_warn=5, threshold_crit=10):

    fig = px.bar(
        vif_df, 
        x="VIF", 
        y="feature", 
        orientation='h',
        title="Feature Multicollinearity (VIF Scores)",
        color="VIF",
        color_continuous_scale="OrRd", 
        log_x=True, # Enable log scale
        labels={"VIF": "VIF Score (Log Scale)", "feature": "Feature Name"}
    )

    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        height=max(400, len(vif_df) * 25),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=50)
    )

    fig.show(renderer='png', scale=1.5)


def plot_feature_weights_plotly(importance_df, model, feature_list):
    coef_map = pl.DataFrame({
        "feature": feature_list,
        "home_coef": model.coef_[0],
        "away_coef": model.coef_[1]
    })

    # 2. Join with importance_df to keep only 'surviving' (non-zero) features
    # We sort by the magnitude of the home coefficient for better scannability
    plot_df = (
        importance_df.select("feature")
        .join(coef_map, on="feature", how="left")
        .sort("home_coef", descending=False)
    )

    # 3. Build the Plotly Figure
    fig = go.Figure()

    # Home Residual Trace
    fig.add_trace(go.Bar(
        y=plot_df["feature"],
        x=plot_df["home_coef"],
        name="Home Residual",
        orientation='h',
        marker_color='rgba(55, 128, 191, 0.7)',
        marker_line=dict(color='rgba(55, 128, 191, 1.0)', width=1)
    ))

    # Away Residual Trace
    fig.add_trace(go.Bar(
        y=plot_df["feature"],
        x=plot_df["away_coef"],
        name="Away Residual",
        orientation='h',
        marker_color='rgba(219, 64, 82, 0.7)',
        marker_line=dict(color='rgba(219, 64, 82, 1.0)', width=1)
    ))

    # 4. Styling
    fig.update_layout(
        title="<b>Feature Impact: Home vs Away Residuals</b><br><sup>Directional Nudges from MultiTask ElasticNet</sup>",
        xaxis_title="Coefficient Value (Influence on Residual)",
        yaxis_title="Feature",
        barmode='group',
        template="plotly_white",
        height=max(400, len(plot_df) * 40), # Dynamic height
        hovermode="y unified",
        shapes=[dict(
            type='line',
            yref='paper', y0=0, y1=1,
            xref='x', x0=0, x1=0,
            line=dict(color="black", width=2, dash="dash")
        )]
    )

    fig.show(renderer='png', scale=1.5)



def plot_anova(summary_df):
    # 1. Prepare data for plotting
    # We convert to pandas for easier manipulation in Plotly
    plot_df = summary_df.to_pandas()

    # Log-transform F-Statistic
    plot_df["Log_F_Stat"] = np.log10(plot_df["f_stat"] + 1)

    # Calculate Significance Score but CAP it at 5.0
    # This prevents p-values of 1e-100 from ruining the scale
    plot_df["Significance_Capped"] = np.clip(-np.log10(plot_df["p_value"] + 1e-300), 0, 5)

    # Sort by the actual impact (F-Stat)
    plot_df = plot_df.sort_values("Log_F_Stat", ascending=True)

    # 2. Build the Chart
    fig = go.Figure()

    # Add Log10 F-Statistic Bars (The actual 'meat' of the feature)
    fig.add_trace(go.Bar(
        y=plot_df["feature"],
        x=plot_df["Log_F_Stat"],
        name="Log10(F-Statistic) [Impact]",
        orientation='h',
        marker_color='#2c3e50',
        customdata=plot_df["f_stat"],
        hovertemplate="<b>%{y}</b><br>Raw F-Stat: %{customdata:.2f}<br>Log F-Stat: %{x:.2f}"
    ))

    # Add Capped Significance Bars (The 'Confidence')
    fig.add_trace(go.Bar(
        y=plot_df["feature"],
        x=plot_df["Significance_Capped"],
        name="-Log10(p) [Certainty] (Capped at 5)",
        orientation='h',
        marker_color='#e67e22',
        customdata=plot_df["p_value"],
        hovertemplate="p-value: %{customdata:.4e}<br>Significance: %{x:.2f}"
    ))

    # 3. Layout and Threshold
    fig.add_vline(x=1.30, line_dash="dash", line_color="red", 
                annotation_text="p=0.05", annotation_position="bottom right")

    fig.update_layout(
        title="<b>Feature Importance for Outcome (H, D, A): Impact vs. Certainty</b><br><sup>Significance capped to prevent scale distortion</sup>",
        xaxis_title="Score (Log Scale)",
        barmode='group',
        template="plotly_white",
        height=max(500, len(plot_df) * 35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="y unified"
    )

    fig.show(renderer='png', scale=1.5)
