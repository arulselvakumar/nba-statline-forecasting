from pathlib import Path
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="NBA Statline Forecast Demo", layout="wide")

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "final" / "ml_ready_player_games_v3.parquet"
MODELS_DIR = ROOT / "models"

df = pd.read_parquet(DATA_PATH)
df["game_date"] = pd.to_datetime(df["game_date"])

replay_df = df[df["season"] == "2024-25"].copy()

player_role = (
    replay_df.groupby("player_name", as_index=False)
    .agg(
        avg_pts=("season_avg_pts", "mean"),
        avg_min=("season_avg_min", "mean"),
        avg_fga=("season_avg_fga", "mean"),
        avg_usg=("season_avg_usg", "mean"),
        games=("game_id", "nunique")
    )
)

player_role = player_role[player_role["games"] >= 10].copy()

player_role["demo_score"] = (
    player_role["avg_pts"].rank(pct=True) * 0.35 +
    player_role["avg_min"].rank(pct=True) * 0.30 +
    player_role["avg_fga"].rank(pct=True) * 0.25 +
    player_role["avg_usg"].rank(pct=True) * 0.10
)

top_players = player_role[
    player_role["demo_score"] >= player_role["demo_score"].quantile(0.80)
]["player_name"]

replay_df = replay_df[replay_df["player_name"].isin(top_players)].copy()

st.write("Demo player pool:", replay_df["player_name"].nunique(), "players")

model_pts = joblib.load(MODELS_DIR / "xgboost_v4_target_pts.pkl")
model_reb = joblib.load(MODELS_DIR / "xgboost_v4_target_reb.pkl")
model_ast = joblib.load(MODELS_DIR / "xgboost_v4_target_ast.pkl")

drop_cols = {
    "target_pts", "target_reb", "target_ast",
    "pts", "reb", "ast", "min",
    "fgm", "fga", "fg3m", "fg3a",
    "ftm", "fta", "oreb", "dreb",
    "stl", "blk", "tov", "pf",
    "plus_minus", "wl",
    "game_id", "player_name", "game_date",
    "season"
}

feature_cols = [c for c in df.columns if c not in drop_cols]

st.title("NBA Statline Forecasting Demo")
st.caption("Historical replay demo using held-out 2024–25 games.")

if "row" not in st.session_state:
    st.session_state.row = replay_df.sample(1).iloc[0]

if st.button("Generate New Held-Out Game"):
    st.session_state.row = replay_df.sample(1).iloc[0]

row = st.session_state.row

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Game Context")
    st.write(f"**Player:** {row.get('player_name', 'Unknown')}")
    st.write(f"**Team:** {row.get('team_abbr', 'N/A')}")
    st.write(f"**Opponent:** {row.get('opponent_team', 'N/A')}")
    st.write(f"**Date:** {row['game_date'].date()}")
    st.write(f"**Season:** {row.get('season', 'N/A')}")

    st.write("### Can You Beat the Model?")
    user_pts = st.number_input("Your PTS prediction", min_value=0.0, step=1.0)
    user_reb = st.number_input("Your REB prediction", min_value=0.0, step=1.0)
    user_ast = st.number_input("Your AST prediction", min_value=0.0, step=1.0)

with right:
    st.subheader("Prediction Results")

    if st.button("Reveal Results", use_container_width=True):
        X = row[feature_cols].to_frame().T

        pred_pts = model_pts.predict(X)[0]
        pred_reb = model_reb.predict(X)[0]
        pred_ast = model_ast.predict(X)[0]

        actual_pts = row["target_pts"]
        actual_reb = row["target_reb"]
        actual_ast = row["target_ast"]

        results = pd.DataFrame({
            "Stat": ["PTS", "REB", "AST"],
            "Your Guess": [user_pts, user_reb, user_ast],
            "Model Prediction": [pred_pts, pred_reb, pred_ast],
            "Actual": [actual_pts, actual_reb, actual_ast],
        })

        results["Your Abs Error"] = (results["Your Guess"] - results["Actual"]).abs()
        results["Model Abs Error"] = (results["Model Prediction"] - results["Actual"]).abs()

        results["Model Prediction"] = results["Model Prediction"].round(2)
        results["Your Abs Error"] = results["Your Abs Error"].round(2)
        results["Model Abs Error"] = results["Model Abs Error"].round(2)

        st.dataframe(results, use_container_width=True, hide_index=True)

        user_total_error = results["Your Abs Error"].sum()
        model_total_error = results["Model Abs Error"].sum()

        col1, col2 = st.columns(2)
        col1.metric("Your Total Error", round(user_total_error, 2))
        col2.metric("Model Total Error", round(model_total_error, 2))

        if user_total_error < model_total_error:
            st.success("You beat the model!")
        elif user_total_error > model_total_error:
            st.error("Model wins!")
        else:
            st.info("Tie!")

st.divider()

st.subheader("Deployment Logic")
st.write(
    "This demo simulates real-world inference by selecting a held-out 2024–25 game, "
    "using only pregame features, generating predictions with the final V4 XGBoost models, "
    "and comparing model predictions against actual statline outcomes."
)