from pathlib import Path
import os
import pandas as pd
import wandb

os.environ["WANDB_SILENT"] = "true"

ROOT = Path(__file__).resolve().parent

RESULTS_PATH = ROOT / "ipynb" / "outputs" / "results" / "model_results.csv"

print("Using results file:", RESULTS_PATH)
if not RESULTS_PATH.exists():
    raise FileNotFoundError(f"Could not find: {RESULTS_PATH}")

df = pd.read_csv(RESULTS_PATH)

print("Loaded shape:", df.shape)
print("Columns:", df.columns.tolist())

required_cols = [
    "run_id",
    "target",
    "model_version",
    "model_family",
    "feature_group",
    "train_seasons",
    "eval_season",
    "mae",
    "rmse",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df_to_log = df.sort_values("mae").head(12).copy()

print("\nRuns to log:")
print(df_to_log[["run_id", "target", "model_version", "model_family", "mae", "rmse"]])

project_name = "nba-capstone-checkpoint2"

for _, row in df_to_log.iterrows():
    run = wandb.init(
        project=project_name,
        job_type="backfill_experiment",
        name=row["run_id"],
        config={
            "target": row["target"],
            "model_version": row["model_version"],
            "model_family": row["model_family"],
            "feature_group": row["feature_group"],
            "train_seasons": row["train_seasons"],
            "eval_season": row["eval_season"],
        },
    )

    wandb.log({
    "mae": float(row["mae"]),
    "rmse": float(row["rmse"]),
    })

    run.finish()

print("\nDone logging runs to W&B.")