# NBA Statline Forecasting - Checkpoint 2 Implementation Spec

## Objective
Build a historical NBA player-game forecasting framework that predicts:
- points
- rebounds
- assists

using three global models (one per target), not player-specific models.

## Dataset definition
One row = one player in one game.

## Seasons
- Training/development: 2023-24 and 2024-25
- Pure unseen evaluation: 2025-26

## Leakage rule
All engineered features for a game must use only information available before that game.

## Targets
- target_pts
- target_reb
- target_ast

## Core identifiers
- player_id
- player_name
- team_id
- team_abbr
- game_id
- game_date
- season
- opponent_team
- home_away

## Raw inputs to collect
From nba_api or equivalent official NBA stats sources where needed:
- player game logs
- advanced player box score usage (USG_PCT) if accessible
- opponent team pace
- opponent team defensive rating

## Feature groups

### Season-to-date player features
- season_avg_pts
- season_avg_reb
- season_avg_ast
- season_avg_min
- season_avg_fga
- season_avg_fg3a
- season_avg_fta
- season_avg_usg (if usage accessible)

### Rolling player features
Compute both L5 and L10:
- avg_pts
- avg_reb
- avg_ast
- avg_min
- avg_fga
- avg_fg3a
- avg_fta
- avg_usg (if usage accessible)

### Availability definition
For checkpoint 2, absent teammates are defined as teammates who did not appear in that game.

### Availability features
- teammates_out_count
- missing_pts_l5
- missing_reb_l5
- missing_ast_l5
- missing_min_l5
- missing_pts_l10
- missing_reb_l10
- missing_ast_l10
- missing_min_l10
- missing_usg_l5 (if usage accessible)
- missing_usg_l10 (if usage accessible)

### Matchup context
- opponent_team
- home_away
- opponent_pace
- opponent_def_rating

## Model versions

### Baseline A
Predict each target using season-to-date average before the game.

### Baseline B
Predict each target using L5 average before the game.

### V0
Use:
- season features
- L5 features
- L10 features
- opponent_team
- home_away

### V1
Use:
- all V0 features
- all availability features except usage-based ones if usage is unavailable

### V2
Use:
- all V1 features
- usage features
- opponent_pace
- opponent_def_rating

## Model families
Implement for each target:
- Linear Regression
- Random Forest Regressor

Optional:
- Gradient Boosting Regressor or XGBoost if implementation remains organized

## Evaluation
Primary metrics:
- MAE
- RMSE

Evaluate:
- Baseline A
- Baseline B
- V0
- V1
- V2

for each target:
- points
- rebounds
- assists

## Experiment tracking
Log all runs to wandb with:
- target
- model_version
- model_family
- feature_group
- train_seasons
- eval_season
- MAE
- RMSE
- runtime
- notes

## EDA requirements
Produce EDA on the final ML-ready dataset.

### General EDA
- dataset size and season coverage
- missingness summary
- target distributions
- descriptive stats
- correlation heatmap
- minutes vs targets
- season averages vs targets
- L5/L10 averages vs targets

### Modeling-oriented EDA
- teammates_out_count vs targets
- missing_pts_l5 vs target_pts
- missing_reb_l5 vs target_reb
- missing_ast_l5 vs target_ast
- opponent pace vs targets
- opponent defensive rating vs targets

## Deliverables
1. ML-ready dataset
2. EDA notebook and exportable PDF-ready figures
3. Results table for baseline/V0/V1/V2
4. wandb tracking logs
5. clean Python code with comments/docstrings
6. dataset card summary fields for presentation/reporting

## Important implementation constraints
- Do not leak target-game data into features
- Keep code modular and readable
- Prefer nba_api as backbone
- Only add external NBA sources when necessary
- Do not implement shot-profile matchup logic yet
- Do not implement real-time injury parsing yet