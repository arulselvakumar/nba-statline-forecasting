# NBA Statline Forecasting

## Overview
This project builds a context-aware machine learning system to forecast NBA player statlines (Points, Rebounds, Assists) using pregame features and deployable models.

NBA player performance varies significantly based on game context, including opponent strength, usage rate, and roster availability. Traditional forecasting methods relying on static averages fail to capture these dynamics.

This project emphasizes pregame forecasting with strict prevention of data leakage by using only features available before game start.

---

## Project Objectives
- Predict NBA player statistics (Points, Rebounds, Assists)
- Develop context-aware feature engineering (Baseline → V3)
- Compare linear and nonlinear models (Linear Regression, Random Forest, XGBoost)
- Optimize model performance using hyperparameter tuning (V4)
- Evaluate performance on a future season using time-based validation
- Deploy an interactive forecasting system using Streamlit

---

## Dataset
- Source: nba_api
- ~52,707 player-game rows
- 694 players
- ~2,460 games
- Seasons:
  - Train: 2023–24
  - Test: 2024–25 (unseen)

---

## Methodology

### Feature Engineering (Versioning)
- Baseline: season averages
- V0: + opponent context + home/away
- V1: + availability / missing production
- V2: + usage + matchup context
- V3: + interaction features (usage × minutes, pace-adjusted stats, rolling trends)
- V4: hyperparameter tuning on V3 features

### Models
- Linear Regression (baseline interpretability)
- Random Forest (ensemble)
- XGBoost (final model)

### Evaluation
- Time-based holdout (no random split)
- Metrics:
  - MAE (primary)
  - RMSE (secondary)
- Evaluation performed on future season (2024–25)

---

## Results

Best Model: **XGBoost (V4)**

| Target | MAE |
|--------|-----|
| Points (PTS) | ~2.19 |
| Rebounds (REB) | ~0.91 |
| Assists (AST) | ~0.60 |

Key insight:
Feature engineering (V3) combined with hyperparameter tuning (V4) produced largest performance gains.

---

## Deployment

Project includes a Streamlit web application simulating real-world pregame forecasting.

Features:
- Generate random held-out NBA games
- Input user statline predictions
- Compare user predictions vs model vs actual results
- Evaluate performance using total prediction error

Run locally:
```bash
streamlit run deployment/app.py
```

---

## Repository Structure

data/         - datasets (raw, intermediate, final)  
notebooks/    - data processing, modeling, evaluation  
models/       - trained models (local use)  
outputs/      - figures and evaluation results  
deployment/   - Streamlit application  

---

## Monitoring & Governance

- Track MAE / RMSE over time  
- Monitor feature drift (usage, lineup changes)  
- Retrain models periodically (rolling season updates)  
- Maintain model versioning and experiment tracking  

---

## Future Work

- Explore neural network-based models (deep learning)  
- Real-time data pipeline integration  
- Public API or dashboard deployment  
- Enhanced lineup and matchup feature modeling