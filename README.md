# nba-statline-forecasting
This project constructs an injury-aware forecasting system that predicts NBA player statlines (points, rebounds, assists) using player statistics and roster availability information.
When key players are ruled out of NBA games due to injury, teammates often experience significant changes in minutes played, opportunities, and statistical output. Modern forecasting approaches struggle to adapt quickly to these roster changes.
The goal of this project is to develop a data-driven forecasting model that takes into account injury-driven availability and matchup context to predict player performance.

## Project Objectives:
- Predict NBA player statistics (points, rebounds, assists)
- Incorporate injury reports and roster availability
- Analyze player statistics and matchup context
- Evaluate models using MAE and RMSE for error evaluation

## Repository Structure:
data/ - datasets
notebooks/ - EDA notebooks
src/ - modeling and pipeline code
experiments/ - tracking experiment outputs
docs/ - project documentation

## Data Sources:
- NBA player statistics via **nba_api**
- Official NBA injury reports from https://official.nba.com/nba-injury-report-2025-26-season/

## Evaluation Metrics:
Model performance will be evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to determine prediction accuracy and errors.
