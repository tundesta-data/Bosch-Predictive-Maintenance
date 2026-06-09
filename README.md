# Bosch-Predictive-Maintenance

# Bosch Rexroth AG — Predictive Maintenance ML Pipeline

## Overview
Automated ML prediction system for hydraulic
pump unit failure prediction and RUL estimation.

## What this does
Runs every hour automatically:
1. Connects to PostgreSQL database
2. Loads latest sensor telemetry
3. Runs XGBoost RUL and failure classifier
4. Writes predictions to ml_predictions table
5. Power BI dashboard reads new predictions

## Models
- XGBoost RUL Regressor (MAE: 14.87 hours)
- XGBoost Failure Classifier (F1: 99.97%)

## Tools
PostgreSQL · Python · XGBoost · Render.com
