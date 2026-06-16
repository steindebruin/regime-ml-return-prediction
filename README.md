# Regime-Dependent Return Prediction with Machine Learning

BSc thesis Econometrics & Economics by Stein de Bruin (596873sb), Erasmus School of Economics.

## Overview
This repository contains the code for the thesis "Regime-Dependent Return Prediction with Machine Learning". The analysis extends the cross-sectional return prediction framework of Gu, Kelly, and Xiu (2020) with a two-state hidden Markov model that identifies bull and bear market regimes.

## Data
Data is obtained from WRDS through the data scripts. All other data files are generated automatically by the scripts.

## Requirements
Install dependencies with:
```
pip install -r requirements.txt
```

## Usage
Run the full analysis from the `code/` folder with:
```
python main.py
```

## Structure
```
regime-ml-return-prediction/
 - main.py              — runs all scripts in order
 - utils.py             — shared utility functions
 - requirements.txt     — Python dependencies
 - data/                — data files
 - results/             — predictions, metrics, tables, and plots
 - *.py                 — analysis scripts
```