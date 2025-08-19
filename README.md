# NBA Playoff Game Predictor - Machine Learning Classifier

An interactive machine learning project that predicts the outcomes of NBA playoff games (2015-2025 seasons) using team statistics, Dean Oliver's Four Factors, and simulation-based models.
---
![NBA Playoffs Court](https://github.com/savannahvo/NBA-Playoffs-Game-Predictor/blob/main/2025%20NBA%20Playoffs%20Court.jpeg?raw=true)
---
## Overview
This project was built to explore sports analytics and predictive modeling in the NBA. It uses regular season and playoff data to train machine learning models that estimate win probabilities for each game and display predictions in a playoff simulator.

## Features
- Predicts outcomes of playoff games using SVM models
- Incorporates [Dean Oliver's Four Factors](https://www.basketball-reference.com/about/factors.html) (eFG%, TOV%, ORB%, DRB%, FT/FGA)
- Accounts for seeding, head-to-head matchups, momentum, and series context
- Interactive Streamlit app that displays predictions alongside actual results

# Project Structure
````bash
NBA-Playoffs-Game-Predictor/
│── data/               # Raw, processed, and final data (Games.csv, TeamStatistics.csv)
│── notebooks/          # Jupyter notebooks for data prep, feature engineering, modeling, and simulator development
│── models/             # Trained ML model
│── simulator/          # Helper scripts and utilities
│── README.md           # Project documentation

## Project Summary
The goal of this project was to predict NBA playoff game outcomes using machine learning models trained on advanced team statistics, series context, and matchup history. Multiple models were tested (Logistic Regression, Random Forest, Gradient Boosting, and Support Vector Machine) and evaluated on accuracy, precision, recall, and F1 Score.

All modeled performed reasonbly well, but none reached perfect accuracy given the unpredictability of playoff basketball. The best performing model was the **Support Vector Machine (SVM)**, which achieved:
- Accuracy: 59%
- F1 Score: 69%
- Precision: 61%
- Recall 79%

While Random Forest and Gradient Boosting achieved slightly better raw accuracy (up to ~57%), they underperformed on F1 Score. Since playoff predicition requires balancing false positives and false negatives (e.g, not just predicting favorites but capturing potential upsets), **F1 Score was chosen as the primary selection metric.** This made SVM the most suitable model overall. 

     - It's calculated as: `FT/FGA = FTM ÷ FGA`
     - This captures offensive aggression and efficiency; teams that foul less and shoot accurately often convert more points per possession.


