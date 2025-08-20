# NBA Playoff Game Predictor - Machine Learning Classifier
##### An interactive machine learning project that predicts the outcomes of NBA playoff games (2015-2025 seasons) using team statistics, Dean Oliver's Four Factors, and simulation-based models.
---
![Simulator Demo](images/miscellaneous/demo.gif)
---


## Overview
This project was built to explore sports analytics and predictive modeling in the NBA. It uses regular season and playoff data to train machine learning models that estimate win probabilities for each game and display predictions in a playoff simulator.

## Project Summary
The goal of this project was to predict NBA playoff game outcomes using machine learning models trained on advanced team statistics, series context, and matchup history. Multiple models were tested (Logistic Regression, Random Forest, Gradient Boosting, and Support Vector Machine) and evaluated on accuracy, precision, recall, and F1 Score.

All modeled performed reasonbly well, but none reached perfect accuracy given the unpredictability of playoff basketball. The best performing model was the **Support Vector Machine (SVM)**, which achieved:
- Accuracy: 59%
- F1 Score: 69%
- Precision: 61%
- Recall 79%

While Random Forest and Gradient Boosting achieved slightly better raw accuracy (up to ~57%), they underperformed on F1 Score. Since playoff predicition requires balancing false positives and false negatives (e.g, not just predicting favorites but capturing potential upsets), **F1 Score was chosen as the primary selection metric.** This made SVM the most suitable model overall. 

## Features
- Predicts outcomes of playoff games using SVM models
- Incorporates [Dean Oliver's Four Factors](https://www.basketball-reference.com/about/factors.html) (eFG%, TOV%, ORB%, DRB%, FT/FGA)
- Accounts for seeding, head-to-head matchups, momentum, and series context
- Interactive Streamlit app that displays predictions alongside actual results

## Project Structure
```bash
NBA-Playoffs-Game-Predictor/
│── data/            # Datasets (Raw, processed, and final)
│── images/          # Project images and logos
│── model/           # Saved machine learning models and features
│── notebooks/       # Jupyter notebooks for data processing, feature engineering, modeling, and simulator development
│── simulator/       # Streamlit / Python scripts for running the simulator
│── README.md        # Project documentation
```

## Data Source
- [Kaggle NBA Dataset](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores/data)

## How to Run
```bash
# Clone the repository
git clone https://github.com/savannahvo/NBA-Playoffs-Game-Predictor.git
cd NBA-Playoffs-Game-Predictor

# Intall dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run NBA playoffs simulator.py
```

## Results/Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nba-playoffs-game-predictor-6axovmkygjqsmmvcevipcd.streamlit.app/)

## Future Work
- Improve model by incorporating play-level data
- Add live data streaming for ongoing playoffs
- Deploy with a web API

## Contact
# Created by **Savannah Vo**
# Email: savannahlevo@gmail.com || [LinkedIn](https://www.linkedin.com/in/savannahlevo/)


     


