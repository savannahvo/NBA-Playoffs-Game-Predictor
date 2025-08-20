# NBA Playoff Game Predictor 
#### An interactive sports analytics project that predicts the outcomes of NBA playoff games using machine learning. The system incorporates [Dean Oliver's Four Factors](https://www.basketball-reference.com/about/factors.html), series context, and simulation-based models to estimate win probabilities and visualize predictions in a Streamlit application.
---
![Simulator Demo](images/miscellaneous/demo.gif)
---


## Overview
This project explores predictive modeling in sports analytics with a focus on the NBA playoffs, where unpredictability and momentum play major roles. The dataset covers NBA playoff games from 2015-2025, with models trained on 2015-2022 seasons, validated on 2023-2024, and evaluated on the 2025 playoffs to simulate real-world predictive performance.

## Project Summary
The goal was to predict NBA playoff game outcomes using multiple machine learning models trained on team statistics, matchup history, and series dynamics.

Models evaluated:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SMV)

The Support Vector Machine (SVM) emerged as the best overall model based on its ability to balance accuracy with precision and recall, particularly important for playoff scenarios where upsets are common. F1 Score was chosen as the primary evaluation metric, since playoff prediction requires balancing false positives and false negatives.

Final Model Performance (SVM):
- Accuracy: 59%
- F1 Score: 69%
- Precision: 61%
- Recall 79%


## Features
- Predicts outcomes of playoff games using SVM classification models
- Incorporates [Dean Oliver's Four Factors](https://www.basketball-reference.com/about/factors.html) (eFG%, TOV%, ORB%, DRB%, FT/FGA)
- Adjusts for seeding, head-to-head matchups, momentum, and series context
- Interactive Streamlit siumlator displaying predictions vs. actual results
- Modular project design for scalability and future enhancements

## Project Structure
```bash
NBA-Playoffs-Game-Predictor/
│── data/            # Datasets (Raw, processed, and final)
│── images/          # Project images and logos
│── model/           # Saved machine learning models and features
│── notebooks/       # Jupyter notebooks for data processing, feature engineering, modeling, and simulator development
│── simulator/       # Streamlit app / Python scripts for running the simulator
│── README.md        # Project documentation
```

## Data Source
- [Kaggle: Historical NBA Data and Player Box Scores](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores/data)
- Files used: `Games.csv` and `TeamStatistics.csv`

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
- Interactive simulator available via Streamlit
- Displays predicted vs. actual winners, confidence scores, and team statistics
- Provides insights into model strengths and weaknesses across different playoff rounds
  
[![Open in Streamlit](https://nba-playoffs-game-predictor-mto2nkdzcubqnv9curkyfz.streamlit.app/)

## Future Work
- Play-level data integration -> Improve granularity of features by analyzing player actions
- Live data streamining -> Update predictions during ongoing playoff games
- Web API deployment -> Expose predictions for integration into dashboards or external apps

## Tech Stack
- Languages and Tools: Python.....

## Contact
Email: savannahlevo@gmail.com || [LinkedIn](https://www.linkedin.com/in/savannahlevo/)     


