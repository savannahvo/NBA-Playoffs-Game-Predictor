# NBA Playoff Game Predictor 
#### An interactive sports analytics project that predicts the outcomes of NBA playoff games using machine learning. The system incorporates [Dean Oliver's Four Factors](https://www.basketball-reference.com/about/factors.html), series context, and simulation-based models to estimate win probabilities and visualize predictions in a Streamlit application.
---
![Simulator Demo](images/miscellaneous/demo.gif)
---


## Overview
This project explores predictive modeling in sports analytics with a focus on the NBA playoffs, where unpredictability and momentum play major roles. The dataset covers NBA playoff games from 2015-2025, with models trained on 2015-2022 seasons, validated on 2023-2024, and evaluated on the 2025 playoffs to simulate real-world predictive performance.

---
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
---
## Why is accuracy ~59%?
Single-game basketball prediction has a hard ceiling. Even the “true” pre-game favorite loses a lot because of variance (hot/cold 3-point shooting, foul trouble, random runs), matchup quirks, and late-game coin-flip situations. If the average favorite only wins, say, ~60–65% of the time, then the **Bayes-optimal top-1 accuracy** (always pick the higher-probability side) can’t exceed that average very much. In other words, when many games are inherently close, **59% top-1 accuracy is not far from the practical ceiling** without using betting-market information.

#### Sources of irreducible noise
- **High variance from 3s & free throws:** Small swings in 3-point luck or whistle patterns flip outcomes that models can’t foresee.
- **Lineups & injuries:** Late scratches, minute limits, and rotation changes are hard to encode from historical team stats alone.
- **Series dynamics:** Adjustments within a series (matchups, scheme tweaks) shift team strength game-to-game.
- **Small playoff sample sizes:** Relative to the regular season, the playoffs provide far fewer training examples per round/game context.
- **Parity & clutch time:** Many playoff games have one-possession “coin-flip” finishes.

#### Modeling constraints specific to this project
- **Team-level features:** The simulator uses team four-factors, series state, H2H aggregates, etc. It does not (yet) include player-level RAPM/RAPTOR/EPM, injury reports, referee assignments, or betting lines—features that typically boost accuracy.
- **Round/game-specific shifts:** Features are tailored by round/game number, which helps, but the data per bucket is smaller, limiting what more complex models can learn.
- **SVM focus:** The deployed model is an SVM. Tree ensembles (GBM/XGBoost/Random Forest) or stacking can help capture non-linear interactions, and post-hoc **probability calibration** (isotonic/Platt) can improve the quality of probabilities even if top-1 accuracy moves little.

#### Accuracy isn’t the whole story
Top-1 accuracy treats a 51% call the same as a 90% call. For probability models, it’s more informative to report:
- **Brier score** (mean squared error of the probabilities)
- **Log loss** (penalizes over-confident wrong picks)
- **Calibration** plots (how often 60% probabilities win, etc.)
- **AUC/ROC** for ranking strength
These reflect whether the model’s confidence is trustworthy, not just whether the pick matched the final score.

#### Reasonable baselines
- **Home-team or favorite heuristics** typically land near the high-50s/low-60s in many NBA settings.
- **Betting markets** set a practical upper bound; beating them consistently is very difficult. Models that approach market-level accuracy/calibration are already strong.

#### How to push beyond ~59%
- Add **player-level** features (availability, on/off, RAPM-style impact, recent form).
- Ingest **injury news** / starting lineups and **rest/travel** data.
- Include **market signals** (moneyline/point spread) as features (for research comparison; omit if you want a purely “from data” model).
- Use **ensembles** (e.g., gradient boosting + logistic stacking) and **calibrate** probabilities.
- Expand training data where appropriate (regular-season analogues, with playoff-specific adjustments).
- Evaluate by **log loss / Brier** and **calibration**, not only accuracy.

**TL;DR:** A ~59% single-game accuracy in the NBA playoffs is consistent with the sport’s randomness and the feature set used. The simulator is calibrated to give useful probabilities and comparisons to actuals; further gains typically require richer, more timely features (injuries/lineups/markets) and ensembling.

---
## Features
- Predicts outcomes of playoff games using SVM classification models
- Incorporates [Dean Oliver's Four Factors](https://www.basketball-reference.com/about/factors.html) (eFG%, TOV%, ORB%, DRB%, FT/FGA)
- Adjusts for seeding, head-to-head matchups, momentum, and series context
- Interactive Streamlit siumlator displaying predictions vs. actual results
- Modular project design for scalability and future enhancements
---
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
---
## Data Source
- [Kaggle: Historical NBA Data and Player Box Scores](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores/data)
- Files used: `Games.csv` and `TeamStatistics.csv`
---
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
---
## Results / Demo
- Interactive simulator available via Streamlit  
- Displays predicted vs. actual winners, confidence scores, and team statistics  
- Provides insights into model strengths and weaknesses across different playoff rounds  

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nba-playoffs-game-predictor-cegip2jsppkfijyf5lc7xp.streamlit.app/)

---
## Future Work
- Play-level data integration -> Improve granularity of features by analyzing player actions
- Live data streamining -> Update predictions during ongoing playoff games
- Web API deployment -> Expose predictions for integration into dashboards or external apps
---
**Built with:** Python • Streamlit • scikit-learn • pandas • numpy • joblib • (Matplotlib/Altair) • Jupyter • GitHub • Streamlit Community Cloud

---
## Contact
Email: savannahlevo@gmail.com || [LinkedIn](https://www.linkedin.com/in/savannahlevo/)     


