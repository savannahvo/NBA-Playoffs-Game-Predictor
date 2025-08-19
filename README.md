# NBA Playoff Game Predictor - Machine Learning Classifier
---
![NBA Playoffs Court](https://github.com/savannahvo/NBA-Playoffs-Game-Predictor/blob/main/2025%20NBA%20Playoffs%20Court.jpeg?raw=true)
---
## Project Summary
The goal of this project was to predict NBA playoff game outcomes using machine learning models trained on advanced team statistics, series context, and matchup history. Multiple models were tested (Logistic Regression, Random Forest, Gradient Boosting, and Support Vector Machine) and evaluated on accuracy, precision, recall, and F1 Score.

All modeled performed reasonbly well, but none reached perfect accuracy given the unpredictability of playoff basketball. The best performing model was the **Support Vector Machine (SVM)**, which achieved:
- Accuracy: 59%
- F1 Score: 69%
- Precision: 61%
- Recall 79%

While Random Forest and Gradient Boosting achieved slightly better raw accuracy (up to ~57%), they underperformed on F1 Score. Since playoff predicition requires balancing false positives and false negatives (e.g, not just predicting favorites but capturing potential upsets), **F1 Score was chosen as the primary selection metric.** This made SVM the most suitable model overall. 

### Why 2025-2025 Seasons?
I selected the 2015-2025 seasons to capture the modern NBA era. Around 2014-2015, the league underwent a major tactical shift that significantly changed how games are played and decided:  
  1. ** Rise of the 3-Point Era**
     - The Golden State Warriors, led by **Stephen Curry**, revolutionized basketball by showing how the 3-point shot could be the central weapon of an offense.
     - Curry's unprecedented shooting range and efficiency reshaped defenses and forced every team to adjust.
     - League-wide, average 3PA per game jumped from ~22 (2014) to ~27 (2015) and has continued to rise, with some teams now attempting 40+ threes per game.
     - This dramatically affected Effective FG% (eFG%), one of the key metrics in my model.
  2. **Pace and Spacing Revolution**
     - Teams emphasized pace, spacing, and ball movement over isolation-heavy play.
     - The Warriors' "Strength in Numbers" philosophy inspired a league-wide adoption of small-ball and floor spacing.
     - Possession stats (like TOV% and ORB%) became more critical indicators of success in this style of play.
  3. **Consistency with Advanced Metrics**
     - The core features in my project (eFG%, TOV%, ORB%, DRB%, FT/FGA) became more stable and predictive once the league transitioned into this modern style
     - Using older seasons (1990s or early 2000s) would mix two very different styles of basketball, making the model less reliable.
  4. **Data Reliability and Availability**
     - Play-by-play and advanced team statistics are more consistently tracked and reported from 2015 onwards.
     - Ensures clean, standardized datasets for modeling.
  5. **Future-Proofing**
     - By ending at 2025, the project incorporates the most up-to-date trends and allows evaluation against the current playoffs.

### Why I Choose These Advanced Metrics
My model relies on four essential team-level metrics: Effective Field Goal Percentage (eFG%), Turnover Percentage (TOV%), Offensive and Defensive Rebounding Percentage (ORB% and DRB%), and Free Throw Rate (FT/FGA) because they encapsulate the core principles of winning basketball, as identified by Dean Oliver's "Four Factors". This framework is well-respected in analytics for its predictive strength and interpretability:
  1. **Shooting (eFG%) - ~40% weight**
     - Reflects shooting efficiency, adjusting for the greater value of 3-pointers.
     - It's calculated as: `eFG% = (FGM + 0.5 × 3PM) ÷ FGA`
     - According to Basketball Reference, shooting is the most critical factor in winning games.
  2. **Turnovers (TOV%) - ~25% weight**
     - Measures ball security, indicating how often a team gives up possessions.
     - It's calculated as: `TOV% = TOV ÷ (FGA + 0.44 × FTA + TOV)`
     - As a pace-independent metric, it helps explain how well teams protect the ball.
  3. **Rebounding (ORB%/DRB%) - ~20% weight**
     - Captures a team's ability to create extra possessions and limit opponent second-chance points.
     - It's calculated as: `ORB% = ORB ÷ (ORB + Opponent DRB)`
     - Rebounding stands out as one of the most consistent metrics across seasons.
  4. **Free Throw Rate (FT/FGA) - ~15% weight**
     - Signifies how often a team gets to the line relative to field goal attempts.
     - It's calculated as: `FT/FGA = FTM ÷ FGA`
     - This captures offensive aggression and efficiency; teams that foul less and shoot accurately often convert more points per possession.
