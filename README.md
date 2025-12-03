# Financial Asset Recommendation Using Multi-Armed Bandits  
Capstone Project – Fall 2025  
Author: Bharat Khandelwal  
The George Washington University – Data Science Program

## Adaptive Financial Asset Recommendation Through Contextual and Non-Contextual Multi-Armed Bandits

## Abstract
Financial markets are dynamic and uncertain, making asset recommendation a challenging problem that requires adaptive decision-making. Traditional recommender systems such as Collaborative Filtering (CF) struggle in non-stationary financial environments because they depend heavily on user–item interactions.

This project evaluates Multi-Armed Bandit (MAB) approaches for financial asset recommendation using the FAR-Trans dataset (Sanz-Cruzado et al., 2024). We compare Random, Collaborative Filtering, UCB1, and LinUCB algorithms by constructing a structured, weekly snapshot–based dataset suitable for contextual and non-contextual bandit experiments.

Experiments assess regret minimization, reward patterns, and model behavior across different asset subsets. Results show that contextual bandits (LinUCB) perform more consistently when features correlate with observed rewards, while CF baselines struggle with market shifts. Overall, the study demonstrates the potential of bandit-based frameworks for adaptive financial recommendation systems.

## Dataset
The project uses the FAR-Trans dataset, containing anonymized retail investor behavior, asset histories, and transaction records.

### Dataset Highlights
- ~800+ unique assets
- ~29,000 unique customers
- 388k unique transactions
- 253 weekly time steps (Mondays)
- Price data, returns, momentum, volatility
- Customer-level risk (optional scenario)

### Preprocessing Includes
- Asset filtering
- Price merging and feature engineering
- 7-day momentum & 7-day volatility
- Weekly snapshot generation for CMAB
- Optional risk-level enrichment

The final output is a CMAB-ready dataset used to train and test all algorithms.

## Models Implemented

### 1. Multi-Armed Bandits
- UCB1 (Non-contextual)
- LinUCB (Contextual)
- LinUCB + Risk Level (Optional customer risklevel feature)

### 2. Classical Recommenders
- Random Policy
- Item-Item Collaborative Filtering
- Popularity-based baseline (HitRate@k)

Evaluation is based on regret, reward trends, and stability.

## Installation

### Clone the Repository
git clone https://github.com/bharat21999/fall-2025-group10.git

## Usage
### Prepare Raw Dataset, CMAB Dataset & Run LinUCB and other algorithms
python src/main.py (USE_RISKLEVEL=true python main.py #when want to add customer risklevel as a feature)



### Run Exploratory Data Analysis (EDA)
python src/EDA.py


### Compare All Models
python src/Comparison.py (#as you run this it will ask for y/n if you put USE_RISKLEVEL=true in main then put y else n)

All results will appear in the results/ folder.

## Files Overview

### src/Component/
Reusable modules for bandit experiments:
- Data.py – Data ingestion & preprocessing
- environment.py – Bandit environment construction
- model.py – Core LinUCB logic
- mab_recommender.py – UCB1 implementation
- CF_Recommender.py – Item-item CF
- random_recommender.py – Random baseline
- add_risklevel_to_cmab.py – Risk-level augmentation
- utils.py – Shared utilities(for changing configurations change n_customers at the top in utils.py and to change number of assets change min_txn_count=50,
        min_customers=20 in [3/7]select_stable_asset function in main function of utils and put none for both if need all the assets having close prices at all timestamps )

### src/
Main execution scripts:
- main.py – Builds CMAB dataset & runs LinUCB and other algorithms
- EDA.py – Exploratory data analysis
- Comparison.py – Aggregates outputs & generates comparisons

### results/
Contains:
- EDA outputs
- Baseline results
- Bandit logs & regret curves
- Final comparison metrics & plots

## Evaluation Metrics

### Regret (Primary Bandit Metric)
Regret measures how much reward is lost by not choosing the optimal arm at every step.

R(T) = Σ (r*τ – rτ)

Where:
- r*τ = reward of optimal arm
- rτ = reward of chosen arm

Lower regret = better sequential decision-making.

### Additional Metrics
- HitRate@k (for CF baselines) #generated only if you run Baseline_Recommenders.py

## Notes
- Ensure file paths are correct before running.
- Always run src/main.py first then EDA.py for EDA and Comparison.py for comparison
- FAR-Trans is large; adequate RAM recommended.
- Risk-level scenario is optional and requires running add_risklevel_to_cmab.py.

## License
This project is licensed under the MIT License.


