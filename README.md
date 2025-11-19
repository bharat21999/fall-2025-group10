Fall 2025 — Group 10: Multi-Armed Bandits for Recommendation
Short description This project compares baseline recommenders (Random and item–item collaborative filtering) with multi-armed bandit approaches (contextual LinUCB and non-contextual UCB1). The pipeline prepares data, performs EDA, runs baselines, constructs a dataset for MAB experiments, runs multiple recommenders on that dataset, and saves comparative results.

Table of contents

Project overview
Repository structure
Dependencies
Quick start / run order
Script descriptions
Outputs & results
Evaluation metrics
Notes & tips
License & contact
Project overview Pipeline:

Prepare raw data.
Run exploratory data analysis (EDA).
Run baseline recommenders to compare popularity-based vs item–item CF.
Prepare an MAB-compatible dataset.
Run contextual MAB (LinUCB) and non-contextual MAB (UCB1), random, and CF recommenders on the constructed dataset.
Produce a comparative analysis and save outputs under results/.
Repository structure (relevant)

src/
component/
data.py # Data ingestion / preprocessing (run first)
EDA.py # Exploratory data analysis (run second)
baseline_popularity.py # Popularity-based baseline (example name)
baseline_item_item_cf.py # Item-item collaborative filtering baseline (example name)
main.py # Prepares dataset for MAB experiments and runs LinUCB
mab_recommender.py # Runs UCB1 on the dataset created by main.py
random_recommender.py # Random policy on the new MAB dataset
CF_recommender.py # CF-based recommender on the new MAB dataset
Comparison.py # Compares results from all recommenders and saves analysis
data/ # (optional) raw/processed datasets
results/ # EDA outputs, baseline results, MAB results, comparisons
README.md
Dependencies (example)

Python 3.8+
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
joblib
tqdm
Install

Create and activate virtual environment:

python -m venv .venv
source .venv/bin/activate # macOS / Linux
.venv\Scripts\activate # Windows
Install dependencies:

pip install -r requirements.txt (If you don't have requirements.txt yet, create one: pip freeze > requirements.txt)
Quick start — run order and commands Run the scripts in this order (paths assume repo root):

Prepare the dataset python src/component/data.py

Generates cleaned/processed dataset used downstream.
Exploratory Data Analysis (EDA) python src/component/EDA.py

Produces plots/tables saved to results/.
Run baseline recommenders (popularity vs item-item CF) python src/component/baseline_popularity.py python src/component/baseline_item_item_cf.py

Baseline results saved to results/.
Prepare MAB dataset & run contextual LinUCB python src/component/main.py

main.py prepares the MAB dataset and runs contextual LinUCB. Saves dataset and LinUCB outputs to results/.
Run UCB1 on the new dataset python src/component/mab_recommender.py

Run Random and CF recommenders on the same new dataset python src/component/random_recommender.py python src/component/CF_recommender.py

Comparative analysis python src/component/Comparison.py

Aggregates and compares outputs from LinUCB, UCB1, Random, and CF; writes summary, metrics, and plots into results/.
Script descriptions

data.py: Load raw data export processed dataset for EDA and recommenders.
EDA.py: Generate summary statistics, plots, and tables describing dataset characteristics.
baseline_recommenders.py: Popularity-based recommender vs item-item CF based recommender comparison based on Hitrate@k.
main.py: Create the MAB-compatible dataset (sessionized), run LinUCB, and show outputs.
mab_recommender.py: Implement non-contextual UCB1 on the dataset produced by main.py.
random_recommender.py: Baseline random policy on the MAB dataset.
CF_recommender.py: Collaborative filtering recommender tested on the MAB dataset.
Comparison.py: Aggregate and compare outputs from recommenders and save final analysis.
Outputs & where to find them


Evaluation metrics (examples)

Regret (for bandits)


