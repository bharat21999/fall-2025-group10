# Fall 2025 — Group 10: Multi-Armed Bandits for Recommendation

## Short Description
This project compares baseline recommenders (Random and item–item collaborative filtering) with multi-armed bandit approaches (contextual LinUCB and non-contextual UCB1). The pipeline prepares data, performs EDA, runs baselines, constructs a dataset for MAB experiments, runs multiple recommenders on that dataset, and saves comparative results.

---

## Table of Contents

- Project overview
- Repository structure
- Dependencies
- Quick start / run order
- Script descriptions
- Outputs & results
- Evaluation metrics
- Notes & tips
- License & contact

---

## Project Overview

### Pipeline:
1. Prepare raw data.  
2. Run exploratory data analysis (EDA).  
3. Run baseline recommenders to compare popularity-based vs item–item CF.  
4. Prepare an MAB-compatible dataset.  
5. Run contextual MAB (LinUCB) and non-contextual MAB (UCB1), Random, and CF recommenders on the constructed dataset.  
6. Produce a comparative analysis and save outputs under `results/`.

---

## Repository Structure (relevant)

src/
  component/
    data.py                   # Data ingestion / preprocessing (run first)
    EDA.py                    # Exploratory data analysis (run second)
    baseline_popularity.py    # Popularity-based baseline
    baseline_item_item_cf.py  # Item–item collaborative filtering baseline
    main.py                   # Prepares dataset for MAB experiments + runs LinUCB
    mab_recommender.py        # UCB1 on the dataset created by main.py
    random_recommender.py     # Random policy on the MAB dataset
    CF_recommender.py         # CF-based recommender on the MAB dataset
    Comparison.py             # Compares all recommenders and saves analysis

                         
results/                      # EDA outputs, baseline results, MAB results, comparisons
README.md



Aggregates and compares outputs from LinUCB, UCB1, Random, and CF.  
Writes summaries, metrics, and plots into `results/`.

---

## Script Descriptions

- **data.py**: Loads raw data and exports processed dataset for EDA and recommenders.  
- **EDA.py**: Generates summary statistics, plots, and tables describing dataset characteristics.  
- **baseline_recommenders.py**: Popularity-based recommender vs item-item CF comparison using HitRate@k.  
- **main.py**: Creates the MAB-compatible dataset (sessionized), runs LinUCB, and shows outputs.  
- **mab_recommender.py**: Implements non-contextual UCB1 on the dataset produced by main.py.  
- **random_recommender.py**: Baseline random policy on the MAB dataset.  
- **CF_recommender.py**: Collaborative filtering recommender tested on the MAB dataset.  
- **Comparison.py**: Aggregates and compares outputs from recommenders and saves final analysis.

---

## Outputs & Where to Find Them

All results are stored in:



This includes:

- Cleaned datasets  
- EDA outputs  
- Baseline recommender performance  
- MAB dataset and bandit runs  
- Regret curves  
- Comparison plots and metrics  

---

## Evaluation Metrics

### Regret (for bandits)

Regret measures the loss incurred by not choosing the optimal arm.


Where:  
- r*_τ = reward of the optimal arm at time τ  
- r_τ = reward of the chosen arm  

Lower regret indicates better performance.

---

## Notes & Tips

- Ensure correct file paths for all scripts.  
- Run `main.py` before running UCB1, Random, or CF scripts.  
- The FAR-Trans dataset is large; ensure sufficient memory.  
- Use `Comparison.py` for full evaluation across all models.
