
# Capstone Proposal
## FAR-Trans+: Deep Sequential & Graph-Based Financial Asset Recommendation with Risk-Aware, Fair, and Time-Aware Evaluation
### Proposed by: Bharat Khandelwal
#### Email: 
#### Advisor: Amir Jafari
#### The George Washington University, Washington DC  
#### Data Science Program


## 1 Objective:  

            Build on the FAR-Trans dataset to develop improved financial asset recommendation models using
            (1) deep sequential architectures, (2) graph neural networks, and (3) a hybrid pipeline with LLM
            re-ranking/explanations. Complement algorithmic gains with a stronger evaluation framework that
            jointly measures return, risk, robustness over time, and fairness across investor segments.
            

## 2 Dataset:  

            Primary: FAR-Trans dataset (2018–2022) containing investor profiles, transactions, and asset pricing.
            Optional Extension: Synthetic macro features (e.g., yields, volatility) and investor preference tags
            aligned monthly to FAR-Trans timelines to support risk-aware and time-aware evaluation.
            

## 3 Rationale:  

            Prior FAR-Trans benchmarks emphasize ROI@K and nDCG@K but underweight sequential purchase dynamics,
            investor-asset network structure, and risk/fairness considerations. This project aims to (i) capture
            temporal patterns with sequence models, (ii) exploit relational structure via GNNs, and (iii) ensure
            practical deployability by evaluating risk-adjusted performance, stability across market regimes, and
            equitable outcomes across investor segments.
            

## 4 Approach:  

            [Better Algorithms: Deep Sequential, GNNs, Hybrid with LLM]
            - Deep Sequential Models:
              • SASRec / GRU4Rec / BERT4Rec variants to encode investor transaction sequences (buy/sell/wallet share),
                with position encodings and masking for next-asset prediction.
              • Side features: investor risk tier, capacity, recent returns/volatility, and asset metadata embeddings.

            - Graph Neural Networks (GNNs):
              • Build a heterogeneous graph: nodes = {investors, assets}, edges = {transaction, co-ownership, similarity}.
              • Models: GraphSAGE / GAT / LightGCN for collaborative filtering on the investor–asset graph.
              • Edge weights incorporate recency and intensity; asset–asset edges from price correlation/sector proximity.

            - Hybrid with LLM:
              • Two-tower or sequential/GNN candidate generator → LLM re-ranker with concise, controllable prompts.
              • Use LLM for explanation generation (“why recommended”) and optional constraint hints (e.g., risk profile).
              • Hard constraints applied post re-rank (eligibility, diversification, max volatility).

            [Data Processing]
            - Train/val/test via time-aware splits (rolling windows). Normalize IDs and align monthly calendars.
            - Optional macro feature join (synthetic or public proxies) for regime-aware training/analysis.

            [Engineering]
            - Candidate generation (top-N) from sequential or graph model → constraint-aware re-rank.
            - Model management with experiment tracking; reproducible configs and seeds.
            

## 5 Timeline:  

            Week 1: Data ingestion, cleaning, time-aware splits, feature baselines.
            Weeks 2–3: Implement SASRec/BERT4Rec baselines; hyperparameter search; initial ROI/nDCG results.
            Weeks 4–5: Construct investor–asset graph; implement GraphSAGE/GAT/LightGCN; candidate fusion.
            Week 6: Hybrid pipeline with LLM re-ranking and explanation generation; constraint layer.
            Week 7: Risk-aware, fairness, and time-aware evaluation; rolling backtests and regime analysis.
            Week 8: Ablations, sensitivity, and final report/dashboard; write-up of improvements vs. FAR baselines.
            


## 6 Expected Number Students:  

            1 student
            

## 7 Possible Issues:  

            - Class imbalance and cold-start assets/investors may limit generalization—mitigate with meta-embeddings and
              side features. 
            - Overfitting to specific market regimes—use rolling validation and regularization.
            - LLM costs/latency for re-ranking/explanations—batching and caching required.
            - Fairness definitions in finance are context-dependent—report multiple complementary metrics.
            


## Contact
- Author: Bharat Khandelwal
- Email: [](mailto:ajafari@gwu.edu)
