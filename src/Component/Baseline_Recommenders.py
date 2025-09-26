import numpy as np
import pandas as pd
import math
from collections import defaultdict
from pathlib import Path

RESULTS = Path("/home/ubuntu/Capstone/results")


#------------------- Recommendation based on popularity-------------------------
transaction_df = pd.read_csv('/home/ubuntu/Capstone/src/Component/data/FAR-Trans/transactions.csv')

transaction_df1 = transaction_df[transaction_df["transactionType"].str.lower() == "buy"].copy()

transaction_df1 = transaction_df1.dropna(subset=["customerID", "ISIN", "timestamp"])


transaction_df1["timestamp"] = pd.to_datetime(transaction_df1["timestamp"], errors='coerce')

# Time-based split: choose cutoff at the 80th percentile of timestamps
cutoff = transaction_df1["timestamp"].quantile(0.8)
train = transaction_df1[transaction_df1["timestamp"] <= cutoff].copy()
test = transaction_df1[transaction_df1["timestamp"] > cutoff].copy()

# Filter to users who appear in both train and test to get meaningful evaluation
users_train = set(train["customerID"].unique())
users_test = set(test["customerID"].unique())
eval_users = sorted(list(users_train & users_test))

train_user_items = train.groupby("customerID")["ISIN"].apply(set).to_dict()
item_pop = train.groupby("ISIN").size().sort_values(ascending=False)

# For a simple baseline: recommend top-K popular items not yet seen by the user
K = 20
pop_list = item_pop.index.tolist()

def recommend_popularity(user_id):
    seen = train_user_items.get(user_id, set())
    recs = [i for i in pop_list if i not in seen]
    return recs[:K]

# Prepare ground truth per user - test buys
test_user_items = test.groupby("customerID")["ISIN"].apply(set).to_dict()


# Evaluate HitRate@K and Recall@K
hits = []
recalls = []
per_user_rows = []

for u in eval_users:
    gt = test_user_items.get(u, set())
    if not gt:
        continue
    recs = recommend_popularity(u)
    hit_count = len(set(recs) & gt)
    hits.append(1 if hit_count > 0 else 0)
    recalls.append(hit_count / len(gt))
    per_user_rows.append({"customerID": u, "gt_count": len(gt), "hits_in_k": hit_count, "recall_at_k": hit_count / len(gt)})

hitrate_at_k = float(np.mean(hits)) if hits else 0.0
recall_at_k = float(np.mean(recalls)) if recalls else 0.0

summary = pd.DataFrame({
    "metric": ["users_evaluated", f"hitrate@{K}", f"recall@{K}"],
    "value": [len(per_user_rows), round(hitrate_at_k, 4), round(recall_at_k, 4)]
})


print("Baseline metrics (Popularity):")
print(summary)

#--------------------- item-item Collaborative Filtering----------------------

# Build user->items (train) and item popularity
user_items = {u: set(items) for u, items in train.groupby("customerID")["ISIN"]}
item_pop = train.groupby("ISIN").size().to_dict()


# Co-occurrence counts
cooc = defaultdict(lambda: defaultdict(int))

#to keep the computation light we select length 200 items from user history
MAX_ITEMS_PER_USER_FOR_COOC = 400

for u, items in user_items.items():
    items_list = list(items)
    if len(items_list) > MAX_ITEMS_PER_USER_FOR_COOC:
        items_list = items_list[:MAX_ITEMS_PER_USER_FOR_COOC]
    L = len(items_list)
    for i in range(L):
        a = items_list[i]
        for j in range(i+1, L):
            b = items_list[j]
            cooc[a][b] += 1
            cooc[b][a] += 1



# Compute top neighbors with cosine similarity: sim(a,b) = cooc(a,b)/sqrt(pop[a]*pop[b])
TOP_NEIGHBORS = 200
item_sims = {}

for a, nbrs in cooc.items():
    sims = []
    pop_a = item_pop.get(a, 1)
    for b, c in nbrs.items():
        pop_b = item_pop.get(b, 1)
        sim = c / math.sqrt(pop_a * pop_b)
        sims.append((b, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    item_sims[a] = sims[:TOP_NEIGHBORS]


# Recommend for a user: score candidate j by sum(sim(i,j)) over seen items i
def recommend_itemitem(u, K=20):
    seen = user_items.get(u, set())
    scores = defaultdict(float)
    for i in seen:
        for j, s in item_sims.get(i, []):
            if j in seen:
                continue
            scores[j] += s
    if not scores:  # fallback to popularity
        popular = sorted(item_pop.items(), key=lambda x: x[1], reverse=True)
        return [i for i,_ in popular if i not in seen][:K]
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [j for j, _ in ranked[:K]]

# Evaluate on the same users that are present in the train set
hits2 = []
recalls2 = []
for u in eval_users:
    gt = test_user_items.get(u, set())
    if not gt:
        continue
    recs = recommend_itemitem(u, K=20)
    hit_count = len(set(recs) & gt)
    hits2.append(1 if hit_count > 0 else 0)
    recalls2.append(hit_count / len(gt))

summary2 = pd.DataFrame({
    "metric": ["users_evaluated", "hitrate@10", "recall@10"],
    "value": [len(eval_users), round(float(np.mean(hits2)), 4), round(float(np.mean(recalls2)), 4)]
})

print("Baseline metrics (Item-Item CF):")
print(summary2)


summary.to_csv(RESULTS/"Baseline_metrics_(Popularity).csv", index=False)
summary2.to_csv(RESULTS/"Baseline_metrics_(Item-Item CF).csv", index=False)