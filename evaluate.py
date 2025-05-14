import random
import numpy as np
from data_utils import load_data, split_data
from collections import defaultdict
import random
import sklearn.metrics as metrics
from recommender import ItemKNN


random.seed(42)
np.random.seed(42)

recommendations = load_data()
train_set, test_set, game_list = split_data(recommendations)


# Các metric đánh giá
def classification_metrics(y_true, y_score):
    accuracy = metrics.accuracy_score(y_true, [int(p >= 0.5) for p in y_score])
    try:
        auc = metrics.roc_auc_score(y_true, y_score)
    except ValueError:
        auc = 0.0
    return accuracy, auc

def precision_at_k(y_true, k):
    return np.sum(y_true[:k]) / k

def recall_at_k(y_true, y_all, k):
    return np.sum(y_true[:k]) / np.sum(y_all) if np.sum(y_all) > 0 else 0.0

def ndcg_at_k(y_true, k):
    dcg = np.sum(y_true[:k] / np.log2(np.arange(2, k + 2)))
    ideal = np.sort(y_true)[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, k + 2)))
    return dcg / idcg if idcg > 0 else 0.0

def hitrate_at_k(y_true, k):
    return 1.0 if np.sum(y_true[:k]) > 0 else 0.0


# Đánh giá khả năng phân loại
def evaluate_classification(model, test_set):
    print("Evaluating classification metrics...")
    user_game_dict = defaultdict(list)
    user_label_dict = defaultdict(list)

    for row in test_set.itertuples(index=False):
        user_game_dict[row.user_id].append(row.app_id)
        user_label_dict[row.user_id].append(row.is_recommended)

    y_true, y_score = [], []

    for user_id in user_game_dict:
        game_list = user_game_dict[user_id]
        labels = user_label_dict[user_id]
        preds = model.predict(user_id, game_list)
        y_true.extend(labels)
        y_score.extend(preds)

    accuracy, auc = classification_metrics(y_true, y_score)
    return {"Accuracy": accuracy, "AUC": auc}


# Đánh giá khả năng xếp hạng - dựa trên full dataset
def evaluate_ranking_full_corpus(model, train_set, test_set, game_list):
    print("Evaluating ranking metrics on full corpus...")
    user_played_games = defaultdict(set)
    for user_id, game_id in zip(train_set["user_id"].to_numpy(), train_set["app_id"].to_numpy()):
        user_played_games[user_id].add(game_id)
    
    user_test = defaultdict(list)
    for user_id, game_id, label in zip(test_set["user_id"], test_set["app_id"], test_set["is_recommended"]):
        user_test[user_id].append((game_id, label))

    precision_5_results, precision_10_results = [], []
    recall_5_results, recall_10_results = [], []
    ndcg_5_results, ndcg_10_results = [], []
    hitrate_5_results, hitrate_10_results = [], []

    for user_id, interactions in user_test.items():
        played = user_played_games.get(user_id, set())
        unplayed = list(game_list - played)

        if len(unplayed) == 0:
            continue

        label_dict = dict(interactions)
        y_true = [label_dict.get(game_id, 0) for game_id in unplayed]
        y_score = model.predict(user_id, unplayed)

        if np.sum(y_true) == 0:
            continue

        sorted_indices = np.argsort(-np.array(y_score))
        y_true_sorted = np.array(y_true)[sorted_indices]

        precision_5_results.append(precision_at_k(y_true_sorted, 5))
        precision_10_results.append(precision_at_k(y_true_sorted, 10))
        recall_5_results.append(recall_at_k(y_true_sorted, y_true, 5))
        recall_10_results.append(recall_at_k(y_true_sorted, y_true, 10))
        ndcg_5_results.append(ndcg_at_k(y_true_sorted, 5))
        ndcg_10_results.append(ndcg_at_k(y_true_sorted, 10))
        hitrate_5_results.append(hitrate_at_k(y_true_sorted, 5))
        hitrate_10_results.append(hitrate_at_k(y_true_sorted, 10))

    return {
        "Precision@5 (full-corpus)": float(np.mean(precision_5_results)) if precision_5_results else 0.0,
        "Precision@10 (full-corpus)": float(np.mean(precision_10_results)) if precision_10_results else 0.0,
        "Recall@5 (full-corpus)": float(np.mean(recall_5_results)) if recall_5_results else 0.0,
        "Recall@10 (full-corpus)": float(np.mean(recall_10_results)) if recall_10_results else 0.0,
        "NDCG@5 (full-corpus)": float(np.mean(ndcg_5_results)) if ndcg_5_results else 0.0,
        "NDCG@10 (full-corpus)": float(np.mean(ndcg_10_results)) if ndcg_10_results else 0.0,
        "HitRate@5 (full-corpus)": float(np.mean(hitrate_5_results)) if hitrate_5_results else 0.0,
        "HitRate@10 (full-corpus)": float(np.mean(hitrate_10_results)) if hitrate_10_results else 0.0,
    }


# Đánh giá khả năng xếp hạng - dựa trên sampled dataset 100 mẫu cho mỗi user
def evaluate_ranking_sampled(model, train_set, test_set, game_list):
    print("Evaluating ranking metrics on sampled data...")
    user_played_games = defaultdict(set)
    for user_id, game_id in zip(train_set["user_id"].to_numpy(), train_set["app_id"].to_numpy()):
        user_played_games[user_id].add(game_id)

    user_test_dict = defaultdict(list)
    for user_id, game_id, label in zip(test_set["user_id"], test_set["app_id"], test_set["is_recommended"]):
        user_test_dict[user_id].append((game_id, label))

    ndcg_5_results, ndcg_10_results = [], []
    hitrate_5_results, hitrate_10_results = [], []

    for user_id, interactions in user_test_dict.items():
        latest_pos = next((x for x in reversed(interactions) if x[1] == 1), None)
        if latest_pos is None:
            continue
        pos_game = latest_pos[0]

        played = user_played_games.get(user_id, set())
        unplayed = list(game_list - played - {pos_game})

        sampled_negatives = random.sample(unplayed, min(100, len(unplayed)))
        neg_scores = model.predict(user_id, sampled_negatives)

        scores = model.predict(user_id, [pos_game]) + neg_scores
        labels = [1] + [0] * len(sampled_negatives)

        sorted_indices = np.argsort(-np.array(scores))
        y_true_sorted = np.array(labels)[sorted_indices]

        ndcg_5_results.append(ndcg_at_k(y_true_sorted, 5))
        ndcg_10_results.append(ndcg_at_k(y_true_sorted, 10))
        hitrate_5_results.append(hitrate_at_k(y_true_sorted, 5))
        hitrate_10_results.append(hitrate_at_k(y_true_sorted, 10))

    return {
        "NDCG@5 (sampled)": float(np.mean(ndcg_5_results)) if ndcg_5_results else 0.0,
        "NDCG@10 (sampled)": float(np.mean(ndcg_10_results)) if ndcg_10_results else 0.0,
        "HitRate@5 (sampled)": float(np.mean(hitrate_5_results)) if hitrate_5_results else 0.0,
        "HitRate@10 (sampled)": float(np.mean(hitrate_10_results)) if hitrate_10_results else 0.0,
    }


# Load model
model = ItemKNN()
model.load("saved_models/knn.pth")

# Đánh giá classification
classification_result = evaluate_classification(model, test_set)
print("\n--- Classification Metrics ---")
for metric, value in classification_result.items():
    print(f"{metric}: {value:.4f}")

# Đánh giá full corpus
full_corpus_result = evaluate_ranking_full_corpus(model, train_set, test_set, game_list)
print("\n--- Full Corpus Ranking Metrics ---")
for metric, value in full_corpus_result.items():
    print(f"{metric}: {value:.4f}")

# Đánh giá sampled negative
sampled_result = evaluate_ranking_sampled(model, train_set, test_set, game_list)
print("\n--- Sampled Ranking Metrics ---")
for metric, value in sampled_result.items():
    print(f"{metric}: {value:.4f}")