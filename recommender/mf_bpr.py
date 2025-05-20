import torch
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from data_utils import split_data_leave_one_last

random.seed(42) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MF_bpr:
    def __init__(self, evaluation_type, n_factors=256, learning_rate=0.01, n_epochs=30, reg_lambda=5e-5, batch_size=2048, verbose=True):
        self.evaluation_type = evaluation_type

        self.user_embedding = None
        self.game_embedding = None
        self.user_bias = None
        self.game_bias = None
        self.global_bias = None
        self.user_to_idx = None
        self.game_to_idx = None
        self.device = device

        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.reg_lambda = reg_lambda
        self.batch_size = batch_size
        self.verbose = verbose

    def _create_matrix(self, train_set: pd.DataFrame, fill_value=np.nan):
        user_id = train_set["user_id"].unique()
        game_id = train_set["app_id"].unique()

        user_to_idx = {user: i for i, user in enumerate(user_id)}
        game_to_idx = {game: i for i, game in enumerate(game_id)}

        m_is_recommended = train_set.pivot_table(
            index=train_set["user_id"].map(user_to_idx),
            columns=train_set["app_id"].map(game_to_idx),
            values="is_recommended", fill_value=fill_value
        ).values

        m_hours_log = train_set.pivot_table(
            index=train_set["user_id"].map(user_to_idx),
            columns=train_set["app_id"].map(game_to_idx),
            values="hours_log", fill_value=fill_value
        ).values

        return m_is_recommended, m_hours_log, user_to_idx, game_to_idx

    def fit(self, all_train_data):
        train_set, validate_set = split_data_leave_one_last(all_train_data)

        def train(train_set, verbose=False, validate_set=None, tag="Train"):
            print(f"{tag}...")
            interaction_matrix, _, self.user_to_idx, self.game_to_idx = self._create_matrix(train_set)
            interaction_matrix = torch.tensor(interaction_matrix, dtype=torch.float32, device=self.device)

            self.user_embedding = torch.nn.Parameter(torch.empty((len(self.user_to_idx), self.n_factors), device=self.device))
            self.game_embedding = torch.nn.Parameter(torch.empty((len(self.game_to_idx), self.n_factors), device=self.device))

            torch.nn.init.normal_(self.user_embedding, mean=0.0, std=0.01)
            torch.nn.init.normal_(self.game_embedding, mean=0.0, std=0.01)

            self.user_embedding.requires_grad_(True)
            self.game_embedding.requires_grad_(True)

            self.user_bias = torch.nn.Parameter(torch.zeros(len(self.user_to_idx), device=self.device, requires_grad=True))
            self.game_bias = torch.nn.Parameter(torch.zeros(len(self.game_to_idx), device=self.device, requires_grad=True))
            self.global_bias = torch.tensor(torch.nanmean(interaction_matrix).item(), device=self.device)

            optimizer = torch.optim.Adam([
                {'params': [self.user_embedding, self.game_embedding], 'weight_decay': float(self.reg_lambda)},
                {'params': [self.user_bias, self.game_bias], 'weight_decay': 0}
            ], lr=self.learning_rate)

            user_pos_items = defaultdict(set)
            user_neg_items = defaultdict(set)
            for u, g, label in train_set[["user_id", "app_id", "is_recommended"]].values:
                if label == 1:
                    user_pos_items[u].add(g)
                else:
                    user_neg_items[u].add(g)

            user_list = list(self.user_to_idx.keys())

            for epoch in range(self.n_epochs):
                total_loss = 0
                batch_count = 0
                random.shuffle(user_list)

                for start in range(0, len(user_list), self.batch_size):
                    batch_users = user_list[start:start + self.batch_size]
                    user_idx, pos_idx, neg_idx = [], [], []

                    for u in batch_users:
                        pos_games = list(user_pos_items[u])
                        neg_games = list(user_neg_items[u])

                        if not pos_games or not neg_games:
                            continue

                        u_idx = self.user_to_idx[u]
                        for pos in pos_games:
                            pos_id = self.game_to_idx[pos]
                            for neg in neg_games:
                                neg_id = self.game_to_idx[neg]
                                user_idx.append(u_idx)
                                pos_idx.append(pos_id)
                                neg_idx.append(neg_id)

                    if not user_idx:
                        continue

                    user_idx_tensor = torch.tensor(user_idx, dtype=torch.long, device=self.device)
                    pos_idx_tensor = torch.tensor(pos_idx, dtype=torch.long, device=self.device)
                    neg_idx_tensor = torch.tensor(neg_idx, dtype=torch.long, device=self.device)

                    user_emb = self.user_embedding[user_idx_tensor]
                    pos_emb = self.game_embedding[pos_idx_tensor]
                    neg_emb = self.game_embedding[neg_idx_tensor]

                    user_bias = self.user_bias[user_idx_tensor]
                    pos_bias = self.game_bias[pos_idx_tensor]
                    neg_bias = self.game_bias[neg_idx_tensor]

                    pos_score = (user_emb * pos_emb).sum(dim=1) + user_bias + pos_bias + self.global_bias
                    neg_score = (user_emb * neg_emb).sum(dim=1) + user_bias + neg_bias + self.global_bias

                    loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batch_count += 1

                if verbose:
                    val_loss = self.validate(train_set, validate_set)
                    train_loss = total_loss / batch_count
                    print(f"[MF-BPR] Epoch {epoch+1}/{self.n_epochs}, Train loss: {train_loss:.6f}, {val_loss}")

        # train(train_set, verbose=self.verbose, validate_set=validate_set, tag="Train & Validate")
        train(all_train_data, tag="Retrain")
    
    def predict(self, user_id, game_list):
        if user_id not in self.user_to_idx:
            return torch.full((len(game_list),), 0.49).tolist()

        u_idx = self.user_to_idx[user_id]
        u_emb = self.user_embedding[u_idx]
        u_bias = self.user_bias[u_idx]

        valid_game_list = [gid for gid in game_list if gid in self.game_to_idx]
        valid_game_idx = [self.game_to_idx[gid] for gid in valid_game_list]

        g_emb = self.game_embedding[valid_game_idx]      
        g_bias = self.game_bias[valid_game_idx]    

        with torch.no_grad():
            scores = torch.matmul(g_emb, u_emb) + g_bias + u_bias + self.global_bias
            probs = torch.sigmoid(scores)                                           

        result = []
        valid_id_to_score = dict(zip(valid_game_list, probs.tolist()))
        for gid in game_list:
            result.append(valid_id_to_score.get(gid, 0.49))

        return result
        
    def validate(self, train_set, validate_set):
        user_pos_items = defaultdict(set)
        user_neg_items = defaultdict(set)

        for u, g, label in validate_set[["user_id", "app_id", "is_recommended"]].values:
            if u in self.user_to_idx and g in self.game_to_idx:
                if label == 1:
                    user_pos_items[u].add(g)
        
        for u, g, label in train_set[["user_id", "app_id", "is_recommended"]].values:
            if label == 0:
                user_neg_items[u].add(g)
                
        user_idx, pos_idx, neg_idx = [], [], []

        for u in user_pos_items:
            if u not in user_neg_items or len(user_neg_items[u]) == 0:
                continue
            uid = self.user_to_idx[u]
            pos_ids = [self.game_to_idx[g] for g in user_pos_items[u]]

            for p in pos_ids:
                sampled_negatives = random.sample(list(user_neg_items[u]), min(10, len(user_neg_items[u])))
                neg_ids = [self.game_to_idx[g] for g in sampled_negatives]
                for n in neg_ids:
                    user_idx.append(uid)
                    pos_idx.append(p)
                    neg_idx.append(n)

        if not user_idx:
            return float("nan")

        user_idx = torch.tensor(user_idx, dtype=torch.long, device=self.device)
        pos_idx = torch.tensor(pos_idx, dtype=torch.long, device=self.device)
        neg_idx = torch.tensor(neg_idx, dtype=torch.long, device=self.device)

        with torch.no_grad():
            user_emb = self.user_embedding[user_idx]
            pos_emb = self.game_embedding[pos_idx]
            neg_emb = self.game_embedding[neg_idx]

            user_bias = self.user_bias[user_idx]
            pos_bias = self.game_bias[pos_idx]
            neg_bias = self.game_bias[neg_idx]

            pos_score = (user_emb * pos_emb).sum(dim=1) + user_bias + pos_bias + self.global_bias
            neg_score = (user_emb * neg_emb).sum(dim=1) + user_bias + neg_bias + self.global_bias

            loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean().item()
        # return f"Validate loss: {loss:.6f}"

    
        # Tính NDCG, Hit Rate trên validate_set
        game_list = set(self.game_to_idx.keys())

        user_played_games = defaultdict(set)
        for user_id, game_id in zip(train_set["user_id"].to_numpy(), train_set["app_id"].to_numpy()):
            user_played_games[user_id].add(game_id)

        ndcg_10_results = []
        hitrate_10_results = []

        for user_id, interactions in validate_set.groupby("user_id"):
            pos_game = set(interactions[interactions["is_recommended"] == 1]["app_id"])

            played = user_played_games.get(user_id, set())
            unplayed = list(game_list - played - pos_game)

            if len(unplayed) < 9:
                continue

            sampled_negatives = random.sample(unplayed, min(100, len(unplayed)))
            candidate_games = list(pos_game) + sampled_negatives
            y_true = [1] + [0] * len(sampled_negatives)
            y_score = self.predict(user_id, candidate_games)
            y_true_sorted = np.array(y_true)[np.argsort(-np.array(y_score))]
            
            ndcg_10_results.append(
                (np.sum(y_true_sorted[:10] / np.log2(np.arange(2, 12))) / np.sum(sorted(y_true_sorted[:10], reverse=True) / np.log2(np.arange(2, 12))))
                if np.sum(y_true_sorted[:10]) > 0 else 0.0
            )
            hitrate_10_results.append(float(np.any(y_true_sorted[:10])))

        return f"Validate loss: {loss:.6f}, NDCG@10: {np.mean(ndcg_10_results):.6f}, HitRate@10: {np.mean(hitrate_10_results):.6f}"
    
    
    def save(self, path):
        state = {
            'evaluation_type': self.evaluation_type,
            'user_embedding': self.user_embedding.detach().cpu(),
            'game_embedding': self.game_embedding.detach().cpu(),
            'user_bias': self.user_bias.detach().cpu(),
            'game_bias': self.game_bias.detach().cpu(),
            'global_bias': self.global_bias.detach().cpu(),
            'user_to_idx': self.user_to_idx,
            'game_to_idx': self.game_to_idx,
            'n_factors': self.n_factors,
            'learning_rate': self.learning_rate,
            'n_epochs': self.n_epochs,
            'reg_lambda': self.reg_lambda,
            'batch_size': self.batch_size
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path, map_location=device, weights_only=False)

        model = cls(
            evaluation_type=state['evaluation_type'],
            n_factors=state['n_factors'],
            learning_rate=state['learning_rate'],
            n_epochs=state['n_epochs'],
            reg_lambda=state['reg_lambda'],
            batch_size=state['batch_size'],
            verbose=False
        )

        model.user_embedding = state['user_embedding'].to(device)
        model.game_embedding = state['game_embedding'].to(device)
        model.user_bias = state['user_bias'].to(device)
        model.game_bias = state['game_bias'].to(device)
        model.global_bias = state['global_bias'].to(device)

        model.user_to_idx = state['user_to_idx']
        model.game_to_idx = state['game_to_idx']
        model.device = device

        return model



