import torch
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from data_utils import split_data_leave_one_last

random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MF_svm:
    def __init__(self, evaluation_type, n_factors=192, learning_rate=0.01, n_epochs=30, reg_lambda=5e-5, verbose=True):
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
            interaction_matrix = interaction_matrix * 2 - 1

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

            mask = ~torch.isnan(interaction_matrix)

            for epoch in range(self.n_epochs):
                dot_product = self.user_embedding @ self.game_embedding.T
                pred_result = dot_product + self.user_bias.unsqueeze(1) + self.game_bias.unsqueeze(0) + self.global_bias
                loss = torch.clamp(1 - pred_result[mask]*interaction_matrix[mask], min=0).mean()
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose:
                    val_loss = self.validate(train_set, validate_set)
                    print(f"[MF-SVM] Epoch {epoch+1}/{self.n_epochs}, Train loss: {loss.item():.6f}, {val_loss}")        

        train(train_set, verbose=self.verbose, validate_set=validate_set, tag="Train & Validation")
        train(all_train_data, tag="Retrain")
    
    def predict(self, user_id, game_list):
        if user_id not in self.user_to_idx:
            return torch.full((len(game_list),), -0.1).tolist()
        
        u_idx = self.user_to_idx[user_id]
        u_emb = self.user_embedding[u_idx]
        u_bias = self.user_bias[u_idx]

        valid_game_list = [gid for gid in game_list if gid in self.game_to_idx]
        valid_game_idx = [self.game_to_idx[gid] for gid in valid_game_list]

        if len(valid_game_idx) == 0:
            return torch.full((len(game_list),), -0.1).tolist()

        g_emb = self.game_embedding[valid_game_idx] 
        g_bias = self.game_bias[valid_game_idx]     

        with torch.no_grad():
            scores = torch.matmul(g_emb, u_emb) + g_bias + u_bias + self.global_bias  
            scores_np = scores.cpu().numpy().flatten()

        valid_score_dict = dict(zip(valid_game_list, scores_np.tolist()))
        result = [valid_score_dict.get(gid, -0.1) for gid in game_list]

        return result
    
    def validate(self, train_set, validate_set):
        user_list = validate_set["user_id"]
        game_list = validate_set["app_id"]
        true_labels = torch.tensor(validate_set["is_recommended"].values, dtype=torch.float32, device=self.device)
        true_labels = true_labels * 2 - 1

        user_idx = [self.user_to_idx.get(user_id, -1) for user_id in user_list]
        game_idx = [self.game_to_idx.get(game_id, -1) for game_id in game_list]

        valid_mask = (torch.tensor(user_idx) != -1) & (torch.tensor(game_idx) != -1)

        valid_user_idx = torch.tensor(user_idx, device=self.device)[valid_mask]
        valid_game_idx = torch.tensor(game_idx, device=self.device)[valid_mask]

        pred_scores = torch.full((len(user_list),), -0.1, dtype=torch.float32, device=self.device)
        
        if len(valid_user_idx) > 0:
            user_emb = self.user_embedding[valid_user_idx]
            game_emb = self.game_embedding[valid_game_idx]
            user_bias = self.user_bias[valid_user_idx]
            game_bias = self.game_bias[valid_game_idx]

            dot_product = torch.sum(user_emb * game_emb, dim=1)
            pred_score = dot_product + user_bias + game_bias + self.global_bias
            pred_scores[valid_mask] = pred_score

        loss = torch.clamp(1 - pred_scores*true_labels, min=0).mean()

        # return f"Validation loss: {loss:.6f}"
        
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

