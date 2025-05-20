import numpy as np
import pandas as pd
import torch

device = torch.device("cpu")

class ItemKNN:
    def __init__(self, evaluation_type, k=25):
        self.evaluation_type = evaluation_type

        self.interaction_matrix = None
        self.sim_matrix = None
        self.user_to_idx = None
        self.game_to_idx = None
        self.device = device

        self.k = k

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

    def fit(self, train_set):
        print("Fitting model...")
        self.interaction_matrix, _, self.user_to_idx, self.game_to_idx = self._create_matrix(train_set)
        X = torch.tensor(self.interaction_matrix, dtype=torch.float32, device=self.device)
        X = torch.nan_to_num(X, nan=0.0)
        norms = torch.norm(X, dim=0, keepdim=True) + 1e-8
        self.sim_matrix = torch.matmul(X.T, X) / (norms.T @ norms)     

    def predict(self, user_id, game_list):
        if user_id not in self.user_to_idx:
            return [0.49] * len(game_list)

        user_idx = self.user_to_idx[user_id]
        user_vector = torch.tensor(self.interaction_matrix[user_idx], dtype=torch.float32, device=self.device)
        rated_mask = ~torch.isnan(user_vector)
        rated_items = torch.nonzero(rated_mask, as_tuple=True)[0]

        if len(rated_items) == 0:
            return [0.49] * len(game_list)

        ratings = user_vector[rated_items]

        game_indices = torch.tensor(
            [self.game_to_idx.get(g, -1) for g in game_list], device=self.device
        )

        valid_mask = game_indices != -1
        valid_game_indices = game_indices[valid_mask]

        results = torch.full((len(game_list),), 0.49, device=self.device)

        if len(valid_game_indices) == 0:
            return results.tolist()

        sim_submatrix = self.sim_matrix[valid_game_indices][:, rated_items] 

        if self.k < sim_submatrix.shape[1]:
            topk_sims, topk_idx = torch.topk(sim_submatrix, self.k, dim=1)
            topk_ratings = ratings[topk_idx]
        else:
            topk_sims = sim_submatrix
            topk_ratings = ratings.unsqueeze(0).expand(sim_submatrix.shape[0], -1)

        numerators = (topk_sims * topk_ratings).sum(dim=1)
        denominators = topk_sims.sum(dim=1) + 1e-8
        scores = numerators / denominators 

        results[valid_mask] = scores
        return results.tolist()
    
    def save(self, path):
        save_dict = {
            "interaction_matrix": torch.tensor(self.interaction_matrix),
            "sim_matrix": self.sim_matrix,
            "user_to_idx": self.user_to_idx,
            "game_to_idx": self.game_to_idx,
            "evaluation_type": self.evaluation_type,
            "k": self.k
        }
        torch.save(save_dict, path)

    @classmethod
    def load(cls, path):
        save_dict = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        model = cls(
            k=save_dict["k"],
            evaluation_type=save_dict["evaluation_type"]
        )
        model.interaction_matrix = save_dict["interaction_matrix"].numpy()
        model.sim_matrix = save_dict["sim_matrix"]
        model.user_to_idx = save_dict["user_to_idx"]
        model.game_to_idx = save_dict["game_to_idx"]
        return model
    
        

