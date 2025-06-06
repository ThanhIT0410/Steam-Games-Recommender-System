import torch
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from data_utils import split_data_leave_one_last

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNCF:
    def __init__(self, evaluation_type, n_factors=64, learning_rate=0.0001, n_epochs=35, reg_lambda = 1e-4, batch_size=128, verbose=True):
        self.evaluation_type = evaluation_type

        self.model = None
        self.optimizer = None
        self.criterion = torch.nn.BCEWithLogitsLoss()
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
    
    class CNN(torch.nn.Module):
        def __init__(self, n_factors, n_users, n_games):
            super().__init__()
            self.user_embedding = torch.nn.Embedding(n_users, n_factors)
            self.game_embedding = torch.nn.Embedding(n_games, n_factors)

            self.conv_layers = torch.nn.Sequential(
                torch.nn.Conv2d(1, 8, kernel_size=2, stride=2, padding=0),          # 1x64x64 -> 8x32x32
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=0),         # 8x32x32 -> 16x16x16 
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0),        # 16x16x16 -> 32x8x8
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),        # 32x8x8 -> 64x4x4
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0),       # 64x4x4 -> 128x2x2
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0),      # 128x2x2 -> 256x1x1
                torch.nn.ReLU()
            )

            self.fc = torch.nn.Sequential(
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1)
            )
            
        def forward(self, user_idx, item_idx):
            user_vec = self.user_embedding(user_idx)
            item_vec = self.game_embedding(item_idx)

            outer = torch.bmm(user_vec.unsqueeze(2), item_vec.unsqueeze(1))  
            outer = outer.unsqueeze(1) 
                
            x = self.conv_layers(outer)
            x = x.view(x.size(0), -1)
            out = self.fc(x).view(-1)
            return out

    def fit(self, all_train_data):
        train_set, validate_set = split_data_leave_one_last(all_train_data)

        def train(train_set, verbose=False, validate_set=None, tag="Train"):
            print(f"{tag}...")
            interaction_matrix, _, self.user_to_idx, self.game_to_idx = self._create_matrix(train_set)
            interaction_matrix = torch.tensor(interaction_matrix, dtype=torch.float32, device=self.device)
            mask = ~torch.isnan(interaction_matrix)

            num_users = self.user_to_idx.__len__()
            num_games = self.game_to_idx.__len__()
            self.model = self.CNN(self.n_factors, num_users, num_games).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=float(self.reg_lambda))

            user_list, item_list = torch.meshgrid(
                torch.arange(num_users, device=self.device),
                torch.arange(num_games, device=self.device),
                indexing='ij'
            )

            for epoch in range(self.n_epochs):
                print(f"Epoch {epoch+1}/{self.n_epochs}...")
                self.model.train()

                perm = torch.randperm(num_users)
                user_list = user_list[perm]
                item_list = item_list[perm]
                mask = mask[perm]

                for start in range(0, num_users, self.batch_size):
                    end = min(start + self.batch_size, num_users)
                    user_idx = user_list[start:end][mask[start:end]].flatten()
                    item_idx = item_list[start:end][mask[start:end]].flatten()
                    labels = interaction_matrix[user_idx, item_idx].flatten()

                    self.optimizer.zero_grad()
                    logits = self.model(user_idx, item_idx)
                    loss = self.criterion(logits, labels)
                    loss.backward()
                    self.optimizer.step()

                if verbose:
                    val_loss = self.validate(train_set, validate_set)
                    print(f"[ConvNCF] Epoch {epoch+1}/{self.n_epochs}, Train loss: {loss.item():.6f}, {val_loss}")
                

        train(train_set, verbose=self.verbose, validate_set=validate_set, tag="Train & Validate")
        # train(all_train_data, tag="Retrain")


    def predict(self, user_id, game_list):
        self.model.eval()

        if user_id not in self.user_to_idx:
            return torch.full((len(game_list),), 0.49).tolist()

        u_idx = self.user_to_idx[user_id]

        valid_game_list = [gid for gid in game_list if gid in self.game_to_idx]
        valid_game_idx = [self.game_to_idx[gid] for gid in valid_game_list]

        if len(valid_game_list) == 0:
            return torch.full((len(game_list),), 0.49).tolist()

        u_tensor = torch.tensor([u_idx] * len(valid_game_list), dtype=torch.long, device=self.device)
        g_tensor = torch.tensor(valid_game_idx, dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(u_tensor, g_tensor)
            probs = torch.sigmoid(logits).tolist()

        result = []
        id_to_prob = dict(zip(valid_game_list, probs))
        for gid in game_list:
            result.append(id_to_prob.get(gid, 0.49))

        return result
    
    def validate(self, train_set, validate_set):
        self.model.eval()

        with torch.no_grad():
            user_idx = validate_set["user_id"].map(self.user_to_idx)
            game_idx = validate_set["app_id"].map(self.game_to_idx)
            mask = user_idx.notna() & game_idx.notna()

            if mask.sum() == 0:
                return float("inf")
            
            u = torch.tensor(user_idx[mask].values, dtype=torch.long, device=self.device)
            g = torch.tensor(game_idx[mask].values, dtype=torch.long, device=self.device)
            labels = torch.tensor(validate_set["is_recommended"].values[mask], dtype=torch.float32, device=self.device)

            logits = self.model(u, g)
            loss = self.criterion(logits, labels)
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

            if user_id not in self.user_to_idx or next(iter(pos_game)) not in self.game_to_idx:
                continue

            played = user_played_games.get(user_id, set())
            unplayed = list(game_list - played - pos_game)

            if len(unplayed) < 9:
                continue

            sampled_negatives = random.sample(unplayed, min(100, len(unplayed)))
            candidate_games = list(pos_game) + sampled_negatives
            y_true = [1] + [0] * len(sampled_negatives)
            
            u_tensor = torch.tensor([self.user_to_idx[user_id]] * len(candidate_games), dtype=torch.long, device=self.device)
            g_tensor = torch.tensor([self.game_to_idx[gid] for gid in candidate_games], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = self.model(u_tensor, g_tensor)
                y_score = torch.sigmoid(logits).tolist()

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
            'model_state_dict': self.model.state_dict(),
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

        model.user_to_idx = state['user_to_idx']
        model.game_to_idx = state['game_to_idx']

        n_users = len(model.user_to_idx)
        n_games = len(model.game_to_idx)

        model.model = model.CNN(model.n_factors, n_users, n_games).to(device)
        model.model.load_state_dict(state['model_state_dict'])
        model.model.eval()

        return model


