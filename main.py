import pandas as pd
import numpy as np
import recommender


model_path = "saved_models/knn.pth"
model = recommender.ItemKNN()
model.load(model_path)

game_list = list(model.game_to_idx.keys())

user_id = input("Nhập user_id: ")
scores = model.predict(user_id, game_list)

top_10_indices = np.argsort(scores)[::-1][:10]
top_10_game_ids = [game_list[i] for i in top_10_indices]

game_df = pd.read_csv("data/processed/games_processed.csv")
top_games = game_df[game_df["app_id"].isin(top_10_game_ids)]

top_games = top_games.copy()
top_games.loc[:, "order"] = top_games["app_id"].apply(lambda x: top_10_game_ids.index(x))
top_games = top_games.sort_values("order")

print("\nGợi ý top 10 game cho user", user_id)
for i, title in enumerate(top_games["title"].values, start=1):
    print(f"{i}. {title}")
