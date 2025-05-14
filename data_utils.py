import pandas as pd

def load_data(path="data/processed/recommendations_processed.csv"):
    return pd.read_csv(
        path,
        usecols=["app_id", "date", "is_recommended", "hours", "hours_log", "user_id", "review_id"],
        parse_dates=["date"]
    )

def split_data(df, ratio=0.8):
    df = df.sort_values("date")
    train_size = int(len(df) * ratio)
    train_set = df.iloc[:train_size]
    test_set = df.iloc[train_size:]
    game_list = set(df["app_id"].unique())
    return train_set, test_set, game_list
