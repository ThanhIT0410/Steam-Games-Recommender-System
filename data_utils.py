import pandas as pd

def load_data(path="data/processed/recommendations_processed.csv"):
    return pd.read_csv(
        path,
        usecols=["app_id", "date", "is_recommended", "hours", "hours_log", "user_id", "review_id"],
        parse_dates=["date"]
    )

def split_data_full_corpus(df, ratio=0.918):
    # Với tỉ lệ thời gian 0.918 thì sẽ ra được tỉ lệ số lượng ~80%
    df = df.sort_values("date")
    start_date = df["date"].min()
    end_date = df["date"].max()
    pivot_date = start_date + (end_date - start_date) * ratio
    train_set = df[df["date"] < pivot_date].copy()
    test_set = df[df["date"] >= pivot_date].copy()
    return train_set, test_set

def split_data_leave_one_last(df):
    df = df.copy().reset_index(names='original_index')
    test_indices = []

    for _, group in df.groupby("user_id"):
        group_sorted = group.sort_values("date")
        is_rec = group_sorted["is_recommended"].values
        idx = group_sorted.index.values
        for i in range(len(is_rec) - 1, -1, -1):
            if is_rec[i] == 1:
                test_indices.extend(idx[i:])
                break

    test_set = df.loc[test_indices].reset_index(drop=True)
    train_set = df.drop(index=test_indices).reset_index(drop=True)
    test_set = test_set.drop(columns=["original_index"])
    train_set = train_set.drop(columns=["original_index"])
    return train_set, test_set



