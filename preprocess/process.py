import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

games = pd.read_csv(
    "data/raw/games.csv", 
    usecols=["app_id", "title", "date_release", "rating", "positive_ratio", "user_reviews", "price_final"],
    parse_dates=["date_release"]
)

recommendations = pd.read_csv(
    "data/raw/recommendations.csv", 
    usecols=["app_id", "date", "is_recommended", "hours", "user_id", "review_id"],
    parse_dates=["date"]
)

#==============================================================================
# Lọc game
print("Lọc game")
filtered_games = games[
    (games["user_reviews"] >= 1000) &
    (games["date_release"].dt.year >= 2009) &
    (games["date_release"].dt.year <= 2023)
]

#==============================================================================
# Lọc recommendations từ pre-release
print("Lọc recommendations từ pre-release")
filtered_recs = recommendations[recommendations["app_id"].isin(filtered_games["app_id"])]
release_dates = filtered_games.set_index("app_id")["date_release"]
filtered_recs["release_date"] = filtered_recs["app_id"].map(release_dates)
filtered_recs = filtered_recs[filtered_recs["date"] >= filtered_recs["release_date"]]

#==============================================================================
# Lọc user có >= 30 feedback
print("Lọc user có >= 30 feedback")
user_counts = filtered_recs["user_id"].value_counts()
valid_users = user_counts[user_counts >= 30].index
filtered_recs = filtered_recs[filtered_recs["user_id"].isin(valid_users)]

#==============================================================================
# Transform data
filtered_recs["hours_log"] = np.log10(1 + filtered_recs["hours"])
filtered_recs["is_recommended"] = filtered_recs["is_recommended"].astype(int)

#==============================================================================
# Lọc user theo điều kiện pos/neg
print("Lọc user theo điều kiện pos/neg")
filtered_recs = filtered_recs.sort_values("date").reset_index(drop=True)

start_date = filtered_recs["date"].min()
end_date = filtered_recs["date"].max()
pivot_date = start_date + (end_date - start_date) * 0.918

print(f"Ngày bắt đầu: {start_date}")
print(f"Ngày kết thúc: {end_date}")
print(f"Mốc chia: {pivot_date}")

train_df = filtered_recs[filtered_recs["date"] < pivot_date].copy()
test_df = filtered_recs[filtered_recs["date"] >= pivot_date].copy()

valid_users = []

for user_id in filtered_recs["user_id"].unique():
    user_train = train_df[train_df["user_id"] == user_id]
    user_test = test_df[test_df["user_id"] == user_id]

    has_pos_train = (user_train["is_recommended"] == 1).any()
    has_neg_train = (user_train["is_recommended"] == 0).any()
    has_pos_test = (user_test["is_recommended"] == 1).any()

    if has_pos_train and has_neg_train and has_pos_test:
        valid_users.append(user_id)

train_df = train_df[train_df["user_id"].isin(valid_users)].reset_index(drop=True)
test_df = test_df[test_df["user_id"].isin(valid_users)].reset_index(drop=True)
filtered_recs = filtered_recs[filtered_recs["user_id"].isin(valid_users)].reset_index(drop=True)

#==============================================================================
# In số lượng còn lại
print(f"Số user: {filtered_recs['user_id'].nunique()}")
print(f"Số game: {filtered_recs['app_id'].nunique()}")
print(f"Số recommendation: {len(filtered_recs)}")
print(f"Tỉ lệ train/test: {len(train_df)}/{len(test_df)} ({len(train_df) / len(filtered_recs) * 100}%)")

#==============================================================================
# Mô tả số lượng pos, neg, tỉ lệ trong train
train_stats = train_df.groupby(["user_id", "is_recommended"]).size().unstack(fill_value=0)

train_stats.columns = ["Negative", "Positive"] if 0 in train_stats.columns else train_stats.columns
train_stats["Ratio_Positive"] = train_stats["Positive"] / train_stats["Negative"]

print("Describe số lượng Positive:")
print(train_stats["Positive"].describe())

print("\nDescribe số lượng Negative:")
print(train_stats["Negative"].describe())

print("\nDescribe tỷ lệ Positive:")
print(train_stats["Ratio_Positive"].describe())

test_df_sorted = test_df.sort_values(["user_id", "date"]).reset_index(drop=True)

#==============================================================================
# Vẽ heatmap theo thời gian trong tập test
user_id_to_index = {uid: idx for idx, uid in enumerate(test_df_sorted["user_id"].unique())}
test_df_sorted["user_index"] = test_df_sorted["user_id"].map(user_id_to_index)

# Vẽ scatter plot
plt.figure(figsize=(15, 8))
colors = test_df_sorted["is_recommended"].map({1: "lightgreen", 0: "darkgray"})

plt.scatter(
    test_df_sorted["date"],
    test_df_sorted["user_index"],
    c=colors,
    s=10,
    alpha=0.7
)

plt.xlabel("Thời gian review (test set)")
plt.ylabel("Người dùng")
plt.title("Heatmap thời gian test set (sáng: is_recommended=1, tối: is_recommended=0)")
plt.grid(True, linestyle="--", alpha=0.3)

# Format trục thời gian
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.show()

#==============================================================================
# Phân bố số recommendation theo thời gian (tính theo quý)
df = pd.concat([train_df, test_df])
df["quarter"] = df["date"].dt.to_period("Q").dt.to_timestamp()
review_counts = df.groupby("quarter").size()

plt.figure(figsize=(14, 6))
plt.plot(review_counts.index, review_counts.values, marker="o", linestyle="-", color="steelblue")
plt.tight_layout()
plt.show()

#==============================================================================
# Lưu
filtered_games = games[games["app_id"].isin(filtered_recs["app_id"].unique())]

filtered_recs.to_csv("data/processed/recommendations_processed.csv", index=False)
filtered_games.to_csv("data/processed/games_processed.csv", index=False)
print("Đã lưu thành công")

