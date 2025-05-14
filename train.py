import pandas as pd
import numpy as np
import random
import yaml
from data_utils import load_data, split_data
import recommender

def load_config(path="config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)
    
model_map = {
    "ItemKNN": recommender.ItemKNN,
    "MF_logistic": recommender.MF_logistic,
    "MF_bpr": recommender.MF_bpr,
    "MF_svm": recommender.MF_svm,
    "MLP": recommender.MLP,
    "ConvNCF": recommender.ConvNCF
}

random.seed(42)
np.random.seed(42)

config = load_config()
model_name = config["model_name"]
model_config = config[model_name]
model = model_map[model_name](**model_config)

train_ratio = 0.8
recommendations = load_data()
train_set, test_set, game_list = split_data(recommendations, train_ratio)

model.fit(train_set)

path = f"saved_models/{model_name}.pth"
model.save(path)

print(f"Model saved to {path}")
