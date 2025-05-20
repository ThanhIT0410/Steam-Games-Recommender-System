import numpy as np
import random
import yaml
from data_utils import load_data, split_data_full_corpus, split_data_leave_one_last
import recommender

def load_config(path="config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)
    
model_map = {
    "ItemKNN": recommender.ItemKNN,
    "MF_logistic": recommender.MF_logistic,
    "MF_bpr": recommender.MF_bpr,
    "MF_hinge_pointwise": recommender.MF_hinge_pointwise,
    "MF_hinge_pairwise": recommender.MF_hinge_pairwise,
    "MLP": recommender.MLP,
    "ConvNCF": recommender.ConvNCF,
}

random.seed(42)
np.random.seed(42)

config = load_config()
model_name = config["recommender"]
model_config = config[model_name]
evaluation_type = model_config["evaluation_type"]
model = model_map[model_name](**model_config)

if evaluation_type == "Full-corpus":
    train_ratio = 0.8
    recommendations = load_data()
    train_set, test_set = split_data_full_corpus(recommendations, train_ratio)

    model.fit(train_set)

    model.evaluation_type = "Full-corpus"
    path = f"saved_models/{model_name}_fullcorpus.pth"
    model.save(path)

    print(f"Full-corpus training completed. Model saved to {path}")
else:
    recommendations = load_data()
    train_set, test_set = split_data_leave_one_last(recommendations)

    model.fit(train_set)

    model.evaluation_type = "Leave-one-last"
    path = f"saved_models/{model_name}_leaveonelast.pth"
    model.save(path)

    print(f"Leave-one-last training completed. Model saved to {path}")
