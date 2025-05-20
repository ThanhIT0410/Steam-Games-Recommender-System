# Steam-Games-Recommender-System

A recommendation system project that suggests games to users using multiple models

## 📁 Project Structure
- `train.py`: Train and save models using config
- `evaluate.py`: Evaluate models on classification and ranking metrics
- `main.py`: Load a trained model and recommend top 10 games for a user
- `data_utils.py`: Data loading and preprocessing
- `recommender/`: Implementation of all models
- `config.yaml`: Configuration for model hyperparameters

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model
python train.py

# Recommend top 10 games for a user
python main.py
```

## ⚙️ Configuration

All hyperparameters are defined in config.yaml. You can switch between models easily by changing model_name.

📦 Dependencies
- Python 3.8+
- PyTorch
- pandas, numpy, yaml


## 📥 Download Dataset
This repository does not contain raw data due to file size limitations on GitHub.

To run the project, please follow the steps below:

Download the dataset from the following link:
👉 [Download Dataset from Google Drive](https://drive.google.com/drive/folders/1MezT2dg6sDR6HgV542VppE5jA1Yvp12_?usp=drive_link)

Unzip the downloaded file.

Place the extracted csv files into data/processed


