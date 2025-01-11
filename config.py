# config.py

# Paths
DATA_PATH = './creditcard_2023.csv'
MODEL_PATH = 'random_forest_model.joblib'

# Model parameters
RANDOM_FOREST_PARAMS = {
    'random_state': 42,
    # Add other parameters as needed, e.g., 'max_depth': 10
}
XGBOOST_PARAMS = {
    'eval_metric': 'logloss',
    'random_state': 42,
}
