"""
Configuration settings for the taxi trip duration prediction project
"""

import os

# ========== Data Paths ==========
DATA_PATH = "data/green_tripdata.csv"
TARGET_COLUMN = "trip_duration"

# ========== Model & Artifact Paths ==========
MODEL_SAVE_PATH = "models/"
HYPEROPT_ROOT_DIR = "hyperopt_results/"

# Create directories if they don't already exist (optional utility)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(HYPEROPT_ROOT_DIR, exist_ok=True)

# ========== MLflow ==========
TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "taxi_trip_duration_prediction"

# ========== Default Model Hyperparameters ==========
DEFAULT_MODEL_PARAMS = {
    "lr": {
        # LinearRegression default has no hyperparams to tune for basic usage
    },
    "rf": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 3,
        "random_state": 42,
        "n_jobs": -1  # Use all available cores
    },
    "xgb": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1  # Use all available cores
    },
    "lgbm": {
        "n_estimators": 100,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1  # Use all available cores
    }
}

# ========== Hyperparameter Optimization Defaults ==========
HYPEROPT_DEFAULT_EVALS = 50         # Default number of hyperopt iterations
HYPEROPT_EARLY_STOPPING = 20        # Early stopping rounds (for xgb/lgbm, can be None if not used)
HYPEROPT_RANDOM_SEED = 42           # Seed for hyperopt/randomness

# ========== Financial Columns To Flag (for data cleaning/feature engineering) ==========
FINANCIAL_COLUMNS = [
    'fare_amount', 
    'extra', 
    'mta_tax', 
    'tip_amount', 
    'tolls_amount', 
    'improvement_surcharge', 
    'total_amount',
    'congestion_surcharge'
]

# ========== Feature Engineering Settings ==========
TRIP_DURATION_MIN_SECONDS = 60        # 1 minute minimum
TRIP_DURATION_MAX_SECONDS = 10800     # 3 hours maximum
MAX_SPEED_MPH = 80                    # Maximum realistic speed in MPH

# ========== Miscellaneous ==========
RANDOM_STATE = 42                     # Default split/seed
