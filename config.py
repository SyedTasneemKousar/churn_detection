"""
Configuration file for the Customer Churn Prediction System
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR]:
    directory.mkdir(exist_ok=True)

# Data file paths
RAW_DATA_PATH = DATA_DIR / "telecom_churn.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_churn.csv"
CUSTOMER_FEEDBACK_PATH = DATA_DIR / "customer_feedback.csv"

# Model configurations
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Model file paths
LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
RF_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost.joblib"
LIGHTGBM_MODEL_PATH = MODELS_DIR / "lightgbm.joblib"
NEURAL_NET_PATH = MODELS_DIR / "neural_network.pth"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.pth"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
LABEL_ENCODERS_PATH = MODELS_DIR / "label_encoders.joblib"

# NLP model configurations
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_TEXT_LENGTH = 512

# API configurations
API_HOST = "0.0.0.0"
API_PORT = 8000

# Streamlit configurations
DASHBOARD_HOST = "localhost"
DASHBOARD_PORT = 8501

# Feature engineering parameters
RFM_QUANTILES = 5
COHORT_PERIODS = 12

# Model hyperparameters
NEURAL_NET_CONFIG = {
    "hidden_layers": [128, 64, 32],
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32
}

LSTM_CONFIG = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 32,
    "sequence_length": 12
}

# Clustering parameters
N_CLUSTERS = 5
CLUSTERING_FEATURES = [
    'tenure', 'monthly_charges', 'total_charges',
    'avg_monthly_usage', 'support_calls', 'sentiment_score'
]

