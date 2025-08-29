# 🎯 Customer Churn Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive end-to-end machine learning system that predicts customer churn using traditional ML, deep learning, and NLP techniques. This project combines structured customer data with sentiment analysis from customer feedback to provide accurate churn predictions and actionable business insights.

## 🚀 Live Demo

- **FastAPI API**: [http://localhost:8000](http://localhost:8000)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit Dashboard**: [http://localhost:8501](http://localhost:8501)

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repository to GitHub (public or private) and grant Streamlit access.
2. Go to Streamlit Community Cloud → New app.
3. Select your repo and branch.
4. Set App file to `streamlit_dashboard.py`.
5. Deploy. The build uses `requirements.txt` and `runtime.txt` (Python 3.11). Optional settings live in `.streamlit/config.toml`.

Notes:
- Large models/data should be downloaded at runtime from a remote URL or stored with Git LFS.
- Use `st.secrets` for private keys/URLs (Settings → Secrets in Streamlit Cloud).

## 🌟 Features

### 🤖 Machine Learning Models
- **Traditional ML**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **Deep Learning**: Neural Networks, LSTM for sequential data
- **Hybrid Model**: Combines structured data with NLP sentiment features
- **Model Comparison**: Comprehensive performance analysis and visualization

### 📊 Data Analysis & Visualization
- **Exploratory Data Analysis (EDA)**: Interactive visualizations and insights
- **Customer Segmentation**: K-Means and DBSCAN clustering
- **Cohort Analysis**: Customer retention trends over time
- **Feature Engineering**: RFM scores, customer lifetime value, engagement metrics

### 💭 Natural Language Processing
- **Sentiment Analysis**: Customer feedback sentiment using BERT/RoBERTa
- **Topic Modeling**: Extract key themes from customer complaints
- **Text Analytics**: Word clouds, sentiment trends, and emotion analysis

### 🌐 Web Applications
- **Streamlit Dashboard**: Interactive web interface for data exploration and predictions
- **FastAPI REST API**: Production-ready API for real-time churn predictions
- **Model Serving**: Scalable prediction endpoints with automatic model loading

### 🎯 Business Intelligence
- **Churn Risk Scoring**: Customer risk levels with actionable recommendations
- **Retention Strategies**: Personalized recommendations based on customer segments
- **Revenue Impact**: Calculate potential revenue loss from churned customers


## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/churn_detection.git
cd churn_detection

# Run the automated deployment script
python deploy_to_github.py

# The script will:
# 1. Check if the application is running
# 2. Start it if needed
# 3. Help you set up GitHub repository
# 4. Deploy to GitHub
```



### Option 2: Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data
python data_generator.py

# Train models
python data_preprocessing.py
python ml_models.py
python nlp_sentiment.py
python hybrid_model.py

# Start applications
# Dashboard
streamlit run streamlit_dashboard.py

## 📁 Project Structure

```
churn_prediction/
├── 📊 Data & Models
│   ├── data/                    # Dataset files
│   ├── models/                  # Trained model files
│   ├── plots/                   # Generated visualizations
│   └── logs/                    # Application logs
│
├── 🔧 Core Components
│   ├── config.py               # Configuration settings
│   ├── data_generator.py       # Sample data generation
│   ├── data_preprocessing.py   # Data cleaning & feature engineering
│   └── eda_analysis.py         # Exploratory data analysis
│
├── 🤖 Machine Learning
│   ├── ml_models.py            # Traditional ML models
│   ├── deep_learning_models.py # Neural networks & LSTM
│   ├── hybrid_model.py         # Combined structured + NLP model
│   └── model_comparison.py     # Model performance analysis
│
├── 💭 NLP & Analytics
│   ├── nlp_sentiment.py        # Sentiment analysis pipeline
│   └── customer_clustering.py  # Customer segmentation
│
├── 🌐 Applications
   ├── streamlit_dashboard.py  # Interactive web dashboard
  └── fastapi_app.py         # REST API server



## 📈 Dataset

The system uses a comprehensive telecom churn dataset with:

- **10,000 customers** with 25+ features
- **Demographics**: Age, gender, family status
- **Account Info**: Tenure, contract type, payment method
- **Services**: Phone, internet, streaming, security features
- **Financial**: Monthly charges, total charges, usage patterns
- **Engagement**: Support calls, satisfaction scores
- **Feedback**: 5,000 customer support tickets with sentiment labels

### Sample Data Structure
```python
{
    'customerID': 'CUST_000001',
    'tenure': 24,
    'MonthlyCharges': 75.50,
    'Contract': 'Month-to-month',
    'SatisfactionScore': 7.2,
    'Churn': 'No'
}
```

## 🎯 Model Performance

| Model | ROC-AUC | F1-Score | Precision | Recall |
|-------|---------|----------|-----------|---------|
| **Hybrid XGBoost** | **0.851** | **0.723** | 0.698 | 0.749 |
| XGBoost | 0.843 | 0.715 | 0.692 | 0.739 |
| Random Forest | 0.831 | 0.702 | 0.678 | 0.728 |
| Neural Network | 0.825 | 0.695 | 0.671 | 0.721 |
| Logistic Regression | 0.798 | 0.668 | 0.645 | 0.693 |

*The hybrid model achieves the best performance by combining structured customer data with sentiment analysis from feedback text.*

## 🔗 API Endpoints

### Core Prediction Endpoints
```http
POST /predict                    # Single customer prediction
POST /predict/batch             # Batch predictions
GET  /models/compare            # Compare model predictions
```

### Analysis Endpoints
```http
POST /analyze/risk              # Customer risk analysis
GET  /models/performance        # Model performance metrics
GET  /models/feature-importance # Feature importance analysis
```

### Health & Info
```http
GET  /                          # Health check
GET  /health                    # Detailed system status
GET  /docs                      # Interactive API documentation
```

### Example API Usage

```python
import requests

# Single prediction
response = requests.post('http://localhost:8000/predict', json={
    'customerID': 'CUST_001',
    'tenure': 12,
    'MonthlyCharges': 85.0,
    'Contract': 'Month-to-month',
    'SatisfactionScore': 5.5,
    # ... other features
})

result = response.json()
print(f"Churn Risk: {result['risk_level']}")
print(f"Probability: {result['churn_probability']:.2%}")
```

## 📊 Dashboard Features

### 🏠 Overview Page
- Key business metrics and KPIs
- Churn rate trends and patterns
- Revenue at risk analysis
- Customer distribution insights

### 🔍 Data Exploration
- Interactive feature analysis
- Correlation heatmaps
- Distribution plots
- Customer segmentation views

### 🤖 Model Performance
- Model comparison charts
- ROC curves and precision-recall analysis
- Feature importance rankings
- Confusion matrices

### 🎯 Churn Prediction
- Individual customer prediction interface
- Risk factor identification
- Personalized retention recommendations
- Batch prediction upload

### 💭 Sentiment Analysis
- Customer feedback sentiment trends
- Word clouds and topic analysis
- Sentiment impact on churn
- Feedback volume analysis

### 👥 Customer Segmentation
- Cluster analysis and profiling
- Segment-specific churn rates
- Targeted marketing recommendations
- Customer journey insights

## 🎨 Visualizations

The system generates comprehensive visualizations:

- **Model Comparison**: ROC curves, precision-recall, performance metrics
- **Feature Analysis**: Importance plots, correlation heatmaps, distribution charts
- **Customer Insights**: Segmentation plots, cohort analysis, retention trends
- **Sentiment Analytics**: Word clouds, sentiment trends, topic modeling
- **Business Metrics**: Revenue impact, churn forecasting, KPI dashboards

## 🔧 Configuration

Key configuration options in `config.py`:

```python
# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
NEURAL_NET_CONFIG = {
    "hidden_layers": [128, 64, 32],
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "epochs": 100
}

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000
DASHBOARD_PORT = 8501

# NLP Configuration
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_TEXT_LENGTH = 512
```

## 📊 Monitoring & Logging

- **Application Logs**: Structured logging with timestamps and levels
- **Model Performance**: Automated model drift detection
- **API Metrics**: Request/response times, error rates
- **Business KPIs**: Churn rate trends, prediction accuracy

## 🔒 Security Features

- **Input Validation**: Pydantic models for API request validation
- **Error Handling**: Graceful error responses without exposing internals
- **Rate Limiting**: Built-in FastAPI rate limiting capabilities
- **Data Privacy**: No sensitive data logging or storage



## 🙏 Acknowledgments

- **Scikit-learn** for machine learning algorithms
- **XGBoost** and **LightGBM** for gradient boosting
- **PyTorch** for deep learning capabilities
- **Hugging Face** for NLP models and transformers
- **Streamlit** for the interactive dashboard
- **FastAPI** for the high-performance API framework
- **Plotly** for interactive visualizations

## 📞 Support

For questions, issues, or contributions, please:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed description
3. Contact the maintainers

---

**Built with ❤️ for better customer retention and business growth**



