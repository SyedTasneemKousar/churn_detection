# Customer Churn Prediction System - Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models logs plots

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 8501

# Create entrypoint script
RUN echo '#!/bin/bash\n\
\n\
echo "🚀 Starting Customer Churn Prediction System"\n\
echo "============================================"\n\
\n\
# Check if models exist, if not run training\n\
if [ ! -f "models/xgboost.joblib" ]; then\n\
    echo "📊 Training models (this may take a few minutes)..."\n\
    python data_preprocessing.py\n\
    python ml_models.py\n\
    echo "✅ Model training completed"\n\
fi\n\
\n\
# Start the application based on command\n\
case "$1" in\n\
    "api")\n\
        echo "🔗 Starting FastAPI server on port 8000..."\n\
        uvicorn fastapi_app:app --host 0.0.0.0 --port 8000\n\
        ;;\n\
    "dashboard")\n\
        echo "📊 Starting Streamlit dashboard on port 8501..."\n\
        streamlit run streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0\n\
        ;;\n\
    "train")\n\
        echo "🤖 Training all models..."\n\
        python data_preprocessing.py\n\
        python ml_models.py\n\
        python nlp_sentiment.py\n\
        python hybrid_model.py\n\
        python model_comparison.py\n\
        echo "✅ Training completed"\n\
        ;;\n\
    "analyze")\n\
        echo "📈 Running complete analysis..."\n\
        python eda_analysis.py\n\
        python customer_clustering.py\n\
        echo "✅ Analysis completed"\n\
        ;;\n\
    *)\n\
        echo "Usage: docker run <image> [api|dashboard|train|analyze]"\n\
        echo "  api       - Start FastAPI server"\n\
        echo "  dashboard - Start Streamlit dashboard"\n\
        echo "  train     - Train all models"\n\
        echo "  analyze   - Run EDA and clustering"\n\
        exit 1\n\
        ;;\n\
esac\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
