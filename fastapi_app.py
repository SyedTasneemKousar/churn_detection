"""
FastAPI REST API for Customer Churn Prediction
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from config import *
from data_preprocessing import DataPreprocessor
from ml_models import ChurnMLModels
from hybrid_model import HybridChurnModel

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="REST API for predicting customer churn using various ML models",
    version="1.0.0"
)

# Global variables for models
preprocessor = None
ml_models = None
hybrid_model = None

# Pydantic models for request/response
class CustomerData(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    AvgMonthlyUsage: float
    SupportCalls: int
    SatisfactionScore: float
    NumServices: int

class PredictionResponse(BaseModel):
    customerID: str
    churn_prediction: str
    churn_probability: float
    risk_level: str
    model_used: str
    confidence_score: float

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: Dict[str, int]

class ModelPerformance(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float

# Startup event to load models
@app.on_event("startup")
async def load_models():
    """Load all models on startup"""
    global preprocessor, ml_models, hybrid_model
    
    try:
        # Load preprocessor
        preprocessor = DataPreprocessor()
        preprocessor.load_preprocessor()
        print("✅ Preprocessor loaded")
        
        # Load ML models
        ml_models = ChurnMLModels()
        ml_models.load_trained_models()
        print(f"✅ Loaded {len(ml_models.models)} ML models")
        
        # Load hybrid model
        hybrid_model = HybridChurnModel()
        hybrid_model.load_hybrid_model()
        print("✅ Hybrid model loaded")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")

# Health check endpoint
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": {
            "preprocessor": preprocessor is not None,
            "ml_models": ml_models is not None and len(ml_models.models) > 0,
            "hybrid_model": hybrid_model is not None and hybrid_model.hybrid_model is not None
        },
        "available_models": list(ml_models.models.keys()) if ml_models else []
    }

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """Predict churn for a single customer"""
    try:
        # Convert to DataFrame
        customer_df = pd.DataFrame([customer.dict()])
        
        # Use the best available model (try hybrid first, then XGBoost)
        model_used = "unknown"
        churn_probability = 0.0
        
        if hybrid_model and hybrid_model.hybrid_model is not None:
            try:
                # Try hybrid model first
                prediction, probability = hybrid_model.predict_churn_hybrid(customer_df)
                churn_probability = probability[0]
                model_used = "hybrid_xgboost"
            except Exception as e:
                print(f"Hybrid model error: {e}")
                # Fallback to traditional ML
                if ml_models and 'xgboost' in ml_models.models:
                    predictions = ml_models.predict_churn(customer_df)
                    churn_probability = predictions['xgboost'][0]
                    model_used = "xgboost"
        elif ml_models and 'xgboost' in ml_models.models:
            predictions = ml_models.predict_churn(customer_df)
            churn_probability = predictions['xgboost'][0]
            model_used = "xgboost"
        else:
            raise HTTPException(status_code=500, detail="No models available")
        
        # Determine risk level
        if churn_probability > 0.7:
            risk_level = "HIGH"
        elif churn_probability > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Determine prediction
        churn_prediction = "Yes" if churn_probability > 0.5 else "No"
        
        # Calculate confidence score
        confidence_score = abs(churn_probability - 0.5) * 2
        
        return PredictionResponse(
            customerID=customer.customerID,
            churn_prediction=churn_prediction,
            churn_probability=churn_probability,
            risk_level=risk_level,
            model_used=model_used,
            confidence_score=confidence_score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_churn_batch(customers: List[CustomerData]):
    """Predict churn for multiple customers"""
    try:
        predictions = []
        risk_summary = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        
        for customer in customers:
            # Get individual prediction
            prediction = await predict_churn(customer)
            predictions.append(prediction)
            risk_summary[prediction.risk_level] += 1
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=risk_summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Model comparison endpoint
@app.get("/models/compare")
async def compare_models(customer_id: str = "SAMPLE_001"):
    """Compare predictions across all available models"""
    try:
        # Create sample customer data
        sample_customer = CustomerData(
            customerID=customer_id,
            gender="Male",
            SeniorCitizen=0,
            Partner="Yes",
            Dependents="No",
            tenure=24,
            PhoneService="Yes",
            MultipleLines="No",
            InternetService="Fiber optic",
            OnlineSecurity="No",
            OnlineBackup="Yes",
            DeviceProtection="No",
            TechSupport="No",
            StreamingTV="Yes",
            StreamingMovies="No",
            Contract="Month-to-month",
            PaperlessBilling="Yes",
            PaymentMethod="Electronic check",
            MonthlyCharges=75.0,
            TotalCharges=1800.0,
            AvgMonthlyUsage=45.0,
            SupportCalls=2,
            SatisfactionScore=6.5,
            NumServices=4
        )
        
        customer_df = pd.DataFrame([sample_customer.dict()])
        
        results = {}
        
        # Traditional ML models
        if ml_models:
            ml_predictions = ml_models.predict_churn(customer_df)
            for model_name, prob in ml_predictions.items():
                results[model_name] = {
                    "probability": float(prob[0]) if isinstance(prob, np.ndarray) else float(prob),
                    "prediction": "Yes" if prob[0] > 0.5 else "No",
                    "model_type": "Traditional ML"
                }
        
        # Hybrid model
        if hybrid_model and hybrid_model.hybrid_model is not None:
            try:
                prediction, probability = hybrid_model.predict_churn_hybrid(customer_df)
                results["hybrid_model"] = {
                    "probability": float(probability[0]),
                    "prediction": "Yes" if probability[0] > 0.5 else "No",
                    "model_type": "Hybrid ML+NLP"
                }
            except Exception as e:
                print(f"Hybrid model comparison error: {e}")
        
        return {
            "customer_id": customer_id,
            "model_predictions": results,
            "best_model": max(results.keys(), key=lambda x: results[x]["probability"]) if results else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model comparison error: {str(e)}")

# Model performance endpoint
@app.get("/models/performance", response_model=List[ModelPerformance])
async def get_model_performance():
    """Get performance metrics for all models"""
    try:
        # Load model summary if available
        try:
            summary_df = pd.read_csv(MODELS_DIR / 'comprehensive_model_summary.csv')
            
            performance_list = []
            for _, row in summary_df.iterrows():
                performance_list.append(ModelPerformance(
                    model_name=row['Model'],
                    accuracy=float(row['Accuracy']),
                    precision=float(row['Precision']),
                    recall=float(row['Recall']),
                    f1_score=float(row['F1_Score']),
                    roc_auc=float(row['ROC_AUC'])
                ))
            
            return performance_list
            
        except FileNotFoundError:
            # Return basic info if summary not available
            return [
                ModelPerformance(
                    model_name="Models not evaluated",
                    accuracy=0.0,
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    roc_auc=0.0
                )
            ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance retrieval error: {str(e)}")

# Customer risk analysis endpoint
@app.post("/analyze/risk")
async def analyze_customer_risk(customer: CustomerData):
    """Analyze risk factors for a customer"""
    try:
        risk_factors = []
        recommendations = []
        
        # Analyze risk factors
        if customer.MonthlyCharges > 80:
            risk_factors.append("High monthly charges")
            recommendations.append("Consider offering discount or loyalty program")
        
        if customer.tenure < 12:
            risk_factors.append("Short tenure (new customer)")
            recommendations.append("Implement new customer onboarding program")
        
        if customer.Contract == "Month-to-month":
            risk_factors.append("Month-to-month contract")
            recommendations.append("Incentivize longer-term contract commitment")
        
        if customer.SupportCalls > 3:
            risk_factors.append("High number of support calls")
            recommendations.append("Proactive customer service outreach")
        
        if customer.SatisfactionScore < 6:
            risk_factors.append("Low satisfaction score")
            recommendations.append("Conduct satisfaction survey and address concerns")
        
        if customer.PaymentMethod == "Electronic check":
            risk_factors.append("Electronic check payment method")
            recommendations.append("Encourage automatic payment methods")
        
        # Get churn prediction
        prediction_response = await predict_churn(customer)
        
        return {
            "customer_id": customer.customerID,
            "churn_probability": prediction_response.churn_probability,
            "risk_level": prediction_response.risk_level,
            "risk_factors": risk_factors,
            "recommendations": recommendations,
            "risk_score": len(risk_factors) / 6.0  # Normalized risk score
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk analysis error: {str(e)}")

# Feature importance endpoint
@app.get("/models/feature-importance")
async def get_feature_importance():
    """Get feature importance from the best model"""
    try:
        if ml_models and 'xgboost' in ml_models.models:
            model = ml_models.models['xgboost']
            
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_names = preprocessor.feature_columns
                
                feature_importance = [
                    {"feature": name, "importance": float(imp)}
                    for name, imp in zip(feature_names, importance)
                ]
                
                # Sort by importance
                feature_importance.sort(key=lambda x: x["importance"], reverse=True)
                
                return {
                    "model": "xgboost",
                    "feature_importance": feature_importance[:20]  # Top 20 features
                }
        
        return {"message": "Feature importance not available"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature importance error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
