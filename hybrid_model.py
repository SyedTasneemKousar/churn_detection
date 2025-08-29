"""
Hybrid Model combining structured features with NLP sentiment analysis
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from config import *
from data_preprocessing import DataPreprocessor
from nlp_sentiment import SentimentAnalyzer

class HybridChurnModel:
    """Hybrid model combining structured features with sentiment analysis"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.hybrid_model = None
        self.feature_columns = []
        
    def load_and_prepare_data(self):
        """Load and prepare data with sentiment features"""
        print("Loading and preparing hybrid data...")
        
        # Load structured data
        df_structured = pd.read_csv(PROCESSED_DATA_PATH)
        print(f"Loaded structured data: {df_structured.shape}")
        
        # Load sentiment data
        try:
            df_sentiment = pd.read_csv(DATA_DIR / 'customer_sentiment_scores.csv')
            print(f"Loaded sentiment data: {df_sentiment.shape}")
        except FileNotFoundError:
            print("Sentiment data not found. Running sentiment analysis...")
            self.sentiment_analyzer.run_complete_sentiment_analysis()
            df_sentiment = pd.read_csv(DATA_DIR / 'customer_sentiment_scores.csv')
        
        # Merge structured and sentiment data
        df_hybrid = df_structured.merge(
            df_sentiment[['customerID', 'avg_sentiment', 'sentiment_std', 'feedback_count', 'avg_confidence']], 
            on='customerID', 
            how='left'
        )
        
        # Fill missing sentiment values for customers without feedback
        sentiment_columns = ['avg_sentiment', 'sentiment_std', 'feedback_count', 'avg_confidence']
        for col in sentiment_columns:
            if col == 'feedback_count':
                df_hybrid[col] = df_hybrid[col].fillna(0)
            elif col == 'sentiment_std':
                df_hybrid[col] = df_hybrid[col].fillna(0)
            else:  # avg_sentiment, avg_confidence
                df_hybrid[col] = df_hybrid[col].fillna(df_hybrid[col].median())
        
        print(f"Hybrid dataset shape: {df_hybrid.shape}")
        print(f"Customers with sentiment data: {(df_hybrid['feedback_count'] > 0).sum()}")
        
        return df_hybrid
    
    def create_sentiment_features(self, df):
        """Create additional features from sentiment data"""
        print("Creating sentiment-based features...")
        
        # Sentiment engagement features
        df['has_feedback'] = (df['feedback_count'] > 0).astype(int)
        df['high_feedback_volume'] = (df['feedback_count'] > df['feedback_count'].quantile(0.75)).astype(int)
        df['negative_sentiment'] = (df['avg_sentiment'] < -0.1).astype(int)
        df['positive_sentiment'] = (df['avg_sentiment'] > 0.1).astype(int)
        df['mixed_sentiment'] = (df['sentiment_std'] > df['sentiment_std'].quantile(0.75)).astype(int)
        
        # Interaction features
        df['sentiment_feedback_interaction'] = df['avg_sentiment'] * df['feedback_count']
        df['confidence_weighted_sentiment'] = df['avg_sentiment'] * df['avg_confidence']
        
        print("Created sentiment-based features")
        return df
    
    def prepare_hybrid_features(self, df):
        """Prepare final feature set for hybrid model"""
        print("Preparing hybrid feature set...")
        
        # Load original preprocessor to get feature columns
        self.preprocessor.load_preprocessor()
        
        # Original features
        original_features = self.preprocessor.feature_columns
        
        # New sentiment features
        sentiment_features = [
            'avg_sentiment', 'sentiment_std', 'feedback_count', 'avg_confidence',
            'has_feedback', 'high_feedback_volume', 'negative_sentiment', 
            'positive_sentiment', 'mixed_sentiment', 'sentiment_feedback_interaction',
            'confidence_weighted_sentiment'
        ]
        
        # Combine all features
        self.feature_columns = original_features + sentiment_features
        
        # Remove customerID and target from features
        exclude_columns = ['customerID', 'Churn']
        self.feature_columns = [col for col in self.feature_columns if col in df.columns and col not in exclude_columns]
        
        print(f"Total hybrid features: {len(self.feature_columns)}")
        print(f"Original features: {len(original_features)}")
        print(f"Sentiment features: {len(sentiment_features)}")
        
        return df[self.feature_columns + ['Churn']]
    
    def train_hybrid_model(self, X_train, y_train, model_type='xgboost'):
        """Train hybrid model"""
        print(f"Training hybrid {model_type} model...")
        
        if model_type == 'xgboost':
            # Calculate scale_pos_weight for imbalanced data
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            
            self.hybrid_model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                eval_metric='logloss'
            )
        
        elif model_type == 'random_forest':
            self.hybrid_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                n_jobs=-1
            )
        
        elif model_type == 'logistic':
            self.hybrid_model = LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                class_weight='balanced'
            )
        
        # Train the model
        self.hybrid_model.fit(X_train, y_train)
        
        # Cross-validation
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(self.hybrid_model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.hybrid_model
    
    def evaluate_hybrid_model(self, X_test, y_test):
        """Evaluate hybrid model performance"""
        print("Evaluating hybrid model...")
        
        # Predictions
        y_pred = self.hybrid_model.predict(X_test)
        y_pred_proba = self.hybrid_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Hybrid Model Performance:")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return roc_auc, f1, y_pred, y_pred_proba
    
    def save_hybrid_model(self):
        """Save the trained hybrid model"""
        model_path = MODELS_DIR / 'hybrid_churn_model.joblib'
        joblib.dump(self.hybrid_model, model_path)
        
        # Save feature columns
        feature_info = {
            'feature_columns': self.feature_columns,
            'model_type': type(self.hybrid_model).__name__
        }
        joblib.dump(feature_info, MODELS_DIR / 'hybrid_model_info.joblib')
        
        print(f"Hybrid model saved to {model_path}")
    
    def load_hybrid_model(self):
        """Load pre-trained hybrid model"""
        model_path = MODELS_DIR / 'hybrid_churn_model.joblib'
        info_path = MODELS_DIR / 'hybrid_model_info.joblib'
        
        if model_path.exists() and info_path.exists():
            self.hybrid_model = joblib.load(model_path)
            feature_info = joblib.load(info_path)
            self.feature_columns = feature_info['feature_columns']
            print(f"Loaded hybrid model: {feature_info['model_type']}")
            return True
        else:
            print("Hybrid model files not found")
            return False

def main():
    """Main function to run hybrid model training"""
    hybrid_model = HybridChurnModel()
    
    print("=" * 60)
    print("TRAINING HYBRID CHURN PREDICTION MODEL")
    print("=" * 60)
    
    # Load and prepare data
    df_hybrid = hybrid_model.load_and_prepare_data()
    df_hybrid = hybrid_model.create_sentiment_features(df_hybrid)
    df_features = hybrid_model.prepare_hybrid_features(df_hybrid)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X = df_features.drop('Churn', axis=1)
    y = df_features['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train model
    hybrid_model.train_hybrid_model(X_train, y_train, model_type='xgboost')
    
    # Evaluate model
    roc_auc, f1, y_pred, y_pred_proba = hybrid_model.evaluate_hybrid_model(X_test, y_test)
    
    # Save model
    hybrid_model.save_hybrid_model()
    
    print("=" * 60)
    print("HYBRID MODEL TRAINING COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()




