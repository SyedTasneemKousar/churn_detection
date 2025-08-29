"""
Machine Learning Models for Customer Churn Prediction
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score
)
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

from config import *
from data_preprocessing import DataPreprocessor

class ChurnMLModels:
    """Machine Learning models for churn prediction"""
    
    def __init__(self):
        self.models = {}
        self.model_scores = {}
        self.preprocessor = DataPreprocessor()
        
    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        
        # Load preprocessor
        self.preprocessor.load_preprocessor()
        
        # Load processed data
        df = pd.read_csv(PROCESSED_DATA_PATH)
        
        # Split features and target
        X = df[self.preprocessor.feature_columns]
        y = df[self.preprocessor.target_column]
        
        # Split train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Features: {len(self.preprocessor.feature_columns)}")
        
        return X_train, X_test, y_train, y_test
    
    def handle_imbalanced_data(self, X_train, y_train):
        """Handle imbalanced dataset using SMOTE"""
        print("Applying SMOTE for handling imbalanced data...")
        
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"Original training set: {len(X_train)} samples")
        print(f"Balanced training set: {len(X_train_balanced)} samples")
        print(f"Original churn rate: {y_train.mean():.2%}")
        print(f"Balanced churn rate: {y_train_balanced.mean():.2%}")
        
        return X_train_balanced, y_train_balanced
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        print("\nTraining Logistic Regression...")
        
        model = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models['logistic_regression'] = model
        joblib.dump(model, LOGISTIC_MODEL_PATH)
        
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        print("\nTraining Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models['random_forest'] = model
        joblib.dump(model, RF_MODEL_PATH)
        
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        print("\nTraining XGBoost...")
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models['xgboost'] = model
        joblib.dump(model, XGBOOST_MODEL_PATH)
        
        return model
    
    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM model"""
        print("\nTraining LightGBM...")
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models['lightgbm'] = model
        joblib.dump(model, LIGHTGBM_MODEL_PATH)
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        print(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Store scores
        self.model_scores[model_name] = {
            'roc_auc': roc_auc,
            'f1_score': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return roc_auc, f1, y_pred, y_pred_proba
    
    def plot_model_comparison(self, X_test, y_test):
        """Plot model comparison visualizations"""
        print("\nGenerating model comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Colors for different models
        colors = ['blue', 'green', 'red', 'orange']
        
        # 1. ROC Curves
        for i, (model_name, model) in enumerate(self.models.items()):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            axes[0, 0].plot(fpr, tpr, color=colors[i], 
                           label=f'{model_name.replace("_", " ").title()} (AUC = {auc_score:.3f})')
        
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curves
        for i, (model_name, model) in enumerate(self.models.items()):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            
            axes[0, 1].plot(recall, precision, color=colors[i], 
                           label=f'{model_name.replace("_", " ").title()}')
        
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Model Scores Comparison
        model_names = [name.replace('_', ' ').title() for name in self.model_scores.keys()]
        roc_scores = [scores['roc_auc'] for scores in self.model_scores.values()]
        f1_scores = [scores['f1_score'] for scores in self.model_scores.values()]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, roc_scores, width, label='ROC-AUC', alpha=0.8)
        axes[1, 0].bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
        
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Scores')
        axes[1, 0].set_title('Model Performance Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Confusion Matrix for Best Model
        best_model_name = max(self.model_scores.keys(), key=lambda x: self.model_scores[x]['roc_auc'])
        best_y_pred = self.model_scores[best_model_name]['y_pred']
        
        cm = confusion_matrix(y_test, best_y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title(f'Confusion Matrix - {best_model_name.replace("_", " ").title()}')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Model comparison plot saved to {PLOTS_DIR / 'model_comparison.png'}")
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        print("\nGenerating feature importance plots...")
        
        # Get tree-based models
        tree_models = {k: v for k, v in self.models.items() if k in ['random_forest', 'xgboost', 'lightgbm']}
        
        if not tree_models:
            print("No tree-based models found for feature importance analysis.")
            return
        
        fig, axes = plt.subplots(1, len(tree_models), figsize=(20, 8))
        if len(tree_models) == 1:
            axes = [axes]
        
        for i, (model_name, model) in enumerate(tree_models.items()):
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                continue
            
            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'feature': self.preprocessor.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=True)
            
            # Plot top 15 features
            top_features = feature_importance_df.tail(15)
            
            axes[i].barh(top_features['feature'], top_features['importance'])
            axes[i].set_title(f'Feature Importance - {model_name.replace("_", " ").title()}')
            axes[i].set_xlabel('Importance')
            
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Feature importance plot saved to {PLOTS_DIR / 'feature_importance.png'}")
    
    def train_all_models(self, use_smote=True):
        """Train all ML models"""
        print("=" * 60)
        print("TRAINING MACHINE LEARNING MODELS")
        print("=" * 60)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Handle imbalanced data if requested
        if use_smote:
            X_train, y_train = self.handle_imbalanced_data(X_train, y_train)
        
        # Train models
        print("\nTraining models...")
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_lightgbm(X_train, y_train)
        
        # Evaluate models
        print("\n" + "=" * 60)
        print("EVALUATING MODELS")
        print("=" * 60)
        
        for model_name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, model_name)
        
        # Generate visualizations
        self.plot_model_comparison(X_test, y_test)
        self.plot_feature_importance()
        
        # Print summary
        self.print_model_summary()
        
        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETED")
        print("=" * 60)
        
        return X_train, X_test, y_train, y_test
    
    def print_model_summary(self):
        """Print summary of all models"""
        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 60)
        
        # Create summary dataframe
        summary_data = []
        for model_name, scores in self.model_scores.items():
            summary_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'ROC-AUC': f"{scores['roc_auc']:.4f}",
                'F1-Score': f"{scores['f1_score']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Find best model
        best_model = max(self.model_scores.keys(), key=lambda x: self.model_scores[x]['roc_auc'])
        best_auc = self.model_scores[best_model]['roc_auc']
        
        print(f"\nBest Model: {best_model.replace('_', ' ').title()} (ROC-AUC: {best_auc:.4f})")
        
        # Save summary
        summary_df.to_csv(MODELS_DIR / 'model_summary.csv', index=False)
        print(f"Model summary saved to {MODELS_DIR / 'model_summary.csv'}")
    
    def load_trained_models(self):
        """Load pre-trained models"""
        print("Loading trained models...")
        
        model_paths = {
            'logistic_regression': LOGISTIC_MODEL_PATH,
            'random_forest': RF_MODEL_PATH,
            'xgboost': XGBOOST_MODEL_PATH,
            'lightgbm': LIGHTGBM_MODEL_PATH
        }
        
        for model_name, model_path in model_paths.items():
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name}")
            else:
                print(f"Model file not found: {model_path}")
        
        print(f"Loaded {len(self.models)} models")
    
    def predict_churn(self, customer_data):
        """Predict churn for new customer data"""
        if not self.models:
            self.load_trained_models()
        
        # Preprocess the data
        processed_data = self.preprocessor.transform_new_data(customer_data)
        
        # Get predictions from all models
        predictions = {}
        for model_name, model in self.models.items():
            pred_proba = model.predict_proba(processed_data)[:, 1]
            predictions[model_name] = pred_proba
        
        return predictions

def main():
    """Main function to train all ML models"""
    ml_models = ChurnMLModels()
    X_train, X_test, y_train, y_test = ml_models.train_all_models(use_smote=True)
    
    print("\nMachine Learning pipeline completed successfully!")
    print(f"Models saved in: {MODELS_DIR}")
    print(f"Plots saved in: {PLOTS_DIR}")

if __name__ == "__main__":
    main()




