"""
Data preprocessing and feature engineering for churn prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import *

class DataPreprocessor:
    """Handle data preprocessing and feature engineering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'Churn'
        
    def load_data(self, data_path=RAW_DATA_PATH):
        """Load raw data"""
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def basic_preprocessing(self, df):
        """Basic data cleaning and preprocessing"""
        print("Performing basic preprocessing...")
        
        # Create a copy
        df = df.copy()
        
        # Handle missing values
        print(f"Missing values before cleaning:\n{df.isnull().sum()}")
        
        # Convert TotalCharges to numeric (some might be strings)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Fill missing TotalCharges with 0 (new customers)
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        # Convert SeniorCitizen to categorical
        df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        
        print(f"Missing values after cleaning:\n{df.isnull().sum()}")
        
        return df
    
    def feature_engineering(self, df):
        """Create new features"""
        print("Creating engineered features...")
        
        df = df.copy()
        
        # 1. Tenure groups
        df['TenureGroup'] = pd.cut(df['tenure'], 
                                  bins=[0, 12, 24, 48, 72], 
                                  labels=['0-1 year', '1-2 years', '2-4 years', '4+ years'])
        
        # 2. Monthly charges groups
        df['ChargesGroup'] = pd.cut(df['MonthlyCharges'], 
                                   bins=[0, 35, 65, 90, 120], 
                                   labels=['Low', 'Medium', 'High', 'Very High'])
        
        # 3. Average charges per month
        df['AvgChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)  # +1 to avoid division by zero
        
        # 4. Customer value score (simple scoring)
        df['CustomerValue'] = (
            df['tenure'] * 0.3 + 
            df['TotalCharges'] / 1000 * 0.4 + 
            df['NumServices'] * 0.3
        )
        
        # 5. Service adoption rate
        total_possible_services = 8  # Total number of services offered
        df['ServiceAdoptionRate'] = df['NumServices'] / total_possible_services
        
        # 6. High value customer flag
        df['HighValueCustomer'] = (df['TotalCharges'] > df['TotalCharges'].quantile(0.75)).astype(int)
        
        # 7. Support intensity
        df['SupportIntensity'] = df['SupportCalls'] / (df['tenure'] + 1)
        
        # 8. Satisfaction category
        df['SatisfactionCategory'] = pd.cut(df['SatisfactionScore'], 
                                          bins=[0, 4, 7, 10], 
                                          labels=['Low', 'Medium', 'High'])
        
        # 9. Contract risk (month-to-month is risky)
        df['ContractRisk'] = (df['Contract'] == 'Month-to-month').astype(int)
        
        # 10. Payment risk (electronic check is risky)
        df['PaymentRisk'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
        
        print(f"Added engineered features. Dataset now has {len(df.columns)} columns")
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        print("Encoding categorical features...")
        
        df = df.copy()
        
        # Identify categorical columns (excluding target)
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if self.target_column in categorical_columns:
            categorical_columns.remove(self.target_column)
        if 'customerID' in categorical_columns:
            categorical_columns.remove('customerID')
            
        print(f"Encoding {len(categorical_columns)} categorical columns")
        
        # Encode categorical features
        for col in categorical_columns:
            if fit:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    unique_values = set(df[col].astype(str).unique())
                    known_values = set(self.label_encoders[col].classes_)
                    
                    if unique_values - known_values:
                        print(f"Warning: Unseen categories in {col}: {unique_values - known_values}")
                        # Map unseen categories to most frequent category
                        most_frequent = self.label_encoders[col].classes_[0]
                        df[col] = df[col].astype(str).apply(
                            lambda x: x if x in known_values else most_frequent
                        )
                    
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def encode_target(self, df, fit=True):
        """Encode target variable"""
        if fit:
            self.target_encoder = LabelEncoder()
            df[self.target_column] = self.target_encoder.fit_transform(df[self.target_column])
        else:
            df[self.target_column] = self.target_encoder.transform(df[self.target_column])
        
        return df
    
    def scale_features(self, df, fit=True):
        """Scale numerical features"""
        print("Scaling numerical features...")
        
        # Identify numerical columns
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if self.target_column in numerical_columns:
            numerical_columns.remove(self.target_column)
        if 'customerID' in df.columns:
            numerical_columns = [col for col in numerical_columns if col != 'customerID']
            
        print(f"Scaling {len(numerical_columns)} numerical columns")
        
        if fit:
            df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        else:
            df[numerical_columns] = self.scaler.transform(df[numerical_columns])
            
        return df
    
    def prepare_features(self, df):
        """Prepare final feature set"""
        # Remove non-feature columns
        columns_to_drop = ['customerID']
        df_features = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # Store feature columns
        self.feature_columns = [col for col in df_features.columns if col != self.target_column]
        
        print(f"Final feature set: {len(self.feature_columns)} features")
        
        return df_features
    
    def split_data(self, df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
        """Split data into train/test sets"""
        print(f"Splitting data with test_size={test_size}")
        
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Churn rate in training: {y_train.mean():.2%}")
        print(f"Churn rate in test: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self):
        """Save preprocessing objects"""
        print("Saving preprocessing objects...")
        
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(self.label_encoders, LABEL_ENCODERS_PATH)
        
        # Save feature columns and target encoder
        preprocessing_info = {
            'feature_columns': self.feature_columns,
            'target_encoder': self.target_encoder,
            'target_column': self.target_column
        }
        joblib.dump(preprocessing_info, MODELS_DIR / 'preprocessing_info.joblib')
        
        print("Preprocessing objects saved successfully")
    
    def load_preprocessor(self):
        """Load preprocessing objects"""
        print("Loading preprocessing objects...")
        
        self.scaler = joblib.load(SCALER_PATH)
        self.label_encoders = joblib.load(LABEL_ENCODERS_PATH)
        
        preprocessing_info = joblib.load(MODELS_DIR / 'preprocessing_info.joblib')
        self.feature_columns = preprocessing_info['feature_columns']
        self.target_encoder = preprocessing_info['target_encoder']
        self.target_column = preprocessing_info['target_column']
        
        print("Preprocessing objects loaded successfully")
    
    def process_pipeline(self, df=None, save_processed=True):
        """Complete preprocessing pipeline"""
        print("=" * 50)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("=" * 50)
        
        if df is None:
            df = self.load_data()
        
        # Preprocessing steps
        df = self.basic_preprocessing(df)
        df = self.feature_engineering(df)
        df = self.encode_categorical_features(df, fit=True)
        df = self.encode_target(df, fit=True)
        df = self.scale_features(df, fit=True)
        df = self.prepare_features(df)
        
        # Save processed data
        if save_processed:
            df.to_csv(PROCESSED_DATA_PATH, index=False)
            print(f"Processed data saved to {PROCESSED_DATA_PATH}")
        
        # Save preprocessing objects
        self.save_preprocessor()
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        print("=" * 50)
        print("PREPROCESSING PIPELINE COMPLETED")
        print("=" * 50)
        
        return X_train, X_test, y_train, y_test, df
    
    def transform_new_data(self, df):
        """Transform new data using fitted preprocessors"""
        print("Transforming new data...")
        
        # Load preprocessors if not already loaded
        if not hasattr(self, 'target_encoder'):
            self.load_preprocessor()
        
        # Apply same preprocessing steps (without fitting)
        df = self.basic_preprocessing(df)
        df = self.feature_engineering(df)
        df = self.encode_categorical_features(df, fit=False)
        
        # Don't encode target if it doesn't exist (prediction data)
        if self.target_column in df.columns:
            df = self.encode_target(df, fit=False)
        
        df = self.scale_features(df, fit=False)
        df = self.prepare_features(df)
        
        # Return only feature columns
        return df[self.feature_columns]

def main():
    """Main preprocessing function"""
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, processed_df = preprocessor.process_pipeline()
    
    print("\nPreprocessing Summary:")
    print(f"Total samples: {len(processed_df)}")
    print(f"Features: {len(preprocessor.feature_columns)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature columns: {preprocessor.feature_columns[:10]}...")  # Show first 10

if __name__ == "__main__":
    main()



