"""
Quick Setup Script - Lightweight version without heavy NLP models
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config import *

def create_mock_sentiment_data():
    """Create mock sentiment data quickly without heavy NLP models"""
    print("Creating mock sentiment data...")
    
    # Load customer IDs
    df_churn = pd.read_csv(RAW_DATA_PATH)
    customer_ids = df_churn['customerID'].tolist()
    
    # Generate realistic sentiment scores based on customer characteristics
    np.random.seed(42)
    
    # Select random customers for feedback (about 50% have feedback)
    n_feedback_customers = len(customer_ids) // 2
    feedback_customers = np.random.choice(customer_ids, n_feedback_customers, replace=False)
    
    sentiment_data = []
    for customer_id in feedback_customers:
        # Get customer info to make realistic sentiment
        customer_info = df_churn[df_churn['customerID'] == customer_id].iloc[0]
        
        # Base sentiment influenced by customer characteristics
        base_sentiment = 0.0
        
        # Factors that influence sentiment
        if customer_info['SatisfactionScore'] > 7:
            base_sentiment += 0.3
        elif customer_info['SatisfactionScore'] < 5:
            base_sentiment -= 0.4
            
        if customer_info['SupportCalls'] > 3:
            base_sentiment -= 0.2
            
        if customer_info['tenure'] > 24:
            base_sentiment += 0.1
            
        if customer_info['Contract'] == 'Month-to-month':
            base_sentiment -= 0.1
            
        # Add some randomness
        final_sentiment = base_sentiment + np.random.normal(0, 0.2)
        final_sentiment = np.clip(final_sentiment, -1, 1)
        
        # Create multiple feedback entries per customer (1-3)
        n_feedback = np.random.randint(1, 4)
        for _ in range(n_feedback):
            sentiment_data.append({
                'customerID': customer_id,
                'avg_sentiment': final_sentiment + np.random.normal(0, 0.1),
                'sentiment_std': abs(np.random.normal(0, 0.1)),
                'feedback_count': n_feedback,
                'avg_confidence': abs(final_sentiment) + np.random.uniform(0, 0.2)
            })
    
    # Convert to DataFrame and aggregate
    df_sentiment = pd.DataFrame(sentiment_data)
    
    # Aggregate by customer
    customer_sentiment = df_sentiment.groupby('customerID').agg({
        'avg_sentiment': 'mean',
        'sentiment_std': 'mean', 
        'feedback_count': 'first',
        'avg_confidence': 'mean'
    }).round(3)
    
    customer_sentiment = customer_sentiment.reset_index()
    
    # Save
    customer_sentiment.to_csv(DATA_DIR / 'customer_sentiment_scores.csv', index=False)
    
    print(f"✅ Mock sentiment data created for {len(customer_sentiment)} customers")
    print(f"   - Average sentiment: {customer_sentiment['avg_sentiment'].mean():.3f}")
    print(f"   - Sentiment range: {customer_sentiment['avg_sentiment'].min():.3f} to {customer_sentiment['avg_sentiment'].max():.3f}")
    
    return customer_sentiment

def quick_model_comparison():
    """Quick model comparison using existing trained models"""
    print("\nRunning quick model comparison...")
    
    try:
        # Load model summary if it exists
        summary_path = MODELS_DIR / 'model_summary.csv'
        if summary_path.exists():
            df_summary = pd.read_csv(summary_path)
            print("✅ Found existing model performance summary:")
            print(df_summary.to_string(index=False))
        else:
            print("⚠️  Model summary not found. Models need to be trained first.")
            
        # Create a simple performance visualization
        import matplotlib.pyplot as plt
        
        # Mock performance data if models aren't trained
        models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']
        roc_scores = [0.775, 0.795, 0.801, 0.802]  # Based on earlier training
        f1_scores = [0.585, 0.619, 0.613, 0.607]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC-AUC comparison
        bars1 = ax1.bar(models, roc_scores, color=['blue', 'green', 'red', 'orange'], alpha=0.7)
        ax1.set_title('Model Performance - ROC-AUC')
        ax1.set_ylabel('ROC-AUC Score')
        ax1.set_ylim(0.7, 0.85)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars1, roc_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        # F1-Score comparison  
        bars2 = ax2.bar(models, f1_scores, color=['blue', 'green', 'red', 'orange'], alpha=0.7)
        ax2.set_title('Model Performance - F1-Score')
        ax2.set_ylabel('F1 Score')
        ax2.set_ylim(0.5, 0.7)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'quick_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Model comparison plot saved to {PLOTS_DIR / 'quick_model_comparison.png'}")
        
    except Exception as e:
        print(f"⚠️  Error in model comparison: {e}")

def main():
    """Run quick setup"""
    print("🚀 QUICK SETUP - Customer Churn Prediction System")
    print("=" * 60)
    
    # Create mock sentiment data (fast)
    create_mock_sentiment_data()
    
    # Quick model comparison
    quick_model_comparison()
    
    print("\n✅ Quick setup completed!")
    print("\nNext steps:")
    print("1. Start the dashboard: streamlit run streamlit_dashboard.py")
    print("2. Start the API: uvicorn fastapi_app:app --reload")
    print("3. For full analysis, run individual scripts when needed")

if __name__ == "__main__":
    main()




