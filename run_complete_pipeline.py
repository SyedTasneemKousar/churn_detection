"""
Complete Pipeline Runner for Customer Churn Prediction System
"""
import os
import sys
import time
from pathlib import Path

def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "="*80)
    print(f"🎯 {text}")
    print("="*80)

def run_script(script_name, description):
    """Run a Python script with error handling"""
    print(f"\n📊 {description}...")
    print(f"Running: python {script_name}")
    
    start_time = time.time()
    result = os.system(f"python {script_name}")
    end_time = time.time()
    
    if result == 0:
        print(f"✅ {description} completed successfully in {end_time - start_time:.1f} seconds")
        return True
    else:
        print(f"❌ {description} failed with exit code {result}")
        return False

def main():
    """Run the complete pipeline"""
    print_banner("CUSTOMER CHURN PREDICTION SYSTEM - COMPLETE PIPELINE")
    
    # Check if we're in the right directory
    if not Path("config.py").exists():
        print("❌ Error: config.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Pipeline steps
    pipeline_steps = [
        ("data_generator.py", "Data Generation"),
        ("data_preprocessing.py", "Data Preprocessing & Feature Engineering"),
        ("eda_analysis.py", "Exploratory Data Analysis"),
        ("ml_models.py", "Traditional Machine Learning Models Training"),
        ("nlp_sentiment.py", "NLP Sentiment Analysis"),
        ("hybrid_model.py", "Hybrid Model Training"),
        ("customer_clustering.py", "Customer Clustering Analysis"),
        ("model_comparison.py", "Comprehensive Model Comparison")
    ]
    
    # Track success/failure
    results = {}
    total_start_time = time.time()
    
    # Run each step
    for script, description in pipeline_steps:
        success = run_script(script, description)
        results[description] = success
        
        if not success:
            print(f"\n⚠️  Warning: {description} failed. Continuing with next step...")
    
    # Summary
    total_time = time.time() - total_start_time
    print_banner("PIPELINE EXECUTION SUMMARY")
    
    successful_steps = sum(results.values())
    total_steps = len(results)
    
    print(f"📈 Pipeline Statistics:")
    print(f"   • Total Steps: {total_steps}")
    print(f"   • Successful: {successful_steps}")
    print(f"   • Failed: {total_steps - successful_steps}")
    print(f"   • Success Rate: {successful_steps/total_steps:.1%}")
    print(f"   • Total Time: {total_time/60:.1f} minutes")
    
    print(f"\n📊 Step Results:")
    for step, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   • {step}: {status}")
    
    # Generated files summary
    print_banner("GENERATED FILES SUMMARY")
    
    # Check data files
    data_files = [
        "data/telecom_churn.csv",
        "data/customer_feedback.csv", 
        "data/processed_churn.csv",
        "data/customer_sentiment_scores.csv",
        "data/customer_clusters.csv"
    ]
    
    print("📁 Data Files:")
    for file_path in data_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size / 1024  # KB
            print(f"   ✅ {file_path} ({size:.1f} KB)")
        else:
            print(f"   ❌ {file_path} (missing)")
    
    # Check model files
    model_files = [
        "models/logistic_regression.joblib",
        "models/random_forest.joblib",
        "models/xgboost.joblib",
        "models/lightgbm.joblib",
        "models/scaler.joblib",
        "models/hybrid_churn_model.joblib"
    ]
    
    print("\n🤖 Model Files:")
    for file_path in model_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size / 1024  # KB
            print(f"   ✅ {file_path} ({size:.1f} KB)")
        else:
            print(f"   ❌ {file_path} (missing)")
    
    # Check visualization files
    plot_files = [
        "plots/churn_analysis_dashboard.png",
        "plots/correlation_heatmap.png",
        "plots/model_comparison.png",
        "plots/feature_importance.png",
        "plots/comprehensive_model_comparison.png",
        "plots/sentiment_analysis.png",
        "plots/kmeans_clustering_analysis.png"
    ]
    
    print("\n📈 Visualization Files:")
    for file_path in plot_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size / 1024  # KB
            print(f"   ✅ {file_path} ({size:.1f} KB)")
        else:
            print(f"   ❌ {file_path} (missing)")
    
    # Next steps
    print_banner("NEXT STEPS")
    
    if successful_steps >= total_steps * 0.8:  # 80% success rate
        print("🎉 Pipeline completed successfully! You can now:")
        print("\n🌐 Start the applications:")
        print("   • Dashboard: streamlit run streamlit_dashboard.py")
        print("   • API: uvicorn fastapi_app:app --reload")
        print("   • Docker: docker-compose up -d")
        
        print("\n📊 Explore the results:")
        print("   • Check the 'plots' directory for visualizations")
        print("   • Review model performance in 'models/model_summary.csv'")
        print("   • Explore customer segments in 'data/customer_clusters.csv'")
        
        print("\n🔗 API Documentation:")
        print("   • Interactive docs: http://localhost:8000/docs")
        print("   • Health check: http://localhost:8000/health")
        
        print("\n📱 Dashboard Features:")
        print("   • Data exploration and visualization")
        print("   • Model performance comparison")
        print("   • Individual customer churn prediction")
        print("   • Customer segmentation analysis")
        
    else:
        print("⚠️  Pipeline completed with some failures.")
        print("Please check the error messages above and fix any issues.")
        print("You can re-run individual scripts or the complete pipeline.")
    
    print(f"\n🎯 Customer Churn Prediction System Setup Complete!")
    print(f"   📁 Project Directory: {Path.cwd()}")
    print(f"   ⏱️  Total Setup Time: {total_time/60:.1f} minutes")

if __name__ == "__main__":
    main()



