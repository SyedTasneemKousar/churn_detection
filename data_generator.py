"""
Generate sample telecom churn dataset and customer feedback data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_telecom_churn_data(n_customers=10000):
    """Generate synthetic telecom churn dataset"""
    
    # Customer demographics
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]
    
    # Demographics
    gender = np.random.choice(['Male', 'Female'], n_customers, p=[0.5, 0.5])
    senior_citizen = np.random.choice([0, 1], n_customers, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], n_customers, p=[0.52, 0.48])
    dependents = np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7])
    
    # Account information
    tenure = np.random.exponential(scale=24, size=n_customers).astype(int)
    tenure = np.clip(tenure, 1, 72)  # 1-72 months
    
    # Services
    phone_service = np.random.choice(['Yes', 'No'], n_customers, p=[0.9, 0.1])
    multiple_lines = np.random.choice(['Yes', 'No', 'No phone service'], n_customers, p=[0.42, 0.48, 0.1])
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers, p=[0.34, 0.44, 0.22])
    online_security = np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.28, 0.5, 0.22])
    online_backup = np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.34, 0.44, 0.22])
    device_protection = np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.34, 0.44, 0.22])
    tech_support = np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.29, 0.49, 0.22])
    streaming_tv = np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.38, 0.4, 0.22])
    streaming_movies = np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.38, 0.4, 0.22])
    
    # Contract and billing
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers, p=[0.55, 0.21, 0.24])
    paperless_billing = np.random.choice(['Yes', 'No'], n_customers, p=[0.59, 0.41])
    payment_method = np.random.choice([
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ], n_customers, p=[0.34, 0.19, 0.22, 0.25])
    
    # Charges
    monthly_charges = np.random.normal(65, 30, n_customers)
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)
    
    total_charges = monthly_charges * tenure + np.random.normal(0, 100, n_customers)
    total_charges = np.maximum(total_charges, monthly_charges)
    
    # Usage patterns
    avg_monthly_usage = np.random.exponential(scale=50, size=n_customers)  # GB
    support_calls = np.random.poisson(lam=2, size=n_customers)
    
    # Customer satisfaction score (1-10)
    satisfaction_score = np.random.normal(7, 2, n_customers)
    satisfaction_score = np.clip(satisfaction_score, 1, 10)
    
    # Number of products/services
    num_services = (
        (phone_service == 'Yes').astype(int) +
        (internet_service != 'No').astype(int) +
        (online_security == 'Yes').astype(int) +
        (online_backup == 'Yes').astype(int) +
        (device_protection == 'Yes').astype(int) +
        (tech_support == 'Yes').astype(int) +
        (streaming_tv == 'Yes').astype(int) +
        (streaming_movies == 'Yes').astype(int)
    )
    
    # Generate churn based on realistic factors
    churn_probability = 0.1  # Base churn rate
    
    # Factors that increase churn probability
    churn_probability += (contract == 'Month-to-month') * 0.25
    churn_probability += (tenure < 12) * 0.2
    churn_probability += (monthly_charges > 80) * 0.15
    churn_probability += (support_calls > 5) * 0.2
    churn_probability += (satisfaction_score < 5) * 0.3
    churn_probability += (num_services < 3) * 0.1
    churn_probability += (senior_citizen == 1) * 0.1
    churn_probability += (payment_method == 'Electronic check') * 0.1
    
    # Factors that decrease churn probability
    churn_probability -= (contract == 'Two year') * 0.2
    churn_probability -= (tenure > 36) * 0.15
    churn_probability -= (partner == 'Yes') * 0.05
    churn_probability -= (dependents == 'Yes') * 0.05
    churn_probability -= (satisfaction_score > 8) * 0.15
    
    churn_probability = np.clip(churn_probability, 0.01, 0.8)
    
    # Generate actual churn
    churn = np.random.binomial(1, churn_probability, n_customers)
    churn_labels = ['Yes' if c == 1 else 'No' for c in churn]
    
    # Create DataFrame
    df = pd.DataFrame({
        'customerID': customer_ids,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': np.round(monthly_charges, 2),
        'TotalCharges': np.round(total_charges, 2),
        'AvgMonthlyUsage': np.round(avg_monthly_usage, 2),
        'SupportCalls': support_calls,
        'SatisfactionScore': np.round(satisfaction_score, 1),
        'NumServices': num_services,
        'Churn': churn_labels
    })
    
    return df

def generate_customer_feedback_data(customer_ids, n_feedback=5000):
    """Generate synthetic customer feedback/support tickets"""
    
    # Sample customer IDs for feedback
    feedback_customers = np.random.choice(customer_ids, n_feedback, replace=True)
    
    # Positive feedback templates
    positive_feedback = [
        "Great service, very satisfied with the internet speed and customer support.",
        "Love the streaming quality and the pricing is reasonable.",
        "Excellent customer service, resolved my issue quickly.",
        "Very happy with the fiber optic connection, fast and reliable.",
        "Good value for money, all services work perfectly.",
        "Customer support was helpful and professional.",
        "Internet service is consistent and fast.",
        "Happy with the bundle package, saves money.",
        "Technical support resolved the issue efficiently.",
        "Satisfied with the service quality and billing."
    ]
    
    # Negative feedback templates
    negative_feedback = [
        "Frequent internet outages, very frustrating experience.",
        "Customer service is terrible, long wait times and unhelpful staff.",
        "Overpriced for the service quality provided.",
        "Internet speed is much slower than advertised.",
        "Billing issues every month, incorrect charges.",
        "Poor technical support, couldn't resolve my problem.",
        "Service interruptions during important calls.",
        "Hidden fees not mentioned during signup.",
        "Difficulty canceling services, poor customer experience.",
        "Internet connection drops frequently, unreliable service."
    ]
    
    # Neutral feedback templates
    neutral_feedback = [
        "Service is okay, nothing special but works fine.",
        "Average internet speed, meets basic needs.",
        "Customer service is adequate, could be better.",
        "Pricing is competitive but service could improve.",
        "Internet works most of the time, occasional issues.",
        "Billing is usually correct, few minor issues.",
        "Service is reliable during weekdays.",
        "Customer support is polite but slow to resolve issues.",
        "Internet speed varies throughout the day.",
        "Service meets expectations, nothing outstanding."
    ]
    
    # Generate feedback with sentiment distribution
    sentiments = np.random.choice(['positive', 'negative', 'neutral'], n_feedback, p=[0.4, 0.35, 0.25])
    
    feedback_texts = []
    for sentiment in sentiments:
        if sentiment == 'positive':
            feedback_texts.append(random.choice(positive_feedback))
        elif sentiment == 'negative':
            feedback_texts.append(random.choice(negative_feedback))
        else:
            feedback_texts.append(random.choice(neutral_feedback))
    
    # Generate timestamps (last 2 years)
    start_date = datetime.now() - timedelta(days=730)
    timestamps = [start_date + timedelta(days=random.randint(0, 730)) for _ in range(n_feedback)]
    
    # Create DataFrame
    feedback_df = pd.DataFrame({
        'customerID': feedback_customers,
        'feedback_text': feedback_texts,
        'timestamp': timestamps,
        'true_sentiment': sentiments
    })
    
    return feedback_df

def main():
    """Generate and save datasets"""
    print("Generating telecom churn dataset...")
    churn_df = generate_telecom_churn_data(n_customers=10000)
    
    print("Generating customer feedback data...")
    feedback_df = generate_customer_feedback_data(churn_df['customerID'].tolist(), n_feedback=5000)
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Save datasets
    churn_df.to_csv(data_dir / "telecom_churn.csv", index=False)
    feedback_df.to_csv(data_dir / "customer_feedback.csv", index=False)
    
    print(f"Generated datasets:")
    print(f"- Churn dataset: {len(churn_df)} customers")
    print(f"- Feedback dataset: {len(feedback_df)} feedback records")
    print(f"- Churn rate: {(churn_df['Churn'] == 'Yes').mean():.2%}")
    print(f"- Files saved in 'data' directory")

if __name__ == "__main__":
    main()




