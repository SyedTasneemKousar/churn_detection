"""
Streamlit Dashboard for Customer Churn Prediction System
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import requests
import warnings
warnings.filterwarnings('ignore')

from config import *
from data_preprocessing import DataPreprocessor
from ml_models import ChurnMLModels
from hybrid_model import HybridChurnModel
from nlp_sentiment import SentimentAnalyzer

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
          
          <style>
    /* General metric card styling */
    .metric-card {
        background-color: #1e1e1e; /* Darker card for contrast */
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
        color: #f5f5f5 !important; /* Light text */
    }

    .metric-card h4 {
        color: #1f77b4 !important;
        margin-bottom: 0.5rem;
    }

    .metric-card p {
        color: #f5f5f5 !important; /* Ensure paragraph text is visible */
        margin: 0.25rem 0;
    }

    /* Prediction colors */
    .prediction-high {
        color: #ff4c4c !important; /* Bright red */
        font-weight: bold;
    }

    .prediction-low {
        color: #4cd964 !important; /* Bright green */
        font-weight: bold;
    }

    .prediction-medium {
        color: #ff9500 !important; /* Bright orange */
        font-weight: bold;
    }

    /* Override Streamlit default text colors */
    .stMarkdown p {
        color: #f5f5f5 !important;
    }

    .stMarkdown h4 {
        color: #1f77b4 !important;
    }

    /* Ensure all text in metric cards is visible */
    .element-container .metric-card * {
        color: #f5f5f5 !important;
    }

    .element-container .metric-card .prediction-high {
        color: #ff4c4c !important;
    }

    .element-container .metric-card .prediction-low {
        color: #4cd964 !important;
    }

    .element-container .metric-card .prediction-medium {
        color: #ff9500 !important;
    }

    /* Readability fixes scoped to Risk Factors and Retention sections */
    .risk-factors * { color: #e5e7eb !important; }
    .retention-recs * { color: #e5e7eb !important; }
    .risk-factors h4, .retention-recs h4 { color: #ffffff !important; }
    /* Ensure Streamlit alert text inside banner remains visible */
    .stAlert div, .stAlert p, .stAlert span, .stAlert li { color: #e5e7eb !important; }

    /* Header */
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Ensure required dataset exists in Streamlit Cloud by auto-downloading if missing
def _download_if_missing():
    try:
        if not RAW_DATA_PATH.exists():
            url = st.secrets.get("RAW_DATA_URL", os.getenv("RAW_DATA_URL", ""))
            if not url:
                return  # No URL configured; leave default behavior
            DATA_DIR.mkdir(exist_ok=True)
            st.info("Downloading dataset...")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            RAW_DATA_PATH.write_bytes(resp.content)
            st.success(f"Dataset downloaded to {RAW_DATA_PATH}")
    except Exception as e:
        st.warning(f"Dataset download skipped: {e}")

_download_if_missing()

class ChurnDashboard:
    """Streamlit dashboard for churn prediction"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.ml_models = ChurnMLModels()
        self.hybrid_model = HybridChurnModel()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Load data and models
        self.load_dashboard_data()
        self.load_models()
    
    def load_dashboard_data(self):
        """Load data for dashboard"""
        try:
            self.df_raw = pd.read_csv(RAW_DATA_PATH)
            self.df_processed = pd.read_csv(PROCESSED_DATA_PATH)
            
            # Try to load sentiment data
            try:
                self.df_sentiment = pd.read_csv(DATA_DIR / 'customer_sentiment_scores.csv')
                self.has_sentiment_data = True
            except FileNotFoundError:
                self.df_sentiment = None
                self.has_sentiment_data = False
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()
    
    def load_models(self):
        """Load trained models"""
        try:
            # Load preprocessor
            self.preprocessor.load_preprocessor()
            
            # Load ML models
            self.ml_models.load_trained_models()
            
            # Load hybrid model if available
            self.hybrid_model.load_hybrid_model()
            
        except Exception as e:
            st.warning(f"Some models could not be loaded: {e}")
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = len(self.df_raw)
            st.metric("Total Customers", f"{total_customers:,}")
        
        with col2:
            churn_rate = (self.df_raw['Churn'] == 'Yes').mean()
            st.metric("Overall Churn Rate", f"{churn_rate:.1%}")
        
        with col3:
            avg_tenure = self.df_raw['tenure'].mean()
            st.metric("Average Tenure", f"{avg_tenure:.1f} months")
        
        with col4:
            avg_monthly_charges = self.df_raw['MonthlyCharges'].mean()
            st.metric("Avg Monthly Charges", f"${avg_monthly_charges:.2f}")
    
    def render_sidebar(self):
        """Render sidebar with navigation"""
        st.sidebar.title("Navigation")
        
        page_options = [
            "📊 Overview",
            "🔍 Data Exploration", 
            "🤖 Model Performance",
            "🎯 Churn Prediction",
            "💭 Sentiment Analysis",
            "👥 Customer Segmentation"
        ]
        
        selected_page = st.sidebar.selectbox("Select Page", page_options)
        
        return selected_page
    
    def render_overview_page(self):
        """Render overview page"""
        st.header("📊 Business Overview")
        
        # Churn distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Churn Distribution")
            churn_counts = self.df_raw['Churn'].value_counts()
            fig = px.pie(values=churn_counts.values, names=churn_counts.index, 
                        title="Customer Churn Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Churn by Contract Type")
            contract_churn = self.df_raw.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean())
            fig = px.bar(x=contract_churn.index, y=contract_churn.values,
                        title="Churn Rate by Contract Type",
                        labels={'x': 'Contract Type', 'y': 'Churn Rate'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Revenue analysis
        st.subheader("💰 Revenue Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly charges distribution
            fig = px.histogram(self.df_raw, x='MonthlyCharges', color='Churn',
                             title="Monthly Charges Distribution by Churn")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Revenue at risk
            churned_revenue = self.df_raw[self.df_raw['Churn'] == 'Yes']['MonthlyCharges'].sum()
            total_revenue = self.df_raw['MonthlyCharges'].sum()
            revenue_at_risk = churned_revenue / total_revenue
            
            st.metric("Monthly Revenue at Risk", f"${churned_revenue:,.2f}", 
                     f"{revenue_at_risk:.1%} of total revenue")
            
            # Tenure analysis
            fig = px.box(self.df_raw, x='Churn', y='tenure',
                        title="Customer Tenure by Churn Status")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_data_exploration_page(self):
        """Render data exploration page"""
        st.header("🔍 Data Exploration")
        
        # Feature selection
        st.subheader("Explore Features")
        
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyUsage', 
                            'SupportCalls', 'SatisfactionScore', 'NumServices']
        categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Contract',
                              'PaymentMethod', 'InternetService']
        
        tab1, tab2, tab3 = st.tabs(["Numerical Features", "Categorical Features", "Correlation Analysis"])
        
        with tab1:
            selected_num_feature = st.selectbox("Select Numerical Feature", numerical_features)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(self.df_raw, x=selected_num_feature, color='Churn',
                                 title=f"{selected_num_feature} Distribution by Churn")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(self.df_raw, x='Churn', y=selected_num_feature,
                           title=f"{selected_num_feature} by Churn Status")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            selected_cat_feature = st.selectbox("Select Categorical Feature", categorical_features)
            
            # Churn rate by category
            churn_by_category = self.df_raw.groupby(selected_cat_feature)['Churn'].apply(
                lambda x: (x == 'Yes').mean()
            )
            
            fig = px.bar(x=churn_by_category.index, y=churn_by_category.values,
                        title=f"Churn Rate by {selected_cat_feature}",
                        labels={'x': selected_cat_feature, 'y': 'Churn Rate'})
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Feature Correlations")
            
            # Prepare numerical data for correlation
            df_corr = self.df_raw[numerical_features + ['Churn']].copy()
            df_corr['Churn'] = (df_corr['Churn'] == 'Yes').astype(int)
            
            correlation_matrix = df_corr.corr()
            
            fig = px.imshow(correlation_matrix, 
                          title="Feature Correlation Matrix",
                          aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_model_performance_page(self):
        """Render model performance page"""
        st.header("🤖 Model Performance")
        
        # Load model performance data
        try:
            model_summary = pd.read_csv(MODELS_DIR / 'model_summary.csv')
            st.subheader("Model Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(model_summary, x='Model', y='ROC_AUC',
                           title="Model Performance - ROC-AUC")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(model_summary, x='Model', y='F1_Score',
                           title="Model Performance - F1_Score")
                st.plotly_chart(fig, use_container_width=True)
            
            # Model details table
            st.subheader("Detailed Performance Metrics")
            st.dataframe(model_summary, use_container_width=True)
            
        except FileNotFoundError:
            st.warning("Model performance data not available. Please train the models first.")
        
        # Feature importance (if available)
        try:
            if hasattr(self.ml_models, 'models') and 'xgboost' in self.ml_models.models:
                st.subheader("Feature Importance (XGBoost)")
                
                model = self.ml_models.models['xgboost']
                importance = model.feature_importances_
                feature_names = self.preprocessor.feature_columns
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False).head(15)
                
                fig = px.bar(importance_df, x='importance', y='feature',
                           orientation='h', title="Top 15 Feature Importance")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.info("Feature importance analysis not available.")
    
    def render_prediction_page(self):
        """Render churn prediction page"""
        st.header("🎯 Customer Churn Prediction")
        
        st.subheader("Predict Churn for Individual Customer")
        
        # Customer input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Demographics**")
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
                partner = st.selectbox("Partner", ["No", "Yes"])
                dependents = st.selectbox("Dependents", ["No", "Yes"])
            
            with col2:
                st.write("**Account Information**")
                tenure = st.slider("Tenure (months)", 1, 72, 24)
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                payment_method = st.selectbox("Payment Method", 
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            
            with col3:
                st.write("**Services & Charges**")
                monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
                internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
                phone_service = st.selectbox("Phone Service", ["No", "Yes"])
                support_calls = st.slider("Support Calls", 0, 10, 2)
                satisfaction_score = st.slider("Satisfaction Score", 1.0, 10.0, 7.0)
            
            predict_button = st.form_submit_button("Predict Churn")
        
        if predict_button:
            # Create customer data
            customer_data = pd.DataFrame({
                'customerID': ['PRED_001'],
                'gender': [gender],
                'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
                'Partner': [partner],
                'Dependents': [dependents],
                'tenure': [tenure],
                'PhoneService': [phone_service],
                'MultipleLines': ['No' if phone_service == 'No' else 'No'],
                'InternetService': [internet_service],
                'OnlineSecurity': ['No'],
                'OnlineBackup': ['No'],
                'DeviceProtection': ['No'],
                'TechSupport': ['No'],
                'StreamingTV': ['No'],
                'StreamingMovies': ['No'],
                'Contract': [contract],
                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [monthly_charges * tenure],
                'AvgMonthlyUsage': [50.0],
                'SupportCalls': [support_calls],
                'SatisfactionScore': [satisfaction_score],
                'NumServices': [3]
            })
            
            try:
                # Make predictions using different models
                st.subheader("Prediction Results")
                
                # Traditional ML models
                if hasattr(self.ml_models, 'models') and self.ml_models.models:
                    ml_predictions = self.ml_models.predict_churn(customer_data)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    for i, (model_name, prob) in enumerate(ml_predictions.items()):
                        col = [col1, col2, col3][i % 3]
                        with col:
                            churn_prob = prob[0] if isinstance(prob, np.ndarray) else prob
                            
                            if churn_prob > 0.7:
                                risk_level = "HIGH RISK"
                                color = "🔴"
                                css_class = "prediction-high"
                            elif churn_prob > 0.4:
                                risk_level = "MEDIUM RISK"
                                color = "🟡"
                                css_class = "prediction-medium"
                            else:
                                risk_level = "LOW RISK"
                                color = "🟢"
                                css_class = "prediction-low"
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>{model_name.replace('_', ' ').title()}</h4>
                                <p class="{css_class}">{color} {risk_level}</p>
                                <p>Churn Probability: {churn_prob:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Risk factors analysis
                st.subheader("Risk Factors Analysis")
                
                risk_factors = []
                if monthly_charges > 80:
                    risk_factors.append("High monthly charges")
                if tenure < 12:
                    risk_factors.append("Short tenure (new customer)")
                if contract == "Month-to-month":
                    risk_factors.append("Month-to-month contract")
                if support_calls > 3:
                    risk_factors.append("High number of support calls")
                if satisfaction_score < 6:
                    risk_factors.append("Low satisfaction score")
                if payment_method == "Electronic check":
                    risk_factors.append("Electronic check payment method")
                
                if risk_factors:
                    st.warning("⚠️ **Risk Factors Identified:**")
                    st.markdown('<div class="risk-factors">', unsafe_allow_html=True)
                    st.markdown(
                        "<ul>" + "".join([f'<li>{factor}</li>' for factor in risk_factors]) + "</ul>",
                        unsafe_allow_html=True,
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.success("✅ **No major risk factors identified**")
                
                # Recommendations
                st.subheader("Retention Recommendations")
                
                recommendations = []
                if monthly_charges > 80:
                    recommendations.append("Consider offering a discount or loyalty program")
                if contract == "Month-to-month":
                    recommendations.append("Incentivize longer-term contract commitment")
                if support_calls > 3:
                    recommendations.append("Proactive customer service outreach")
                if satisfaction_score < 6:
                    recommendations.append("Conduct satisfaction survey and address concerns")
                if tenure < 12:
                    recommendations.append("Implement new customer onboarding program")
                
                if recommendations:
                    st.markdown('<div class="retention-recs">', unsafe_allow_html=True)
                    for rec in recommendations:
                        st.markdown(f'<p>💡 {rec}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("Customer appears to be in good standing. Continue regular engagement.")
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    def render_sentiment_page(self):
        """Render sentiment analysis page"""
        st.header("💭 Sentiment Analysis")
        
        if not self.has_sentiment_data:
            st.warning("Sentiment analysis data not available. Please run the NLP sentiment analysis first.")
            return
        
        st.subheader("Customer Sentiment Overview")
        
        # Load feedback data
        try:
            df_feedback = pd.read_csv(DATA_DIR / 'processed_feedback_sentiment.csv')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment distribution
                sentiment_counts = df_feedback['predicted_sentiment'].value_counts()
                fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                           title="Overall Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sentiment over time
                df_feedback['timestamp'] = pd.to_datetime(df_feedback['timestamp'])
                daily_sentiment = df_feedback.groupby(
                    df_feedback['timestamp'].dt.date
                )['sentiment_score'].mean()
                
                fig = px.line(x=daily_sentiment.index, y=daily_sentiment.values,
                            title="Average Sentiment Over Time")
                st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment vs Churn analysis
            st.subheader("Sentiment Impact on Churn")
            
            # Merge sentiment with churn data
            df_merged = self.df_raw.merge(self.df_sentiment, on='customerID', how='inner')
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(df_merged, x='Churn', y='avg_sentiment',
                           title="Average Sentiment by Churn Status")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Churn rate by sentiment category
                df_merged['sentiment_category'] = pd.cut(
                    df_merged['avg_sentiment'], 
                    bins=[-1, -0.1, 0.1, 1], 
                    labels=['Negative', 'Neutral', 'Positive']
                )
                
                churn_by_sentiment = df_merged.groupby('sentiment_category')['Churn'].apply(
                    lambda x: (x == 'Yes').mean()
                )
                
                fig = px.bar(x=churn_by_sentiment.index, y=churn_by_sentiment.values,
                           title="Churn Rate by Sentiment Category")
                st.plotly_chart(fig, use_container_width=True)
            
        except FileNotFoundError:
            st.error("Feedback sentiment data not found.")
    
    def render_segmentation_page(self):
        """Render customer segmentation page"""
        st.header("👥 Customer Segmentation")
        
        st.subheader("Customer Segments Analysis")
        
        # Create customer segments based on key features
        df_segment = self.df_raw.copy()
        
        # Tenure segments
        df_segment['tenure_segment'] = pd.cut(
            df_segment['tenure'], 
            bins=[0, 12, 24, 48, 72], 
            labels=['New (0-1y)', 'Growing (1-2y)', 'Mature (2-4y)', 'Loyal (4y+)']
        )
        
        # Revenue segments
        df_segment['revenue_segment'] = pd.cut(
            df_segment['MonthlyCharges'], 
            bins=[0, 35, 65, 90, 120], 
            labels=['Low', 'Medium', 'High', 'Premium']
        )
        
        tab1, tab2, tab3 = st.tabs(["Tenure Segments", "Revenue Segments", "Risk Segments"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                segment_counts = df_segment['tenure_segment'].value_counts()
                fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                           title="Customer Distribution by Tenure Segment")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                churn_by_tenure = df_segment.groupby('tenure_segment')['Churn'].apply(
                    lambda x: (x == 'Yes').mean()
                )
                fig = px.bar(x=churn_by_tenure.index, y=churn_by_tenure.values,
                           title="Churn Rate by Tenure Segment")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                segment_counts = df_segment['revenue_segment'].value_counts()
                fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                           title="Customer Distribution by Revenue Segment")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                churn_by_revenue = df_segment.groupby('revenue_segment')['Churn'].apply(
                    lambda x: (x == 'Yes').mean()
                )
                fig = px.bar(x=churn_by_revenue.index, y=churn_by_revenue.values,
                           title="Churn Rate by Revenue Segment")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Risk-based segmentation
            df_risk = df_segment.copy()
            
            # Calculate risk score
            risk_score = 0
            risk_score += (df_risk['Contract'] == 'Month-to-month') * 0.3
            risk_score += (df_risk['tenure'] < 12) * 0.2
            risk_score += (df_risk['MonthlyCharges'] > 80) * 0.2
            risk_score += (df_risk['SupportCalls'] > 3) * 0.15
            risk_score += (df_risk['SatisfactionScore'] < 6) * 0.15
            
            df_risk['risk_score'] = risk_score
            df_risk['risk_segment'] = pd.cut(
                df_risk['risk_score'], 
                bins=[0, 0.3, 0.6, 1.0], 
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                segment_counts = df_risk['risk_segment'].value_counts()
                fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                           title="Customer Distribution by Risk Segment")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                actual_churn_by_risk = df_risk.groupby('risk_segment')['Churn'].apply(
                    lambda x: (x == 'Yes').mean()
                )
                fig = px.bar(x=actual_churn_by_risk.index, y=actual_churn_by_risk.values,
                           title="Actual Churn Rate by Risk Segment")
                st.plotly_chart(fig, use_container_width=True)
    
    def run_dashboard(self):
        """Run the Streamlit dashboard"""
        # Render header
        self.render_header()
        
        # Render sidebar and get selected page
        selected_page = self.render_sidebar()
        
        # Render selected page
        if selected_page == "📊 Overview":
            self.render_overview_page()
        elif selected_page == "🔍 Data Exploration":
            self.render_data_exploration_page()
        elif selected_page == "🤖 Model Performance":
            self.render_model_performance_page()
        elif selected_page == "🎯 Churn Prediction":
            self.render_prediction_page()
        elif selected_page == "💭 Sentiment Analysis":
            self.render_sentiment_page()
        elif selected_page == "👥 Customer Segmentation":
            self.render_segmentation_page()

def main():
    """Main function to run the dashboard"""
    dashboard = ChurnDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()




