"""
Exploratory Data Analysis (EDA) for Customer Churn Prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from config import *

class ChurnEDA:
    """Comprehensive EDA for churn analysis"""
    
    def __init__(self, data_path=RAW_DATA_PATH):
        """Initialize with data"""
        self.df = pd.read_csv(data_path)
        self.prepare_data()
        
    def prepare_data(self):
        """Basic data preparation for EDA"""
        # Convert TotalCharges to numeric
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        self.df['TotalCharges'] = self.df['TotalCharges'].fillna(0)
        
        # Convert SeniorCitizen to categorical
        self.df['SeniorCitizen'] = self.df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        
        print(f"Dataset loaded: {self.df.shape[0]} customers, {self.df.shape[1]} features")
        print(f"Churn rate: {(self.df['Churn'] == 'Yes').mean():.2%}")
    
    def basic_statistics(self):
        """Generate basic statistics"""
        print("=" * 60)
        print("BASIC DATASET STATISTICS")
        print("=" * 60)
        
        # Dataset overview
        print(f"Total Customers: {len(self.df):,}")
        print(f"Total Features: {len(self.df.columns)}")
        print(f"Churned Customers: {(self.df['Churn'] == 'Yes').sum():,}")
        print(f"Retained Customers: {(self.df['Churn'] == 'No').sum():,}")
        print(f"Churn Rate: {(self.df['Churn'] == 'Yes').mean():.2%}")
        
        # Missing values
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\nMissing Values:")
            print(missing_values[missing_values > 0])
        else:
            print("\nNo missing values found!")
        
        # Numerical features summary
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyUsage', 'SupportCalls', 'SatisfactionScore']
        print(f"\nNumerical Features Summary:")
        print(self.df[numerical_cols].describe())
        
        return self.df.describe()
    
    def churn_analysis(self):
        """Analyze churn patterns"""
        print("\n" + "=" * 60)
        print("CHURN PATTERN ANALYSIS")
        print("=" * 60)
        
        # Churn by categorical features
        categorical_features = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]
        
        churn_analysis = {}
        
        for feature in categorical_features:
            churn_rate = self.df.groupby(feature)['Churn'].apply(lambda x: (x == 'Yes').mean())
            churn_analysis[feature] = churn_rate.to_dict()
            
            print(f"\nChurn Rate by {feature}:")
            for category, rate in churn_rate.items():
                print(f"  {category}: {rate:.2%}")
        
        return churn_analysis
    
    def create_churn_visualizations(self):
        """Create comprehensive churn visualizations"""
        print("\nGenerating churn visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Customer Churn Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Churn Distribution
        churn_counts = self.df['Churn'].value_counts()
        axes[0, 0].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Overall Churn Distribution')
        
        # 2. Churn by Contract Type
        contract_churn = pd.crosstab(self.df['Contract'], self.df['Churn'], normalize='index') * 100
        contract_churn.plot(kind='bar', ax=axes[0, 1], rot=45)
        axes[0, 1].set_title('Churn Rate by Contract Type')
        axes[0, 1].set_ylabel('Churn Rate (%)')
        axes[0, 1].legend(title='Churn')
        
        # 3. Churn by Tenure
        axes[0, 2].hist([self.df[self.df['Churn'] == 'No']['tenure'], 
                        self.df[self.df['Churn'] == 'Yes']['tenure']], 
                       bins=20, alpha=0.7, label=['No Churn', 'Churn'])
        axes[0, 2].set_title('Churn Distribution by Tenure')
        axes[0, 2].set_xlabel('Tenure (months)')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].legend()
        
        # 4. Monthly Charges Distribution
        axes[1, 0].boxplot([self.df[self.df['Churn'] == 'No']['MonthlyCharges'],
                           self.df[self.df['Churn'] == 'Yes']['MonthlyCharges']], 
                          labels=['No Churn', 'Churn'])
        axes[1, 0].set_title('Monthly Charges by Churn')
        axes[1, 0].set_ylabel('Monthly Charges ($)')
        
        # 5. Churn by Internet Service
        internet_churn = pd.crosstab(self.df['InternetService'], self.df['Churn'], normalize='index') * 100
        internet_churn.plot(kind='bar', ax=axes[1, 1], rot=45)
        axes[1, 1].set_title('Churn Rate by Internet Service')
        axes[1, 1].set_ylabel('Churn Rate (%)')
        axes[1, 1].legend(title='Churn')
        
        # 6. Support Calls vs Churn
        axes[1, 2].scatter(self.df[self.df['Churn'] == 'No']['SupportCalls'], 
                          self.df[self.df['Churn'] == 'No']['SatisfactionScore'], 
                          alpha=0.5, label='No Churn')
        axes[1, 2].scatter(self.df[self.df['Churn'] == 'Yes']['SupportCalls'], 
                          self.df[self.df['Churn'] == 'Yes']['SatisfactionScore'], 
                          alpha=0.5, label='Churn')
        axes[1, 2].set_title('Support Calls vs Satisfaction Score')
        axes[1, 2].set_xlabel('Support Calls')
        axes[1, 2].set_ylabel('Satisfaction Score')
        axes[1, 2].legend()
        
        # 7. Churn by Payment Method
        payment_churn = pd.crosstab(self.df['PaymentMethod'], self.df['Churn'], normalize='index') * 100
        payment_churn.plot(kind='bar', ax=axes[2, 0], rot=45)
        axes[2, 0].set_title('Churn Rate by Payment Method')
        axes[2, 0].set_ylabel('Churn Rate (%)')
        axes[2, 0].legend(title='Churn')
        
        # 8. Senior Citizens Churn
        senior_churn = pd.crosstab(self.df['SeniorCitizen'], self.df['Churn'], normalize='index') * 100
        senior_churn.plot(kind='bar', ax=axes[2, 1], rot=0)
        axes[2, 1].set_title('Churn Rate by Senior Citizen Status')
        axes[2, 1].set_ylabel('Churn Rate (%)')
        axes[2, 1].legend(title='Churn')
        
        # 9. Number of Services vs Churn
        services_churn = self.df.groupby('NumServices')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
        axes[2, 2].bar(services_churn.index, services_churn.values)
        axes[2, 2].set_title('Churn Rate by Number of Services')
        axes[2, 2].set_xlabel('Number of Services')
        axes[2, 2].set_ylabel('Churn Rate (%)')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'churn_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Churn analysis dashboard saved to {PLOTS_DIR / 'churn_analysis_dashboard.png'}")
    
    def correlation_analysis(self):
        """Analyze feature correlations"""
        print("\nGenerating correlation analysis...")
        
        # Prepare numerical data for correlation
        df_numeric = self.df.copy()
        
        # Encode categorical variables for correlation analysis
        categorical_columns = df_numeric.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col == 'customerID':
                continue
            df_numeric[col] = pd.Categorical(df_numeric[col]).codes
        
        # Calculate correlation matrix
        correlation_matrix = df_numeric.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(16, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Top correlations with Churn
        churn_correlations = correlation_matrix['Churn'].abs().sort_values(ascending=False)
        print("\nTop 10 features correlated with Churn:")
        for feature, corr in churn_correlations.head(11).items():  # 11 to exclude Churn itself
            if feature != 'Churn':
                print(f"  {feature}: {corr:.3f}")
        
        return correlation_matrix
    
    def cohort_analysis(self):
        """Perform cohort analysis"""
        print("\nGenerating cohort analysis...")
        
        # Create tenure groups for cohort analysis
        df_cohort = self.df.copy()
        df_cohort['TenureGroup'] = pd.cut(df_cohort['tenure'], 
                                         bins=[0, 6, 12, 24, 36, 72], 
                                         labels=['0-6m', '6-12m', '1-2y', '2-3y', '3y+'])
        
        # Cohort retention analysis
        cohort_data = df_cohort.groupby('TenureGroup').agg({
            'Churn': lambda x: (x == 'No').mean(),
            'customerID': 'count',
            'MonthlyCharges': 'mean',
            'TotalCharges': 'mean',
            'SatisfactionScore': 'mean'
        }).round(3)
        
        cohort_data.columns = ['Retention_Rate', 'Customer_Count', 'Avg_Monthly_Charges', 'Avg_Total_Charges', 'Avg_Satisfaction']
        
        print("\nCohort Analysis by Tenure Groups:")
        print(cohort_data)
        
        # Visualize cohort analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Customer Cohort Analysis', fontsize=16, fontweight='bold')
        
        # Retention rate by cohort
        axes[0, 0].bar(cohort_data.index, cohort_data['Retention_Rate'] * 100)
        axes[0, 0].set_title('Retention Rate by Tenure Group')
        axes[0, 0].set_ylabel('Retention Rate (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Customer count by cohort
        axes[0, 1].bar(cohort_data.index, cohort_data['Customer_Count'])
        axes[0, 1].set_title('Customer Count by Tenure Group')
        axes[0, 1].set_ylabel('Number of Customers')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Average monthly charges by cohort
        axes[1, 0].bar(cohort_data.index, cohort_data['Avg_Monthly_Charges'])
        axes[1, 0].set_title('Average Monthly Charges by Tenure Group')
        axes[1, 0].set_ylabel('Monthly Charges ($)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Average satisfaction by cohort
        axes[1, 1].bar(cohort_data.index, cohort_data['Avg_Satisfaction'])
        axes[1, 1].set_title('Average Satisfaction by Tenure Group')
        axes[1, 1].set_ylabel('Satisfaction Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'cohort_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cohort_data
    
    def interactive_visualizations(self):
        """Create interactive visualizations using Plotly"""
        print("\nGenerating interactive visualizations...")
        
        # 1. Interactive Churn Distribution by Multiple Features
        fig1 = px.sunburst(
            self.df, 
            path=['Contract', 'InternetService', 'Churn'], 
            title='Churn Distribution by Contract and Internet Service'
        )
        fig1.write_html(PLOTS_DIR / 'interactive_churn_sunburst.html')
        
        # 2. 3D Scatter Plot
        fig2 = px.scatter_3d(
            self.df, 
            x='tenure', 
            y='MonthlyCharges', 
            z='TotalCharges',
            color='Churn',
            title='3D Customer Profile: Tenure vs Charges vs Churn',
            labels={'tenure': 'Tenure (months)', 'MonthlyCharges': 'Monthly Charges ($)', 'TotalCharges': 'Total Charges ($)'}
        )
        fig2.write_html(PLOTS_DIR / 'interactive_3d_scatter.html')
        
        # 3. Interactive Correlation Heatmap
        df_numeric = self.df.select_dtypes(include=[np.number])
        correlation_matrix = df_numeric.corr()
        
        fig3 = px.imshow(
            correlation_matrix,
            title='Interactive Correlation Heatmap',
            aspect='auto'
        )
        fig3.write_html(PLOTS_DIR / 'interactive_correlation.html')
        
        # 4. Customer Satisfaction vs Support Calls
        fig4 = px.scatter(
            self.df,
            x='SupportCalls',
            y='SatisfactionScore',
            color='Churn',
            size='MonthlyCharges',
            hover_data=['tenure', 'Contract'],
            title='Customer Satisfaction vs Support Calls (sized by Monthly Charges)'
        )
        fig4.write_html(PLOTS_DIR / 'interactive_satisfaction_support.html')
        
        print("Interactive visualizations saved to plots directory:")
        print("- interactive_churn_sunburst.html")
        print("- interactive_3d_scatter.html") 
        print("- interactive_correlation.html")
        print("- interactive_satisfaction_support.html")
    
    def generate_eda_report(self):
        """Generate comprehensive EDA report"""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE EDA REPORT")
        print("=" * 80)
        
        # Run all analyses
        basic_stats = self.basic_statistics()
        churn_patterns = self.churn_analysis()
        correlation_matrix = self.correlation_analysis()
        cohort_data = self.cohort_analysis()
        
        # Create visualizations
        self.create_churn_visualizations()
        self.interactive_visualizations()
        
        # Generate summary insights
        insights = self.generate_insights()
        
        # Save report
        self.save_eda_report(basic_stats, churn_patterns, correlation_matrix, cohort_data, insights)
        
        print("\n" + "=" * 80)
        print("EDA REPORT GENERATION COMPLETED")
        print("=" * 80)
        
        return {
            'basic_stats': basic_stats,
            'churn_patterns': churn_patterns,
            'correlation_matrix': correlation_matrix,
            'cohort_data': cohort_data,
            'insights': insights
        }
    
    def generate_insights(self):
        """Generate key insights from the analysis"""
        insights = []
        
        # Churn rate insights
        overall_churn_rate = (self.df['Churn'] == 'Yes').mean()
        insights.append(f"Overall churn rate is {overall_churn_rate:.2%}")
        
        # Contract insights
        contract_churn = self.df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean())
        highest_churn_contract = contract_churn.idxmax()
        insights.append(f"Highest churn rate is among {highest_churn_contract} customers ({contract_churn[highest_churn_contract]:.2%})")
        
        # Tenure insights
        avg_tenure_churned = self.df[self.df['Churn'] == 'Yes']['tenure'].mean()
        avg_tenure_retained = self.df[self.df['Churn'] == 'No']['tenure'].mean()
        insights.append(f"Churned customers have lower average tenure ({avg_tenure_churned:.1f} months vs {avg_tenure_retained:.1f} months)")
        
        # Charges insights
        avg_monthly_churned = self.df[self.df['Churn'] == 'Yes']['MonthlyCharges'].mean()
        avg_monthly_retained = self.df[self.df['Churn'] == 'No']['MonthlyCharges'].mean()
        insights.append(f"Churned customers pay higher monthly charges on average (${avg_monthly_churned:.2f} vs ${avg_monthly_retained:.2f})")
        
        # Support calls insights
        avg_support_churned = self.df[self.df['Churn'] == 'Yes']['SupportCalls'].mean()
        avg_support_retained = self.df[self.df['Churn'] == 'No']['SupportCalls'].mean()
        insights.append(f"Churned customers make more support calls ({avg_support_churned:.1f} vs {avg_support_retained:.1f})")
        
        # Satisfaction insights
        avg_satisfaction_churned = self.df[self.df['Churn'] == 'Yes']['SatisfactionScore'].mean()
        avg_satisfaction_retained = self.df[self.df['Churn'] == 'No']['SatisfactionScore'].mean()
        insights.append(f"Churned customers have lower satisfaction scores ({avg_satisfaction_churned:.1f} vs {avg_satisfaction_retained:.1f})")
        
        return insights
    
    def save_eda_report(self, basic_stats, churn_patterns, correlation_matrix, cohort_data, insights):
        """Save EDA report to file"""
        report_path = PLOTS_DIR / 'eda_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("CUSTOMER CHURN PREDICTION - EDA REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("KEY INSIGHTS:\n")
            f.write("-" * 20 + "\n")
            for insight in insights:
                f.write(f"• {insight}\n")
            
            f.write(f"\n\nBASIC STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Customers: {len(self.df):,}\n")
            f.write(f"Churn Rate: {(self.df['Churn'] == 'Yes').mean():.2%}\n")
            f.write(f"Average Tenure: {self.df['tenure'].mean():.1f} months\n")
            f.write(f"Average Monthly Charges: ${self.df['MonthlyCharges'].mean():.2f}\n")
            
            f.write(f"\n\nFILES GENERATED:\n")
            f.write("-" * 20 + "\n")
            f.write("• churn_analysis_dashboard.png\n")
            f.write("• correlation_heatmap.png\n")
            f.write("• cohort_analysis.png\n")
            f.write("• interactive_churn_sunburst.html\n")
            f.write("• interactive_3d_scatter.html\n")
            f.write("• interactive_correlation.html\n")
            f.write("• interactive_satisfaction_support.html\n")
        
        print(f"EDA report saved to {report_path}")

def main():
    """Run complete EDA analysis"""
    eda = ChurnEDA()
    report = eda.generate_eda_report()
    
    print("\nEDA Analysis completed successfully!")
    print(f"Check the '{PLOTS_DIR}' directory for all visualizations and reports.")

if __name__ == "__main__":
    main()



