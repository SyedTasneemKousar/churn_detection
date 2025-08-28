"""
Customer Clustering and Recommendation System
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from config import *

class CustomerClusteringSystem:
    """Customer clustering and recommendation system"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.dbscan_model = None
        self.pca_model = None
        self.cluster_profiles = {}
        self.recommendations = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data for clustering"""
        print("Loading data for clustering...")
        
        # Load raw data
        df = pd.read_csv(RAW_DATA_PATH)
        
        # Load sentiment data if available
        try:
            df_sentiment = pd.read_csv(DATA_DIR / 'customer_sentiment_scores.csv')
            df = df.merge(df_sentiment[['customerID', 'avg_sentiment', 'feedback_count']], 
                         on='customerID', how='left')
            df['avg_sentiment'] = df['avg_sentiment'].fillna(0)
            df['feedback_count'] = df['feedback_count'].fillna(0)
            print("Merged with sentiment data")
        except FileNotFoundError:
            print("Sentiment data not available, using structured data only")
            df['avg_sentiment'] = 0
            df['feedback_count'] = 0
        
        print(f"Dataset shape: {df.shape}")
        return df
    
    def prepare_clustering_features(self, df):
        """Prepare features for clustering"""
        print("Preparing clustering features...")
        
        # Convert categorical variables to numerical
        df_cluster = df.copy()
        
        # Encode categorical variables
        categorical_mappings = {
            'gender': {'Male': 1, 'Female': 0},
            'Partner': {'Yes': 1, 'No': 0},
            'Dependents': {'Yes': 1, 'No': 0},
            'PhoneService': {'Yes': 1, 'No': 0},
            'PaperlessBilling': {'Yes': 1, 'No': 0},
            'Churn': {'Yes': 1, 'No': 0}
        }
        
        for col, mapping in categorical_mappings.items():
            if col in df_cluster.columns:
                df_cluster[col] = df_cluster[col].map(mapping)
        
        # Encode contract type
        contract_dummies = pd.get_dummies(df_cluster['Contract'], prefix='Contract')
        df_cluster = pd.concat([df_cluster, contract_dummies], axis=1)
        
        # Encode internet service
        internet_dummies = pd.get_dummies(df_cluster['InternetService'], prefix='Internet')
        df_cluster = pd.concat([df_cluster, internet_dummies], axis=1)
        
        # Encode payment method
        payment_dummies = pd.get_dummies(df_cluster['PaymentMethod'], prefix='Payment')
        df_cluster = pd.concat([df_cluster, payment_dummies], axis=1)
        
        # Select clustering features
        clustering_features = [
            'tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyUsage',
            'SupportCalls', 'SatisfactionScore', 'NumServices', 'avg_sentiment',
            'feedback_count', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'PaperlessBilling'
        ] + [col for col in df_cluster.columns if col.startswith(('Contract_', 'Internet_', 'Payment_'))]
        
        # Keep only available features
        available_features = [col for col in clustering_features if col in df_cluster.columns]
        
        X_cluster = df_cluster[available_features].fillna(0)
        
        print(f"Clustering features: {len(available_features)}")
        print(f"Feature names: {available_features[:10]}...")
        
        return X_cluster, available_features, df_cluster
    
    def find_optimal_clusters(self, X):
        """Find optimal number of clusters using elbow method and silhouette score"""
        print("Finding optimal number of clusters...")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Test different numbers of clusters
        k_range = range(2, 11)
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
        
        # Plot elbow curve and silhouette scores
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow curve
        axes[0].plot(k_range, inertias, 'bo-')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method for Optimal k')
        axes[0].grid(True)
        
        # Silhouette scores
        axes[1].plot(k_range, silhouette_scores, 'ro-')
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Score vs Number of Clusters')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'cluster_optimization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find optimal k (highest silhouette score)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        best_silhouette = max(silhouette_scores)
        
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Best silhouette score: {best_silhouette:.3f}")
        
        return optimal_k, X_scaled
    
    def perform_kmeans_clustering(self, X_scaled, n_clusters):
        """Perform K-Means clustering"""
        print(f"Performing K-Means clustering with {n_clusters} clusters...")
        
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(X_scaled)
        
        print(f"Clustering completed. Silhouette score: {silhouette_score(X_scaled, cluster_labels):.3f}")
        
        return cluster_labels
    
    def perform_dbscan_clustering(self, X_scaled):
        """Perform DBSCAN clustering"""
        print("Performing DBSCAN clustering...")
        
        # Try different eps values
        eps_values = [0.3, 0.5, 0.7, 1.0]
        best_eps = 0.5
        best_score = -1
        
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=5)
            labels = dbscan.fit_predict(X_scaled)
            
            if len(set(labels)) > 1:  # More than just noise
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_eps = eps
        
        self.dbscan_model = DBSCAN(eps=best_eps, min_samples=5)
        dbscan_labels = self.dbscan_model.fit_predict(X_scaled)
        
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)
        
        print(f"DBSCAN completed with eps={best_eps}")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        
        return dbscan_labels
    
    def analyze_cluster_profiles(self, df, cluster_labels, method='kmeans'):
        """Analyze cluster profiles and characteristics"""
        print(f"Analyzing {method} cluster profiles...")
        
        df_analysis = df.copy()
        df_analysis[f'{method}_cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_profiles = {}
        
        # Numerical features for profiling
        numerical_features = [
            'tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyUsage',
            'SupportCalls', 'SatisfactionScore', 'NumServices', 'avg_sentiment'
        ]
        
        # Categorical features for profiling
        categorical_features = ['Contract', 'InternetService', 'PaymentMethod', 'Churn']
        
        unique_clusters = sorted([c for c in df_analysis[f'{method}_cluster'].unique() if c != -1])
        
        for cluster_id in unique_clusters:
            cluster_data = df_analysis[df_analysis[f'{method}_cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df_analysis) * 100,
                'churn_rate': (cluster_data['Churn'] == 'Yes').mean() if 'Churn' in cluster_data.columns else 0,
                'numerical_stats': {},
                'categorical_stats': {}
            }
            
            # Numerical feature statistics
            for feature in numerical_features:
                if feature in cluster_data.columns:
                    profile['numerical_stats'][feature] = {
                        'mean': cluster_data[feature].mean(),
                        'median': cluster_data[feature].median(),
                        'std': cluster_data[feature].std()
                    }
            
            # Categorical feature statistics
            for feature in categorical_features:
                if feature in cluster_data.columns:
                    profile['categorical_stats'][feature] = cluster_data[feature].value_counts(normalize=True).to_dict()
            
            cluster_profiles[f'Cluster_{cluster_id}'] = profile
        
        self.cluster_profiles[method] = cluster_profiles
        
        # Print cluster summary
        print(f"\n{method.upper()} Cluster Summary:")
        print("-" * 50)
        for cluster_name, profile in cluster_profiles.items():
            print(f"{cluster_name}:")
            print(f"  Size: {profile['size']} customers ({profile['percentage']:.1f}%)")
            print(f"  Churn Rate: {profile['churn_rate']:.1%}")
            print(f"  Avg Monthly Charges: ${profile['numerical_stats'].get('MonthlyCharges', {}).get('mean', 0):.2f}")
            print(f"  Avg Tenure: {profile['numerical_stats'].get('tenure', {}).get('mean', 0):.1f} months")
            print()
        
        return cluster_profiles
    
    def visualize_clusters(self, X_scaled, cluster_labels, feature_names, method='kmeans'):
        """Create cluster visualizations"""
        print(f"Creating {method} cluster visualizations...")
        
        # PCA for visualization
        self.pca_model = PCA(n_components=2, random_state=RANDOM_STATE)
        X_pca = self.pca_model.fit_transform(X_scaled)
        
        # Create visualization dataframe
        viz_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': cluster_labels
        })
        
        # Remove noise points for cleaner visualization
        if method == 'dbscan':
            viz_df = viz_df[viz_df['Cluster'] != -1]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{method.upper()} Clustering Analysis', fontsize=16, fontweight='bold')
        
        # 1. PCA scatter plot
        unique_clusters = sorted(viz_df['Cluster'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster in enumerate(unique_clusters):
            cluster_data = viz_df[viz_df['Cluster'] == cluster]
            axes[0, 0].scatter(cluster_data['PC1'], cluster_data['PC2'], 
                              c=[colors[i]], label=f'Cluster {cluster}', alpha=0.6)
        
        axes[0, 0].set_xlabel(f'PC1 ({self.pca_model.explained_variance_ratio_[0]:.1%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({self.pca_model.explained_variance_ratio_[1]:.1%} variance)')
        axes[0, 0].set_title('Clusters in PCA Space')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Cluster sizes
        cluster_sizes = viz_df['Cluster'].value_counts().sort_index()
        axes[0, 1].bar(range(len(cluster_sizes)), cluster_sizes.values, color=colors[:len(cluster_sizes)])
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].set_ylabel('Number of Customers')
        axes[0, 1].set_title('Cluster Sizes')
        axes[0, 1].set_xticks(range(len(cluster_sizes)))
        axes[0, 1].set_xticklabels([f'C{i}' for i in cluster_sizes.index])
        
        # 3. Feature importance in clustering (if available)
        if method == 'kmeans' and hasattr(self.pca_model, 'components_'):
            # Show top features contributing to PC1 and PC2
            pc1_importance = np.abs(self.pca_model.components_[0])
            top_features_idx = np.argsort(pc1_importance)[-10:]
            top_features = [feature_names[i] for i in top_features_idx]
            top_importance = pc1_importance[top_features_idx]
            
            axes[1, 0].barh(range(len(top_features)), top_importance)
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels(top_features, fontsize=8)
            axes[1, 0].set_title('Top Features Contributing to PC1')
            axes[1, 0].set_xlabel('Absolute Contribution')
        
        # 4. Churn rate by cluster (if available)
        if hasattr(self, 'cluster_profiles') and method in self.cluster_profiles:
            cluster_names = list(self.cluster_profiles[method].keys())
            churn_rates = [self.cluster_profiles[method][name]['churn_rate'] for name in cluster_names]
            
            axes[1, 1].bar(range(len(cluster_names)), churn_rates, color=colors[:len(cluster_names)])
            axes[1, 1].set_xlabel('Cluster')
            axes[1, 1].set_ylabel('Churn Rate')
            axes[1, 1].set_title('Churn Rate by Cluster')
            axes[1, 1].set_xticks(range(len(cluster_names)))
            axes[1, 1].set_xticklabels([name.split('_')[1] for name in cluster_names])
            axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'{method}_clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"{method} clustering visualization saved to {PLOTS_DIR / f'{method}_clustering_analysis.png'}")
    
    def generate_recommendations(self, method='kmeans'):
        """Generate recommendations for each cluster"""
        print(f"Generating recommendations for {method} clusters...")
        
        if method not in self.cluster_profiles:
            print("Cluster profiles not available. Run clustering analysis first.")
            return
        
        recommendations = {}
        
        for cluster_name, profile in self.cluster_profiles[method].items():
            cluster_recommendations = []
            
            # High churn rate clusters
            if profile['churn_rate'] > 0.4:
                cluster_recommendations.append("🚨 HIGH CHURN RISK - Implement immediate retention campaigns")
                cluster_recommendations.append("📞 Proactive customer outreach and satisfaction surveys")
            
            # Low tenure customers
            avg_tenure = profile['numerical_stats'].get('tenure', {}).get('mean', 0)
            if avg_tenure < 12:
                cluster_recommendations.append("🆕 Focus on new customer onboarding and engagement")
                cluster_recommendations.append("🎁 Welcome bonuses and early engagement incentives")
            
            # High monthly charges
            avg_charges = profile['numerical_stats'].get('MonthlyCharges', {}).get('mean', 0)
            if avg_charges > 75:
                cluster_recommendations.append("💰 Premium customer segment - loyalty programs and exclusive offers")
                cluster_recommendations.append("⭐ VIP customer service and priority support")
            elif avg_charges < 40:
                cluster_recommendations.append("📈 Upselling opportunities for additional services")
                cluster_recommendations.append("📦 Bundle packages to increase value")
            
            # High support calls
            avg_support = profile['numerical_stats'].get('SupportCalls', {}).get('mean', 0)
            if avg_support > 3:
                cluster_recommendations.append("🔧 Focus on service quality improvement")
                cluster_recommendations.append("📚 Self-service portal and FAQ resources")
            
            # Low satisfaction
            avg_satisfaction = profile['numerical_stats'].get('SatisfactionScore', {}).get('mean', 0)
            if avg_satisfaction < 6:
                cluster_recommendations.append("😔 Address satisfaction issues through surveys and feedback")
                cluster_recommendations.append("🎯 Targeted service improvement initiatives")
            
            # Contract type analysis
            contract_stats = profile['categorical_stats'].get('Contract', {})
            if contract_stats.get('Month-to-month', 0) > 0.6:
                cluster_recommendations.append("📝 Encourage longer-term contract commitments with incentives")
            
            # Payment method analysis
            payment_stats = profile['categorical_stats'].get('PaymentMethod', {})
            if payment_stats.get('Electronic check', 0) > 0.4:
                cluster_recommendations.append("💳 Promote automatic payment methods")
            
            # Sentiment analysis (if available)
            avg_sentiment = profile['numerical_stats'].get('avg_sentiment', {}).get('mean', 0)
            if avg_sentiment < -0.2:
                cluster_recommendations.append("😟 Negative sentiment detected - immediate intervention needed")
            elif avg_sentiment > 0.2:
                cluster_recommendations.append("😊 Positive sentiment - leverage for testimonials and referrals")
            
            recommendations[cluster_name] = cluster_recommendations
        
        self.recommendations[method] = recommendations
        
        # Print recommendations
        print(f"\n{method.upper()} Cluster Recommendations:")
        print("=" * 60)
        for cluster_name, recs in recommendations.items():
            print(f"\n{cluster_name} ({self.cluster_profiles[method][cluster_name]['size']} customers):")
            for rec in recs:
                print(f"  {rec}")
        
        return recommendations
    
    def save_clustering_results(self, df, kmeans_labels, dbscan_labels):
        """Save clustering results"""
        print("Saving clustering results...")
        
        # Add cluster labels to dataframe
        results_df = df.copy()
        results_df['kmeans_cluster'] = kmeans_labels
        results_df['dbscan_cluster'] = dbscan_labels
        
        # Save to CSV
        results_df.to_csv(DATA_DIR / 'customer_clusters.csv', index=False)
        
        # Save cluster profiles and recommendations
        import json
        
        clustering_summary = {
            'cluster_profiles': self.cluster_profiles,
            'recommendations': self.recommendations,
            'model_info': {
                'kmeans_n_clusters': self.kmeans_model.n_clusters if self.kmeans_model else None,
                'pca_explained_variance': self.pca_model.explained_variance_ratio_.tolist() if self.pca_model else None
            }
        }
        
        with open(DATA_DIR / 'clustering_summary.json', 'w') as f:
            json.dump(clustering_summary, f, indent=2, default=str)
        
        print(f"Clustering results saved:")
        print(f"- Customer clusters: {DATA_DIR / 'customer_clusters.csv'}")
        print(f"- Clustering summary: {DATA_DIR / 'clustering_summary.json'}")
    
    def run_complete_clustering_analysis(self):
        """Run complete clustering analysis"""
        print("=" * 60)
        print("CUSTOMER CLUSTERING AND RECOMMENDATION SYSTEM")
        print("=" * 60)
        
        # Load and prepare data
        df = self.load_and_prepare_data()
        X_cluster, feature_names, df_processed = self.prepare_clustering_features(df)
        
        # Find optimal clusters
        optimal_k, X_scaled = self.find_optimal_clusters(X_cluster)
        
        # Perform K-Means clustering
        kmeans_labels = self.perform_kmeans_clustering(X_scaled, optimal_k)
        
        # Perform DBSCAN clustering
        dbscan_labels = self.perform_dbscan_clustering(X_scaled)
        
        # Analyze cluster profiles
        self.analyze_cluster_profiles(df_processed, kmeans_labels, 'kmeans')
        self.analyze_cluster_profiles(df_processed, dbscan_labels, 'dbscan')
        
        # Create visualizations
        self.visualize_clusters(X_scaled, kmeans_labels, feature_names, 'kmeans')
        self.visualize_clusters(X_scaled, dbscan_labels, feature_names, 'dbscan')
        
        # Generate recommendations
        self.generate_recommendations('kmeans')
        self.generate_recommendations('dbscan')
        
        # Save results
        self.save_clustering_results(df_processed, kmeans_labels, dbscan_labels)
        
        print("=" * 60)
        print("CLUSTERING ANALYSIS COMPLETED")
        print("=" * 60)
        
        return {
            'kmeans_labels': kmeans_labels,
            'dbscan_labels': dbscan_labels,
            'cluster_profiles': self.cluster_profiles,
            'recommendations': self.recommendations
        }

def main():
    """Main function to run clustering analysis"""
    clustering_system = CustomerClusteringSystem()
    results = clustering_system.run_complete_clustering_analysis()
    
    print("\nCustomer Clustering Analysis completed successfully!")
    print(f"Results saved in: {DATA_DIR}")
    print(f"Visualizations saved in: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
