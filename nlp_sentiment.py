"""
NLP Sentiment Analysis for Customer Feedback
"""
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from config import *

class SentimentAnalyzer:
    """NLP sentiment analysis for customer feedback"""
    
    def __init__(self):
        self.sentiment_pipeline = None
        self.feedback_df = None
        
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
        except Exception as e:
            print(f"NLTK download warning: {e}")
    
    def load_sentiment_model(self):
        """Load pre-trained sentiment analysis model"""
        print("Loading sentiment analysis model...")
        try:
            # Use a lightweight sentiment model
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model=model_name,
                tokenizer=model_name,
                max_length=MAX_TEXT_LENGTH,
                truncation=True
            )
            print("Sentiment model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using TextBlob as fallback...")
            self.sentiment_pipeline = None
    
    def load_feedback_data(self, data_path=CUSTOMER_FEEDBACK_PATH):
        """Load customer feedback data"""
        print(f"Loading feedback data from {data_path}")
        self.feedback_df = pd.read_csv(data_path)
        print(f"Loaded {len(self.feedback_df)} feedback records")
        return self.feedback_df
    
    def analyze_sentiment_transformers(self, text):
        """Analyze sentiment using transformers model"""
        if self.sentiment_pipeline is None:
            return self.analyze_sentiment_textblob(text)
        
        try:
            result = self.sentiment_pipeline(text)[0]
            
            # Map labels to standard format
            label_mapping = {
                'LABEL_0': 'negative',  # Negative
                'LABEL_1': 'neutral',   # Neutral
                'LABEL_2': 'positive',  # Positive
                'NEGATIVE': 'negative',
                'NEUTRAL': 'neutral',
                'POSITIVE': 'positive'
            }
            
            sentiment = label_mapping.get(result['label'], result['label'].lower())
            confidence = result['score']
            
            return sentiment, confidence
        except Exception as e:
            print(f"Error in transformers analysis: {e}")
            return self.analyze_sentiment_textblob(text)
    
    def analyze_sentiment_textblob(self, text):
        """Analyze sentiment using TextBlob as fallback"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            confidence = abs(polarity)
            return sentiment, confidence
        except:
            return 'neutral', 0.0
    
    def process_feedback_sentiment(self):
        """Process all feedback for sentiment analysis"""
        print("Processing feedback sentiment...")
        
        if self.feedback_df is None:
            self.load_feedback_data()
        
        sentiments = []
        confidences = []
        
        for text in self.feedback_df['feedback_text']:
            sentiment, confidence = self.analyze_sentiment_transformers(text)
            sentiments.append(sentiment)
            confidences.append(confidence)
        
        self.feedback_df['predicted_sentiment'] = sentiments
        self.feedback_df['sentiment_confidence'] = confidences
        
        print("Sentiment analysis completed")
        return self.feedback_df
    
    def calculate_sentiment_scores(self):
        """Calculate numerical sentiment scores"""
        print("Calculating sentiment scores...")
        
        # Create sentiment score (-1 to 1)
        sentiment_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
        self.feedback_df['sentiment_score'] = self.feedback_df['predicted_sentiment'].map(sentiment_mapping)
        
        # Weight by confidence
        self.feedback_df['weighted_sentiment_score'] = (
            self.feedback_df['sentiment_score'] * self.feedback_df['sentiment_confidence']
        )
        
        return self.feedback_df
    
    def aggregate_customer_sentiment(self):
        """Aggregate sentiment scores by customer"""
        print("Aggregating sentiment by customer...")
        
        customer_sentiment = self.feedback_df.groupby('customerID').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'weighted_sentiment_score': 'mean',
            'sentiment_confidence': 'mean'
        }).round(3)
        
        # Flatten column names
        customer_sentiment.columns = [
            'avg_sentiment', 'sentiment_std', 'feedback_count',
            'weighted_avg_sentiment', 'avg_confidence'
        ]
        
        # Fill missing std with 0
        customer_sentiment['sentiment_std'] = customer_sentiment['sentiment_std'].fillna(0)
        
        # Reset index
        customer_sentiment = customer_sentiment.reset_index()
        
        print(f"Aggregated sentiment for {len(customer_sentiment)} customers")
        return customer_sentiment
    
    def create_sentiment_visualizations(self):
        """Create sentiment analysis visualizations"""
        print("Creating sentiment visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Customer Feedback Sentiment Analysis', fontsize=16, fontweight='bold')
        
        # 1. Sentiment Distribution
        sentiment_counts = self.feedback_df['predicted_sentiment'].value_counts()
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Overall Sentiment Distribution')
        
        # 2. Sentiment vs True Sentiment (if available)
        if 'true_sentiment' in self.feedback_df.columns:
            confusion_data = pd.crosstab(
                self.feedback_df['true_sentiment'], 
                self.feedback_df['predicted_sentiment']
            )
            sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
            axes[0, 1].set_title('Predicted vs True Sentiment')
            axes[0, 1].set_xlabel('Predicted Sentiment')
            axes[0, 1].set_ylabel('True Sentiment')
        else:
            axes[0, 1].text(0.5, 0.5, 'True sentiment\nnot available', 
                          ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Predicted vs True Sentiment')
        
        # 3. Confidence Distribution
        axes[0, 2].hist(self.feedback_df['sentiment_confidence'], bins=20, alpha=0.7)
        axes[0, 2].set_title('Sentiment Confidence Distribution')
        axes[0, 2].set_xlabel('Confidence Score')
        axes[0, 2].set_ylabel('Count')
        
        # 4. Sentiment Score Distribution
        axes[1, 0].hist(self.feedback_df['sentiment_score'], bins=[-1.5, -0.5, 0.5, 1.5], 
                       alpha=0.7, color=['red', 'gray', 'green'])
        axes[1, 0].set_title('Sentiment Score Distribution')
        axes[1, 0].set_xlabel('Sentiment Score')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xticks([-1, 0, 1])
        axes[1, 0].set_xticklabels(['Negative', 'Neutral', 'Positive'])
        
        # 5. Sentiment Over Time
        if 'timestamp' in self.feedback_df.columns:
            self.feedback_df['timestamp'] = pd.to_datetime(self.feedback_df['timestamp'])
            daily_sentiment = self.feedback_df.groupby(
                self.feedback_df['timestamp'].dt.date
            )['sentiment_score'].mean()
            
            axes[1, 1].plot(daily_sentiment.index, daily_sentiment.values)
            axes[1, 1].set_title('Average Sentiment Over Time')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Average Sentiment Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'Timestamp data\nnot available', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Sentiment Over Time')
        
        # 6. Top Words by Sentiment
        positive_text = ' '.join(
            self.feedback_df[self.feedback_df['predicted_sentiment'] == 'positive']['feedback_text']
        )
        negative_text = ' '.join(
            self.feedback_df[self.feedback_df['predicted_sentiment'] == 'negative']['feedback_text']
        )
        
        if positive_text and negative_text:
            # Create word frequency comparison
            positive_words = positive_text.lower().split()
            negative_words = negative_text.lower().split()
            
            positive_freq = Counter(positive_words).most_common(10)
            negative_freq = Counter(negative_words).most_common(10)
            
            pos_words, pos_counts = zip(*positive_freq) if positive_freq else ([], [])
            neg_words, neg_counts = zip(*negative_freq) if negative_freq else ([], [])
            
            x_pos = np.arange(len(pos_words))
            x_neg = np.arange(len(neg_words))
            
            if pos_words:
                axes[1, 2].barh(x_pos, pos_counts, alpha=0.7, color='green', label='Positive')
            if neg_words:
                axes[1, 2].barh(x_neg, neg_counts, alpha=0.7, color='red', label='Negative')
            
            axes[1, 2].set_title('Top Words by Sentiment')
            axes[1, 2].set_xlabel('Frequency')
            axes[1, 2].legend()
        else:
            axes[1, 2].text(0.5, 0.5, 'Insufficient data\nfor word analysis', 
                          ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Top Words by Sentiment')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Sentiment analysis plot saved to {PLOTS_DIR / 'sentiment_analysis.png'}")
    
    def create_word_clouds(self):
        """Create word clouds for different sentiments"""
        print("Creating word clouds...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Word Clouds by Sentiment', fontsize=16, fontweight='bold')
        
        sentiments = ['positive', 'neutral', 'negative']
        colors = ['green', 'gray', 'red']
        
        for i, (sentiment, color) in enumerate(zip(sentiments, colors)):
            text_data = ' '.join(
                self.feedback_df[self.feedback_df['predicted_sentiment'] == sentiment]['feedback_text']
            )
            
            if text_data.strip():
                wordcloud = WordCloud(
                    width=400, height=300, 
                    background_color='white',
                    colormap='viridis' if sentiment == 'positive' else 'plasma' if sentiment == 'negative' else 'gray'
                ).generate(text_data)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'{sentiment.title()} Feedback')
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'No {sentiment}\nfeedback available', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{sentiment.title()} Feedback')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'sentiment_wordclouds.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Word clouds saved to {PLOTS_DIR / 'sentiment_wordclouds.png'}")
    
    def evaluate_sentiment_accuracy(self):
        """Evaluate sentiment prediction accuracy if true labels available"""
        if 'true_sentiment' not in self.feedback_df.columns:
            print("True sentiment labels not available for evaluation")
            return None
        
        from sklearn.metrics import classification_report, accuracy_score
        
        y_true = self.feedback_df['true_sentiment']
        y_pred = self.feedback_df['predicted_sentiment']
        
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        
        print(f"\nSentiment Analysis Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(report)
        
        return accuracy, report
    
    def save_sentiment_results(self):
        """Save sentiment analysis results"""
        print("Saving sentiment analysis results...")
        
        # Save processed feedback
        feedback_output_path = DATA_DIR / 'processed_feedback_sentiment.csv'
        self.feedback_df.to_csv(feedback_output_path, index=False)
        
        # Save customer-level sentiment aggregation
        customer_sentiment = self.aggregate_customer_sentiment()
        customer_output_path = DATA_DIR / 'customer_sentiment_scores.csv'
        customer_sentiment.to_csv(customer_output_path, index=False)
        
        print(f"Sentiment results saved:")
        print(f"- Feedback with sentiment: {feedback_output_path}")
        print(f"- Customer sentiment scores: {customer_output_path}")
        
        return customer_sentiment
    
    def run_complete_sentiment_analysis(self):
        """Run complete sentiment analysis pipeline"""
        print("=" * 60)
        print("STARTING NLP SENTIMENT ANALYSIS")
        print("=" * 60)
        
        # Setup
        self.setup_nltk()
        self.load_sentiment_model()
        
        # Load and process data
        self.load_feedback_data()
        self.process_feedback_sentiment()
        self.calculate_sentiment_scores()
        
        # Create visualizations
        self.create_sentiment_visualizations()
        self.create_word_clouds()
        
        # Evaluate if possible
        self.evaluate_sentiment_accuracy()
        
        # Save results
        customer_sentiment = self.save_sentiment_results()
        
        # Generate summary
        self.generate_sentiment_summary()
        
        print("=" * 60)
        print("NLP SENTIMENT ANALYSIS COMPLETED")
        print("=" * 60)
        
        return customer_sentiment
    
    def generate_sentiment_summary(self):
        """Generate sentiment analysis summary"""
        print("\n" + "=" * 50)
        print("SENTIMENT ANALYSIS SUMMARY")
        print("=" * 50)
        
        # Overall statistics
        total_feedback = len(self.feedback_df)
        sentiment_dist = self.feedback_df['predicted_sentiment'].value_counts(normalize=True)
        avg_confidence = self.feedback_df['sentiment_confidence'].mean()
        
        print(f"Total Feedback Analyzed: {total_feedback:,}")
        print(f"Average Confidence Score: {avg_confidence:.3f}")
        print("\nSentiment Distribution:")
        for sentiment, percentage in sentiment_dist.items():
            print(f"  {sentiment.title()}: {percentage:.1%}")
        
        # Customer-level insights
        customer_sentiment = self.aggregate_customer_sentiment()
        print(f"\nCustomer-Level Insights:")
        print(f"  Customers with feedback: {len(customer_sentiment):,}")
        print(f"  Average sentiment score: {customer_sentiment['avg_sentiment'].mean():.3f}")
        print(f"  Most positive customer: {customer_sentiment['avg_sentiment'].max():.3f}")
        print(f"  Most negative customer: {customer_sentiment['avg_sentiment'].min():.3f}")

def main():
    """Main function to run sentiment analysis"""
    analyzer = SentimentAnalyzer()
    customer_sentiment = analyzer.run_complete_sentiment_analysis()
    
    print("\nSentiment Analysis completed successfully!")
    print(f"Results saved in: {DATA_DIR}")
    print(f"Visualizations saved in: {PLOTS_DIR}")

if __name__ == "__main__":
    main()



