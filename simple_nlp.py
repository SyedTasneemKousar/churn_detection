"""
Simplified NLP Sentiment Analysis
"""
import pandas as pd
import numpy as np
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

from config import *

def simple_sentiment_analysis():
    """Run simplified sentiment analysis using TextBlob"""
    print("Running simplified sentiment analysis...")
    
    # Load feedback data
    try:
        df_feedback = pd.read_csv(CUSTOMER_FEEDBACK_PATH)
        print(f"Loaded {len(df_feedback)} feedback records")
    except FileNotFoundError:
        print("Feedback data not found")
        return
    
    # Analyze sentiment using TextBlob
    sentiments = []
    scores = []
    
    for text in df_feedback['feedback_text']:
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            sentiments.append(sentiment)
            scores.append(polarity)
        except:
            sentiments.append('neutral')
            scores.append(0.0)
    
    # Add results to dataframe
    df_feedback['predicted_sentiment'] = sentiments
    df_feedback['sentiment_score'] = scores
    
    # Aggregate by customer
    customer_sentiment = df_feedback.groupby('customerID').agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'predicted_sentiment': lambda x: x.mode().iloc[0] if len(x) > 0 else 'neutral'
    }).round(3)
    
    # Flatten column names
    customer_sentiment.columns = ['avg_sentiment', 'sentiment_std', 'feedback_count', 'dominant_sentiment']
    customer_sentiment['sentiment_std'] = customer_sentiment['sentiment_std'].fillna(0)
    customer_sentiment['avg_confidence'] = np.abs(customer_sentiment['avg_sentiment'])
    customer_sentiment = customer_sentiment.reset_index()
    
    # Save results
    df_feedback.to_csv(DATA_DIR / 'processed_feedback_sentiment.csv', index=False)
    customer_sentiment.to_csv(DATA_DIR / 'customer_sentiment_scores.csv', index=False)
    
    print(f"Sentiment analysis completed!")
    print(f"- Processed feedback saved")
    print(f"- Customer sentiment scores saved")
    print(f"- Sentiment distribution:")
    print(df_feedback['predicted_sentiment'].value_counts())
    
    return customer_sentiment

if __name__ == "__main__":
    simple_sentiment_analysis()




