"""
Lightweight sentiment module safe for Streamlit Cloud.
Uses transformers/TextBlob only if available; otherwise falls back to neutral.
"""

import pandas as pd
import numpy as np
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

from config import *

# Optional heavy dependencies
try:
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover
    pipeline = None

try:
    from textblob import TextBlob  # type: ignore
except Exception:  # pragma: no cover
    TextBlob = None

try:
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    plt = None
    sns = None


class SentimentAnalyzer:
    """NLP sentiment analysis for customer feedback with safe fallbacks."""

    def __init__(self) -> None:
        self.sentiment_pipeline = None
        self.feedback_df: pd.DataFrame | None = None

    def load_sentiment_model(self) -> None:
        """Load transformers model if available; else fallback to TextBlob/neutral."""
        try:
            if pipeline is None:
                raise RuntimeError("transformers not available")
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                truncation=True,
                max_length=512,
            )
            print("Sentiment model loaded")
        except Exception as e:
            print(f"Sentiment model unavailable: {e}")
            self.sentiment_pipeline = None

    def load_feedback_data(self, data_path=CUSTOMER_FEEDBACK_PATH) -> pd.DataFrame:
        self.feedback_df = pd.read_csv(data_path)
        return self.feedback_df

    def analyze_sentiment_textblob(self, text: str) -> tuple[str, float]:
        if TextBlob is None:
            return "neutral", 0.0
        blob = TextBlob(text)
        polarity = float(blob.sentiment.polarity)
        if polarity > 0.1:
            label = "positive"
        elif polarity < -0.1:
            label = "negative"
        else:
            label = "neutral"
        return label, abs(polarity)

    def analyze_sentiment_transformers(self, text: str) -> tuple[str, float]:
        if self.sentiment_pipeline is None:
            return self.analyze_sentiment_textblob(text)
        try:
            result = self.sentiment_pipeline(text)[0]
            mapping = {
                "LABEL_0": "negative",
                "LABEL_1": "neutral",
                "LABEL_2": "positive",
                "NEGATIVE": "negative",
                "NEUTRAL": "neutral",
                "POSITIVE": "positive",
            }
            label = mapping.get(str(result["label"]).upper(), str(result["label"]).lower())
            score = float(result["score"])
            return label, score
        except Exception:
            return self.analyze_sentiment_textblob(text)

    def process_feedback_sentiment(self) -> pd.DataFrame:
        assert self.feedback_df is not None, "Call load_feedback_data() first"
        sentiments: list[str] = []
        confidences: list[float] = []
        for txt in self.feedback_df["feedback_text"]:
            s, c = self.analyze_sentiment_transformers(str(txt))
            sentiments.append(s)
            confidences.append(c)
        self.feedback_df["predicted_sentiment"] = sentiments
        self.feedback_df["sentiment_confidence"] = confidences
        return self.feedback_df

    def calculate_sentiment_scores(self) -> pd.DataFrame:
        assert self.feedback_df is not None
        mapping = {"negative": -1, "neutral": 0, "positive": 1}
        self.feedback_df["sentiment_score"] = self.feedback_df["predicted_sentiment"].map(mapping)
        self.feedback_df["weighted_sentiment_score"] = (
            self.feedback_df["sentiment_score"] * self.feedback_df["sentiment_confidence"]
        )
        return self.feedback_df

    def aggregate_customer_sentiment(self) -> pd.DataFrame:
        assert self.feedback_df is not None
        agg = self.feedback_df.groupby("customerID").agg(
            sentiment_score=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            feedback_count=("feedback_text", "count"),
            avg_confidence=("sentiment_confidence", "mean"),
        ).round(3)
        agg = agg.rename(columns={"sentiment_score": "avg_sentiment"}).fillna({"sentiment_std": 0})
        return agg.reset_index()

    def create_sentiment_visualizations(self) -> None:
        if plt is None:
            print("Plotting libs not available; skipping figures")
            return
        assert self.feedback_df is not None
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        self.feedback_df["predicted_sentiment"].value_counts().plot(kind="bar", ax=axes[0])
        axes[0].set_title("Sentiment Distribution")
        self.feedback_df["sentiment_confidence"].plot(kind="hist", bins=20, ax=axes[1])
        axes[1].set_title("Confidence Distribution")
        plt.tight_layout()

    def save_sentiment_results(self) -> pd.DataFrame:
        assert self.feedback_df is not None
        out_path = DATA_DIR / "processed_feedback_sentiment.csv"
        self.feedback_df.to_csv(out_path, index=False)
        cust = self.aggregate_customer_sentiment()
        cust.to_csv(DATA_DIR / "customer_sentiment_scores.csv", index=False)
        return cust

    def run_complete_sentiment_analysis(self) -> pd.DataFrame:
        self.load_sentiment_model()
        self.load_feedback_data()
        self.process_feedback_sentiment()
        self.calculate_sentiment_scores()
        return self.save_sentiment_results()


def main() -> None:
    analyzer = SentimentAnalyzer()
    analyzer.run_complete_sentiment_analysis()


if __name__ == "__main__":
    main()


