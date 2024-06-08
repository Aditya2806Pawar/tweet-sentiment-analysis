import tkinter as tk
from tkinter import simpledialog
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Function to clean the tweet text
def clean_tweet(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove user mentions
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove non-alphanumeric characters
    text = text.lower().strip()  # Convert to lowercase and strip whitespaces
    return text

# Function to get sentiment score
def get_sentiment_score(text):
    score = analyzer.polarity_scores(text)
    compound_score = score['compound']
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Load the trained model
svm_model = joblib.load('sentiment_model.pkl')
# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# GUI for entering new tweet and displaying sentiment analysis results
def analyze_tweet():
    new_tweet = simpledialog.askstring("Enter New Tweet", "Enter the tweet for sentiment analysis:")
    if new_tweet:
        cleaned_new_tweet = clean_tweet(new_tweet)
        new_X = tfidf_vectorizer.transform([cleaned_new_tweet])
        predicted_sentiment = svm_model.predict(new_X)[0]
        sentiment_label.config(text="Predicted Sentiment: " + predicted_sentiment)
        # Apply sentiment analysis to the new tweet
        sentiment = get_sentiment_score(cleaned_new_tweet)
        sentiment_vader_label.config(text="VADER Sentiment: " + sentiment)
    else:
        sentiment_label.config(text="No tweet entered.")
        sentiment_vader_label.config(text="")

# GUI setup
root = tk.Tk()
root.title("Tweet Sentiment Analysis")
root.geometry("400x200")

# Button for analyzing tweet
analyze_button = tk.Button(root, text="Analyze Tweet", command=analyze_tweet)
analyze_button.pack(pady=10)

# Label for displaying sentiment analysis results
sentiment_label = tk.Label(root, text="")
sentiment_label.pack()
sentiment_vader_label = tk.Label(root, text="")
sentiment_vader_label.pack()

root.mainloop()
