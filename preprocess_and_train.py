import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

# Function to clean the tweet text
def clean_tweet(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove user mentions
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove non-alphanumeric characters
    text = text.lower().strip()  # Convert to lowercase and strip whitespaces
    return text

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

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

# Load the dataset
df = pd.read_csv('twitter_data.csv')

print("Cleaning and preprocessing tweets...")
# Clean the tweet text
df['cleaned_text'] = df['tweet'].apply(clean_tweet)

# Drop rows with NaN values in cleaned_text
df = df.dropna(subset=['cleaned_text'])

# Apply sentiment analysis
df['sentiment'] = df['cleaned_text'].apply(get_sentiment_score)

# Drop rows with NaN values in sentiment
df = df.dropna(subset=['sentiment'])

# Drop unnecessary columns
df.drop(columns=['tweet'], inplace=True)

# Save cleaned data
df.to_csv('cleaned_twitter_data.csv', index=False)
print("Data preprocessing complete. Cleaned data saved to 'cleaned_twitter_data.csv'.")

# Visualization: Sentiment Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=df, palette='viridis')
plt.title('Sentiment Distribution in Twitter Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Vectorize the tweet text
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']

# Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Train a Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X, y)

# Save the trained model
joblib.dump(svm_classifier, 'sentiment_model.pkl')
print("Trained model and TF-IDF vectorizer saved.")
