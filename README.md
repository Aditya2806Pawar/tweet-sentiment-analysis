# Tweet Sentiment Analysis

This project uses machine learning and natural language processing to analyze the sentiment of tweets related to the Russia-Ukraine war. The project employs a Support Vector Machine (SVM) classifier for sentiment prediction and the VADER sentiment analysis tool for comparative analysis. Additionally, a graphical user interface (GUI) is provided for entering new tweets and displaying sentiment results.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project aims to perform sentiment analysis on tweets about the Russia-Ukraine war. It involves:
1. Preprocessing and cleaning tweet data.
2. Training an SVM classifier using TF-IDF vectorization.
3. Using VADER for sentiment analysis.
4. Providing a user-friendly GUI for sentiment prediction on new tweets.

## Dataset

The dataset used in this project consists of tweets related to the Russia-Ukraine war. It is assumed to be in a CSV file named `twitter_data.csv`, containing the following column:
- `tweet`: The tweet text.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/tweet-sentiment-analysis.git
    cd tweet-sentiment-analysis
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Preprocessing and Training

1. Place your dataset (`twitter_data.csv`) in the project directory.
2. Run the preprocessing and training script:
    ```bash
    python preprocess_and_train.py
    ```
   This script will clean the tweets, perform sentiment analysis using VADER, vectorize the text using TF-IDF, train an SVM classifier, and save the trained model and vectorizer.

### GUI for Sentiment Analysis

1. Run the GUI script:
    ```bash
    python gui_sentiment_analysis.py
    ```
2. Enter a tweet in the prompt that appears and see the predicted sentiment using the SVM model and VADER.

## Files

- `preprocess_and_train.py`: Script for preprocessing, sentiment analysis with VADER, and training the SVM model.
- `gui_sentiment_analysis.py`: Script for the graphical user interface to analyze new tweets.
- `requirements.txt`: List of required Python packages.

## Results

The sentiment analysis results are displayed in the GUI. The project uses both an SVM classifier and VADER sentiment analysis to provide comprehensive sentiment insights. Additionally, a visualization of the sentiment distribution in the dataset is generated during preprocessing.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

Author-Aditya2806Pawar
