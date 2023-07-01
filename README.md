# Sentiment_analysis
Sentiment Analysis is the process of computationally identifying and categorizing opinions expressed in a piece of text, determining whether the writer's attitude towards a particular topic, product, etc. is positive, negative, or neutral. This repository contains code that performs Sentiment Analysis on text data using the NaiveBayes Classifier from the nltk library.

# Prerequisites
This code is written in Python 3 and requires the following libraries:

* numpy
* pandas
* nltk
* wordcloud
* matplotlib
You can install these libraries using pip:
```
pip install numpy pandas nltk wordcloud matplotlib
```
# Dataset
The Sentiment dataset used in this code is stored in a CSV file named "Sentiment.csv". Make sure to place the dataset file in the same directory as the code file.

# Usage
Import the required libraries:
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
```
# Load and preprocess the dataset:
```
# Load the dataset
data = pd.read_csv('Sentiment.csv')

# Keep only the necessary columns
data = data[['text', 'sentiment']]

# Split the dataset into train and test sets
train, test = train_test_split(data, test_size=0.1)

# Remove neutral sentiments
train = train[train.sentiment != "Neutral"]
```
# Visualize word clouds of positive and negative words:
```
train_pos = train[train['sentiment'] == 'Positive']
train_neg = train[train['sentiment'] == 'Negative']

# Define a function to generate word clouds
def wordcloud_draw(data, color='black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                            and not word.startswith('@')
                            and not word.startswith('#')
                            and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color=color,
                          width=2500,
                          height=2000).generate(cleaned_word)
    plt.figure(figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

# Generate word cloud for positive words
print("Positive words")
wordcloud_draw(train_pos['text'], 'white')

# Generate word cloud for negative words
print("Negative words")
wordcloud_draw(train_neg['text'])
```
# Remove stopwords from the training set:
```
tweets = []
stopwords_set = set(stopwords.words("english"))

for index, row in train.iterrows():
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
                     if 'http' not in word
                     and not word.startswith('@')
                     and not word.startswith('#')
                     and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    tweets.append((words_without_stopwords, row.sentiment))
```
# Extract word features and train the NaiveBayes Classifier:
```
# Extract word features
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features

w_features = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
```
# Training the Naive Bayes classifier
```
training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
Perform sentiment classification on the test set:
python
Copy code
neg_cnt = 0
pos_cnt = 0

for obj in test['text']:
    res = classifier.classify(extract_features(obj.split()))
    if res == 'Negative':
        neg_cnt += 1
    if res == 'Positive':
        pos_cnt += 1

print('[Negative]: %s/%s ' % (len(test[test['sentiment'] == 'Negative']), neg_cnt))
print('[Positive]: %s/%s ' % (len(test[test['sentiment'] == 'Positive']), pos_cnt))
```
# Conclusion
This project demonstrates the use of nltk and the NaiveBayes Classifier for Sentiment Analysis. It performs well for negative comments but may face challenges with ironic, sarcastic, or contextually complex tweets.
