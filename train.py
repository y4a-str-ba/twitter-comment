#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Saturday, Dec 28th 2024 at 02:34 pm
@author: baong28

@reference: https://github.com/mkhoa/rottentomatoes.git
"""
import pandas as pd
import sys, re, os
import nltk
import pickle
import kagglehub
import pandas as pd

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import wordcloud as wordcloud
from collections import Counter
import streamlit as st

sys.setrecursionlimit(5000)
script_location = Path('social-network/NLP-Sentiment-Analyzer').absolute()

# Download latest version
path = kagglehub.dataset_download("jp797498e/twitter-entity-sentiment-analysis")

def load_data():
    # Expanding the dispay of text sms column
    pd.set_option('display.max_colwidth', 1)

    csv_folder = [f for f in os.listdir(Path(f'{path}').absolute()) if f.endswith('training.csv')]
    data = []
    for f in csv_folder:
        data.append(pd.read_csv(path + '/' + f, header=None))

    df = pd.concat(data, ignore_index=True)
    return df

def transform_data(df):
    df.rename(columns={0:'Tweet_ID', 1: 'Entity', 2: 'Sentiment', 3: 'Tweet_content'}, inplace=True)
    df['Sentiment'] = df['Sentiment'].map({"Negative": -1, "Irrelevant": 0, "Neutral": 0, "Positive": 1})
    df['Tweet_content'] = df['Tweet_content'].astype(str).fillna('')
    return df

def preprocessor(text):
    """ Return a cleaned version of text
        
    """   
    # Remove HTML markup
    text = re.sub(r'<[^>]*>', '', text)
    # Replace n't at note
    text = re.sub(r"n't", 'not', text)
    # Remove emoticons
    text = re.sub(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)
    # Remove any non-word character and digit
    text = re.sub(r'[^A-Za-z ]+', '', text)
    # Also Convert to lower case
    text = (re.sub(r'[\W]+', ' ', text.lower()))    
    return text

def tokenizer_porter(text):
    """Split a text into list of words and apply stemming technic
    
    """
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]
          
def convert(sentiment):
    """Convert from 5 sentiment to 3 sentiment
    
    """
    if sentiment < 0:
        sentiment = -1 # Negative
    if sentiment == 0:
        sentiment = 0 # Neutral
    if sentiment > 0:
        sentiment = 1 # Positive
    return sentiment

def main():
    """Function to run training model
    
    """
    # Load data
    data = load_data()

    # Transform data
    df = transform_data(data)

    # Download stopwords
    nltk.download('stopwords')

    # Convert
    df['converted_sentiment'] = df['Sentiment'].apply(convert)
    df['preprocessed'] = df['Tweet_content'].apply(preprocessor)
    df['keyword_list'] = df['preprocessed'].apply(tokenizer_porter)
    pickle.dump(df, open('DataFrame.sav', 'wb')) 

    # Train model
    stopword = stopwords.words('english')
    X = df['Tweet_content']
    y = df['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    count = CountVectorizer(stop_words=stopword, preprocessor=preprocessor)

    # Construct pipeline
    clf = Pipeline([('vect', count), ('clf', LogisticRegression(random_state=0, max_iter=1000, n_jobs=6))])
    clf.fit(X, y)
    prediction = clf.predict(X_test)

    #Print Result
    print('accuracy:',accuracy_score(y_test, prediction))
    print('confusion matrix:\n',confusion_matrix(y_test, prediction))
    print('classification report:\n',classification_report(y_test, prediction))
        
    # Save trained model to disk
    pickle.dump(clf, open('model1.sav', 'wb'))
    
if __name__ == '__main__':
    main()

        
