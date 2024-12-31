# -*- coding: utf-8 -*-
import pandas as pd
import re, os, sys
import kagglehub
import pickle
from pathlib import Path

sys.setrecursionlimit(5000)
script_location = Path('social-network/NLP-Sentiment-Analyzer').absolute()

# Download latest version
path = kagglehub.dataset_download("jp797498e/twitter-entity-sentiment-analysis")

def load_data():
    # Expanding the dispay of text sms column
    pd.set_option('display.max_colwidth', 1)

    csv_folder = [f for f in os.listdir(Path(f'{path}').absolute()) if f.endswith('validation.csv')]
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

def load_model():
    model1 = pickle.load(open('model1.sav', 'rb'))
    return model1

def predict(phrase):
    return int(model1.predict([phrase]))

model1 = load_model()
def main():  
    # Load data
    data = load_data()

    # Transform data
    df_test = transform_data(data)

    # Applying model
    df_test['Sentiment_Test'] = df_test['Tweet_content'].apply(predict)

    # Export resulf to excel file
    df_test.to_csv('final_result.csv')
    print('Programme Completed')

if __name__ == '__main__':
    main()
