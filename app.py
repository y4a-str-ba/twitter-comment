# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import re
import pickle
import zipfile
import nltk
nltk.download('stopwords')

import seaborn as sns
import matplotlib.pyplot as plt
import wordcloud as wordcloud
import os, urllib
import sys
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from PIL import Image
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sys.setrecursionlimit(15000)
script_location = Path('__filepath__').absolute()

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    '''Main function that will run the whole app
    
    '''
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.title('Twitter Comment Sentiment Analysis')
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Introduction", "Run the app", "Show the source code"])
    if app_mode == "Introduction":
        st.sidebar.success('To continue select "Run the app".')
        intro()
    elif app_mode == "Show the source code":
        st.write('Soure Code: https://github.com/baong28/twitter-comment')
        show_source_code(df)
    elif app_mode == "Run the app":
        run_the_app()
        
# Load saved model
@st.cache_resource
def load_model():
    '''Load save trained logistics regression model from pickle file
    
    '''
    model1 = pickle.load(open('model1.sav', 'rb'))
    return model1

@st.cache_resource
# def load_dataframe():
#     '''Load DataFrame to make basic data analyzing
    
#     '''
#     df = pickle.load(open('DataFrame.sav', 'rb'))
#     return df

def load_dataframe():
    """Load DataFrame từ file .zip, tự động tìm file .sav"""
    zip_path = 'DataFrame.zip'
    extract_path = script_location / "DataFrame"

    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Tìm file .sav trong thư mục giải nén
    for root, _, files in os.walk(extract_path):
        for file in files:
            if file.endswith('.sav'):
                sav_path = os.path.join(root, file)
                with open(sav_path, 'rb') as f:
                    df = pickle.load(f)
                return df

    raise FileNotFoundError("Không tìm thấy file .sav trong file zip!")

@st.cache_data 
def add_stop_word():
    '''Add StopWord relate to 
    
    '''
    stopword_list = {'movi', 'film', 'one', 'hi', 'thi'}
    stopword = stopwords.words('english')
    for word in stopword_list:
        stopword.append(word)
    return stopword

# Most common word
@st.cache_data 
def most_common(df):
    '''Return most common keyword
    
    '''    
    vocab = Counter()
    for phrase in df.preprocessed:
        for word in phrase.split(' '):
              vocab[word] += 1
              
    most_common = vocab.most_common(10)
    return pd.DataFrame(most_common, columns=['Word', 'Count'])

@st.cache_data 
def most_common_negative(df):
    '''Return most common keyword in negative review and remove stop words
    
    '''    
    vocab = Counter()
    for phrase in df[df['Sentiment'] < 0].keyword_list:
        for word in phrase:
              vocab[word] += 1
              
    most_common_negative = vocab.most_common(100)
    most_common_negative = pd.DataFrame(most_common_negative, columns=['Word', 'Count'])
    most_common_negative = most_common_negative[~most_common_negative['Word'].isin(stopword)]
    return most_common_negative

@st.cache_data 
def most_common_positive(df):
    '''Return most common keyword in positive review and remove stop words
    
    '''    
    vocab = Counter()
    for phrase in df[df['Sentiment'] > 0].keyword_list:
        for word in phrase:
              vocab[word] += 1
              
    most_common_positive = vocab.most_common(100)
    most_common_positive = pd.DataFrame(most_common_positive, columns=['Word', 'Count'])
    most_common_positive = most_common_positive[~most_common_positive['Word'].isin(stopword)]
    return most_common_positive

# Download a single file and make its content available as a string.
@st.cache_data (show_spinner=False)
def get_file_content_as_string(path):
    '''To read source code from github
    
    '''
    url = 'https://raw.githubusercontent.com/baong28/twitter-comment/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def createWordCloud(df):
    '''Create World Cloud visualization
    
    '''
    word = []  
    for phrase in df.preprocessed:
        for i in phrase.split(' '):
              word.append(i.lower())   
    occurrences = Counter(word)
    cloud = wordcloud.WordCloud(background_color="white", width=1920, height=1080, min_font_size=8)
    cloud.generate_from_frequencies(occurrences)
    myimage = cloud.to_array()
    plt.imshow(myimage, interpolation = 'nearest')
    plt.axis('off')
    plt.show()    

def preprocessor(text):
    """ Return a cleaned version of text
        
    """   
    # Remove HTML markup
    text = re.sub('<[^>]*>', '', text)
    # Remove emoticons
    text = re.sub('(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)
    # Remove any non-word character and digit
    text = re.sub('[^A-Za-z ]+', '', text)
    # Also Convert to lower case
    text = (re.sub('[\W]+', ' ', text.lower()))    
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

# Predict function by model 1, non-convertred sentiment
def model1_predict(text):
    '''Convert number result to text result for Model 1
    
    '''
    prediction = model1.predict([text])
    if prediction==-1:  
        return 'Negative'
    # elif prediction==1:
    #     return 'Somewhat negative'
    elif prediction==0:
        return 'Neutral'
    # elif prediction==3:
    #     return 'Somewhat positive'
    elif prediction==1:
        return 'Positive'
    
# Introduction
def intro():
    '''Section for introduction and dataset discovery
    
    '''
    st.title('About Dataset')
    image = Image.open('logo-1.jpg')
    st.image(image)
    st.markdown('''
                Twitter Sentiment Analysis Dataset:
                ```
                This is an entity-level sentiment analysis dataset of twitter. 
                Given a message and an entity, the task is to judge the sentiment of the message about the entity.
                
                ```

                This is a multi-class classification problem, which simply means the data set have more than 2 classes(binary classifier). The three classes corresponding to sentiments:

                ```
                -1 - Negative
                 0 - Neutral
                 1 - Positive
                ```          

                We regard messages that are not relevant to the entity (i.e. Irrelevant) as Neutral.

                ''')
    st.markdown('## Data Exploration')
    st.markdown('Let us have a look at first few phrases of the training dataset:')
    st.write(df.sample(10))
    st.markdown('## Number of comments across categories')
    st.write(df['Sentiment'].value_counts())

    fig, ax = plt.subplots()
    sns.barplot(x=df['Sentiment'].value_counts().index, y=df['Sentiment'].value_counts())
    st.pyplot(fig)
    st.markdown('## Most common word in review')
    st.write(most_common(df))
    st.markdown('''Most common word contain very neutral words like 'the, a, of, and, .etc'. We will try eliminate these word using stopword.
                ''')
    st.markdown('### Most common word in positive review')
    st.write(most_common_positive(df))
    st.markdown('### Most common word in negative review')
    st.write(most_common_negative(df))
    st.markdown('**Word Cloud**')
    createWordCloud(df)
    st.pyplot(fig)
    
# Function to run the prediction demo   
def run_the_app():
    '''Execute the app to predict sentiment base on input comment
    
    '''
    user_input = st.text_input("Review Text", 'This is a comment on Twitter')
    st.write('Prediction by model 1, training by using original sentiment')
    st.write(model1_predict(user_input))

def show_source_code(df):
    '''Code to print out my source code
    
    
    '''
    st.code(get_file_content_as_string("train.py")) 
    # Training model on non-converted Sentiment
    X = df['Tweet_content']
    y = df['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    prediction = model1.predict(X_test)
    
    #Print Result
    st.markdown('## Model 1')
    st.write('Model 1 accuracy:',accuracy_score(y_test,prediction))
    st.markdown('**Model 1 Confusion matrix:**')
    cm1 = confusion_matrix(y_test,prediction)
    fig, ax = plt.subplots()
    sns.heatmap(cm1, annot=True,fmt='g', cmap='Blues', xticklabels=['-1', '0', '1'], yticklabels=['-1', '0', '1'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot(fig)
             
            
if __name__ == "__main__":
    # Load saved model
    model1 = load_model()
    df = load_dataframe()
    stopword = add_stop_word()
    main()
    

