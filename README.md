# twitter-comment

## I. Concept

###  Twitter Sentiment Analysis Dataset:
#### This is an entity-level sentiment analysis dataset of social network comments on Twitter. Given a message and an entity, the task is to judge the sentiment of the message about the entity.
#### This is a multi-class classification problem, which simply means the dataset has more than 2 classes (binary classifier). The three classes corresponding to sentiments:

```
     -1 - Negative
      0 - Neutral
      1 - Positive
```

#### We regard messages that are not relevant to the entity (i.e. Irrelevant) as Neutral.

## II. User Guide

###  Step 1: Get data

#### Extract file DataFrame file > 25MB, then the file need to be zipped 
```
Unzipped file DataFrame.zip --> to get file DataFrame.sav 
```

###  Step 2: Get a virtual environment
```
python -m venv "venv_name"
```

```
venv_name\Scripts\activate
```

###  Step 3: Install required libraries 
```
pip install -r requirements.txt
```

###  Step 4: Run the App
```
streamlit run app.py
```

## III. Structure
#### requirements.txt all required libraries
#### train.py         -- train model with the datasets from Kaggle -- file name 'twitter-training.csv'
#### test.py          -- test accuracy of model with datasets -- file name 'twitter-validation.csv' --> return 'final_result.csv'
#### app.py           -- this file is used to run the result on Streamlit
#### final_result.csv -- testing result of the model, column named 'Sentiment_test' is the predicted result of model and column named 'Sentiment' is true value
#### DataFrame.zip    -- This is DataFrame.sav, you need to unzip this file (as the User's Guide on Step 1) after clone this repository 
#### model1.sav       -- Saved train model is stored in the file
#### twitter-practices.ipynb -- Jupyter Notebook for practicing
