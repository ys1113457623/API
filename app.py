# # -*- coding: utf-8 -*-
# """
# Created on Tue Nov 17 21:40:41 2020
# @author: win10
# """

# # 1. Library imports

import os

from yaml import load
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import uvicorn
from fastapi import FastAPI
from typing import Union
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, Query
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
import pandas as pd
# 2. Create the app object
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI()

loaded_model = pickle.load(open('finalmodel.pkl', 'rb'))
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
# fake_df = pd.read_csv('Fake.csv')
# real_df = pd.read_csv('True.csv')
# fake_df.drop(['date', 'subject'], axis=1, inplace=True)
# real_df.drop(['date', 'subject'], axis=1, inplace=True)

# fake_df['class'] = 0
# real_df['class'] = 1


# news_df = pd.concat([fake_df, real_df], ignore_index=True, sort=False)
# news_df['text'] = news_df['title'] + news_df['text']
# news_df.drop('title', axis=1, inplace=True)


# features = news_df['text']
# targets = news_df['class']


# X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.20, random_state=18)
# y = news_df["class"].values
# maxlen=700
# #Converting X to format acceptable by gensim, removing annd punctuation stopwords in the process
# X = []
# stop_words = set(nltk.corpus.stopwords.words("english"))
# tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
# for par in news_df["text"].values:
#     tmp = []
#     sentences = nltk.sent_tokenize(par)
#     for sent in sentences:
#         sent = sent.lower()
#         tokens = tokenizer.tokenize(sent)
#         filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
#         tmp.extend(filtered_words)
#     X.append(tmp)

# del news_df

def fake_news_det(news):

    x = [news]

    
    x = tokenizer.texts_to_sequences(x)
    print("This is the output",x)
    # x = pad_sequences(x,maxlen=700)
    predict = loaded_model.predict(x)[0].astype(float) * 100
    return predict    


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}


@app.get('/{predict}')
def get_name(predict: str):
    pred = fake_news_det(predict)

    return float(pred)

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app)
    
#uvicorn app:app --reload

# from flask import Flask, render_template, request
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import PassiveAggressiveClassifier
# import pickle
# import tensorflow as tf
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import nltk
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer
# from sklearn import metrics

# app = Flask(__name__)
# tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
# loaded_model = pickle.load(open('finalmodel.pkl', 'rb'))

# # We need to fit the TFIDF VEctorizer
# fake_df = pd.read_csv('Fake.csv')
# real_df = pd.read_csv('True.csv')
# fake_df.drop(['date', 'subject'], axis=1, inplace=True)
# real_df.drop(['date', 'subject'], axis=1, inplace=True)

# fake_df['class'] = 0
# real_df['class'] = 1


# news_df = pd.concat([fake_df, real_df], ignore_index=True, sort=False)
# news_df['text'] = news_df['title'] + news_df['text']
# news_df.drop('title', axis=1, inplace=True)


# features = news_df['text']
# targets = news_df['class']
# maxlen = 700 



# X_train, X_test, y_train, y_test = train_test_split(
#     features, targets, test_size=0.20, random_state=18)

# y = news_df["class"].values
# #Converting X to format acceptable by gensim, removing annd punctuation stopwords in the process
# X = []
# stop_words = set(nltk.corpus.stopwords.words("english"))
# tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
# for par in news_df["text"].values:
#     tmp = []
#     sentences = nltk.sent_tokenize(par)
#     for sent in sentences:
#         sent = sent.lower()
#         tokens = tokenizer.tokenize(sent)
#         filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
#         tmp.extend(filtered_words)
#     X.append(tmp)

# del news_df

# def fake_news_det(news):
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(X)
#     x = [news]
#     x = tokenizer.texts_to_sequences(x)
#     x = pad_sequences(x,maxlen=maxlen)
#     predict = loaded_model.predict(x)[0].astype(float) * 100
#     return predict

# # Defining the site route


# @app.route('/')
# def home():
#     return render_template('index.html')


# @app.route('/predict', methods=['GET'])
# def predict():
#     if request.method == 'GET':
#         message = request.args['query']
#         pred = fake_news_det(message)
#         print(pred)
#         return str(pred)
#     else:
#         return render_template('index.html', prediction="Something went wrong")


# if __name__ == "__main__":
#     app.run(debug=True)