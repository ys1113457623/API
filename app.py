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
from fastapi import FastAPI, Query
import pickle
import nltk
from six.moves import range
import six
import numpy as np
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
def padsequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the beginning or the end
    if padding='post.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def fake_news_det(news):

    x = [news]
    x = tokenizer.texts_to_sequences(x)

    
    x = padsequences(x,maxlen=700)
    
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