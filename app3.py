
import os

import uvicorn
from yaml import load
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, render_template, request, jsonify
import nltk
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
origins = ["*"]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ps = PorterStemmer()

model = pickle.load(open('finalmodel.pkl', 'rb'))
tfidfvect = pickle.load(open('tfidfvect2_main.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

def fake_news_det(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    prediction = model.predict(review_vect)[0].astype(float) * 100
    return prediction

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
    