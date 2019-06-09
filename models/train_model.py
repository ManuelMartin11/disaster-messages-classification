import re
import numpy as np
import pickle
import sqlalchemy
import pandas as pd
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.stem import SnowBallStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def tokenize(text):
    wtok = RegexpTokenizer("\w+")
    stemmer = SnowballStemmer("english")
    text = str(text)
    re.sub("[^aA-zZ]", "", text)
    re.sub("-|_|\^", "", text)
    text = pre_clean(text)
    wlist = [stemmer.stem(w.lower())
            for w in wtok.tokenize(text)
            if w.lower()
            not in stopwords.words("english")]

    return wlist

def train():
    """ """

    # Load data
    engine = sqlalchemy.create_engine('sqlite:///../disastermessages.db'))
    df = pd.read_sql_table("messages_08_06_19_17", con=engine)
    X = df.message.values
    Y = df.values[:, 6:]
    categories = df.columns.values[6:]

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.33)

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ("multi", MultiOutputClassifier(RandomForestClassifier(100)))])

    parameters = {"multi__estimator__n_estimators" : [50, 100, 200],
                  "vect__ngram_range": [(1,1), (1,2)]}

    cv = GridSearchCV(pipeline, parameters, iid=True, cv=5
    cv.fit(xtrain, ytrain)

    with open(r"models/model.pkl", "wb") as f:
        pickle.dumps(f, cv)
