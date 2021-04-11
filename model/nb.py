import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class NBClassifier:
    def __init__(self, config=None):
        self.model = MultinomialNB()

    def preprocessing(self, X, mode='train'):
        if mode == 'train':
            self.vectorizer = CountVectorizer(stop_words=None)
            return self.vectorizer.fit_transform(X)
        elif mode == 'test':
            return self.vectorizer.transform(X)

    def fit(self, X_train, y_train):
        X_train = self.preprocessing(X_train, mode='train')
        self.model.fit(X_train, y_train)

    def infer(self, X_test):
        X_test = self.preprocessing(X_test, mode='test')
        return self.model.predict(X_test)
