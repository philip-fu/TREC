import pandas as pd
import numpy as np
import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers import Embedding, Dropout, Dense, LSTM, Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

from src.word_embedding import load_glove, get_embedding_matrix, clean_words

class LSTMClassifier(object):
    def __init__(self, config):
        self.config = config

        # load embeddings
        self.embedding_index = load_glove()
        self.model = Sequential()

    def preprocess(self, X, mode='train'):
        """tokenize and pad/truncate to same length.
        """
        X = clean_words(X)
        if mode == 'train':
            self.tokenizer = Tokenizer(oov_token='UNK')
            self.tokenizer.fit_on_texts(X)
            X = self.tokenizer.texts_to_sequences(X)
            X = pad_sequences(X, maxlen=self.config['max_length'], padding='pre')
        elif mode == 'test':
            x_test_encoded = list()
            for sentence in X:
                x_test = [self.tokenizer.word_index[w] for w in sentence if w in self.tokenizer.word_index]
                x_test_encoded.append(x_test)
            X = pad_sequences(x_test_encoded, maxlen=self.config['max_length'], padding='pre')

        return X


    def preprocess_label(self, y, mode='train'):
        if mode == 'train':
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
            y = self.label_encoder.transform(y)
            y = np_utils.to_categorical(y)
            self.num_classes = y.shape[1]
        elif mode == 'test':
            y = self.label_encoder.transform(y)
            y = np_utils.to_categorical(y, num_classes=len(self.label_encoder.classes_))

        return y


    def fit(self, X_train, y_train):
        """
        Args:
            X_train: pd.Series.
            y_train: pd.Series 
           
        """
        X_train = self.preprocess(X_train.tolist(), mode='train')
        y_train = self.preprocess_label(y_train.tolist(), mode='train')

        self.embedding_matrix = get_embedding_matrix(self.embedding_index,
                                                     word_index=self.tokenizer.word_index,
                                                     embedding_dim=self.config['embedding_dim'])

        self.model.add(Embedding(len(self.tokenizer.word_index)+1, self.config['embedding_dim'], weights=[self.embedding_matrix], input_length=self.config['max_length'], trainable=self.config['train_embedding']))
        self.model.add(LSTM(512))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        adam = Adam(lr=self.config['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])


        self.history = self.model.fit(X_train, y_train,
                                      batch_size=self.config['batch_size'],
                                      shuffle=True,
                                      verbose=2,
                                      epochs=self.config['epochs'])

    def infer(self, X_test):
        X_test = self.preprocess(X_test.tolist(), mode='test')
        preds = self.model.predict(X_test)
        preds = np.argmax(preds, axis=1)

        return self.label_encoder.inverse_transform(preds)


    def evaluate(self, X_test, y_test):
        y_pred = self.infer(X_test).tolist()
        return round(np.mean(y_test == y_pred).item(), 2)
