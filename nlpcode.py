# -*- coding: utf-8 -*-
"""
In this assignment:
·        You learn how use an LSTM for text classification an NLP task using KERAS
.        The text classification task is sentiment analysis using the IMDB movie review dataset


What is sentiment analysis?

Sentiment analysis (or opinion mining) is a natural language processing technique used to determine whether data is positive, negative or neutral.
REF: https://monkeylearn.com/sentiment-analysis/

E,g,

"I love how Zapier takes different apps and ties them together" → Positive
"I still need to further test Zapier to say if its useful for me or not" → Neutral
"Zapier is sooooo confusing to me" → Negative

Dataset

The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset.

It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.
REF: https://monkeylearn.com/blog/sentiment-analysis-examples/

"""

import subprocess
import sys


def install(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install("keras")
install("numpy")

import numpy
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
# Set the model to be a simple feed-forward (layered) architecture
# See https://keras.io/api/models/ and https://keras.io/api/models/sequential/
# not to be confused with a sequence-based alg/model to process sequential data
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# Add a 100 unit LSTM layer
# https://keras.io/api/layers/recurrent_layers/lstm/
model.add(LSTM(100))
# Add a dense output layer with units=1 and sigmoid activation unit 
# https://keras.io/api/layers/core_layers/dense/ 
model.add(Dense(1, activation='sigmoid'))
# Compile the model, specifying (1) the Adam optimizer,
# (2) the 'BinaryCrossentropy' loss function, and (3) metrics=['accuracy']
# See https://keras.io/api/models/model_training_apis/
model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# fit the keras model on the _TRAIN_ dataset set epochs to 3, batch_size to 16
model.fit(X_train, y_train, epochs=3, batch_size=16)

# FINALLY, evaluate your model on TEST dataset
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
