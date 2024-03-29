# -*- coding: utf-8 -*-
"""SpamDetection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tE-1aEUiC4OAEFZghhEoUYqNeHYMxLiK
"""

# Spam Detection ML
# (1) -> Spam
# (0) -> Not Spam

# Libraries
import numpy as np
import pandas as pd
import nltk # Natual Language Tool Kit
from nltk.corpus import stopwords
import string

# Data Loading 
from google.colab import files
uploaded = files.upload()

# CSV file reading
data_frame = pd.read_csv('spam_or_not_spam.csv')
data_frame.head()

# Data Set Dimensions
data_frame.shape

# Columns
data_frame.columns

# Removing duplicates
data_frame.drop_duplicates(inplace = True)
data_frame.shape

# Missing Data (NAN, NaN, na)
data_frame.isnull().sum()
data_frame = data_frame.dropna()

data_frame.shape

# Stopwords package
nltk.download('stopwords')

# Text Processing
def text_processing(text):
  rm_punc = [char for char in text if char not in string.punctuation] # Punctuation
  rm_punc = ''.join(rm_punc)
  clean_words = [word for word in rm_punc.split() if word.lower() not in stopwords.words('english')]# Stopwords
  return clean_words

data_frame['email'].head().apply(text_processing)

# Converting text to a matrix of tokens
from sklearn.feature_extraction.text import CountVectorizer
mail = CountVectorizer(analyzer=text_processing).fit_transform(data_frame['email'])

# Data Splitting 80/20
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mail, data_frame['label'], test_size=0.20, random_state=0)

# Shape of messages_bow
mail.shape

# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(X_train, y_train)

# Predictions
print(classifier.predict(X_train))

# Values
print(y_train.values)

# Evaluation of the model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
predictions = classifier.predict(X_train)
print(classification_report(y_train, predictions))
print('\nConfusion Matrix : \n', confusion_matrix(y_train, predictions))
print('\nAccuracy score : ', accuracy_score(y_train, predictions))

# Predictions
print(classifier.predict(X_test))

# Values
print(y_test.values)

# Evaluation of the model on test data
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))
print('\nConfusion Matrix : \n', confusion_matrix(y_test, predictions))
print('\nAccuracy score : ', accuracy_score(y_test, predictions))