'''
Acknowledgments
This dataset comes from the UCI Machine Learning Repository. Any publications that use this data should cite the repository as follows:
Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. 
Irvine, CA: University of California, School of Information and Computer Science.
This specific dataset can be found in the UCI ML Repository at this URL

'''

import numpy as np
import pandas as pd
import os
import re
os.chdir('D:/project/dl/Stanford/data')

data = pd.read_csv('uci-news-aggregator.csv')
data.head()

## Pre-processing: Data Cleaning

# convert all text to small letters
# remove punctuations
# get rid of extras spaces

data['TITLE'] = [re.sub('\s\W',' ',text) for text in data['TITLE']]
data['TITLE'] = [re.sub('\W\s',' ',text) for text in data['TITLE']]
data['TITLE'] = [re.sub('\s+',' ',text) for text in data['TITLE']]
data['TITLE'] = [text.lower() for text in data['TITLE']]

data.head()

from sklearn.model_selection import train_test_split # split data into train and test
# function for encoding categories
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer # convert text to vector

vect = CountVectorizer().fit(data['TITLE'])
text_data_vec = vect.transform(data['TITLE'])
print('text_data_vec: \n {}'.format(repr(text_data_vec)))

features_name = vect.get_feature_names()
print('Number of features: {}'.format(len(features_name)))
print('\nFirst 10 features: {}'.format(features_name[:20]))
print('\nEvery 5000th features: {}'.format(features_name[::5000]))

encoder = LabelEncoder()
y = encoder.fit_transform(data['CATEGORY'])

X_train, X_test, y_train, y_test = train_test_split(text_data_vec, y,test_size= 0.25, random_state=42 )

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

# fit logistic model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

lr1 = LogisticRegression()
lr1.fit(X_train, y_train)
y_pred = lr1.predict(X_test)

print('Confusion Matrix: \n {}'.format(confusion_matrix(y_test, y_pred)))
print('Classification Report: \n {}'.format(classification_report(y_test, y_pred)))

# the Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)

print('\nNaive Bayes Score: \n {}'.format(nb.score(X_test, y_test)))
print('\nNaive Bayes Confusion Matrix: \n {}'.format(confusion_matrix(y_test, y_pred)))
print('\nNaive Bayes Classification Report: \n {}'.format(classification_report(y_test, y_pred)))
