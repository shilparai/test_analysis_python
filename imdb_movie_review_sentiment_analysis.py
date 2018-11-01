import os
import numpy as np
import pandas as pd
os.chdir('D:/project/dl/Stanford/data/aclImdb')

## Data Sources : http://ai.stanford.edu/~amaas/data/sentiment/
##Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). 
## Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).


from sklearn.datasets import load_files

"""
http://ai.stanford.edu/~amaas/data/sentiment/

download the data from above site. train and test both folders has two sub-folders : pos and neg
pos - > positive review
neg - > negative review
"""

# load train data
reviews_train = load_files('./train/')
text_train, y_train = reviews_train.data, reviews_train.target

print('type of text_train: {}'.format(type(text_train)))
print('length of text_train: {}'.format(len(text_train)))
print('access text_train[1]: \n{}'.format(text_train[1]))
print('\nlable[1]: {}'.format(y_train[1]))
print('\nunique label: {}'.format(set(y_train)))

print('Samples per class (training): {}'.format(np.bincount(y_train)))

## load test data
reviews_test = load_files('./test/')
text_test, y_test = reviews_test.data, reviews_test.target
print('type of text_train: {}'.format(type(text_train)))
print('length of text_train: {}'.format(len(text_train)))
print('access text_train[1]: \n{}'.format(text_train[1]))
print('\nlable[1]: {}'.format(y_train[1]))
print('\nunique label: {}'.format(set(y_train)))

'''
Computing bag of words for a corpus of documents,consists of following steps:

1. Tokenization- split each documents into the words
2. Vocabulary building 
3. Encoding - for each document,count how often each of the words appear in the document

Below simple example to explain bag of words method
'''

sentence = ['In fact, every other movie in the world is better than this one. I would not watch it again']

from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer()
vector.fit(sentence)

print('Vocabulary size: {}'.format(len(vector.vocabulary_)))
print('Vocabulary content:\n {}'.format(vector.vocabulary_))

bag_of_words = vector.transform(sentence)
print('bag_of_words: {}'.format(repr(bag_of_words))) # difference between str() and repr() if we print string using repr() 
# function then it prints with a pair of quotes and if we calculate a value we get more precise value than str() function.

vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train)
print('X_train: \n {}'.format(repr(X_train)))

features_name = vect.get_feature_names()
print('Number of features: {}'.format(len(features_name)))
print('\nFirst 10 features: {}'.format(features_name[:20]))
print('\nEvery 5000th features: {}'.format(features_name[::5000]))


## let's fit model without any other cleaning process
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
print('Mean cross validation accuracy: {:.2f}'.format(np.mean(scores)))

## Predict sentiment for test data
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1,10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train,y_train)

print('Best cross-validation score: {:.2f}'.format(grid.best_score_))
print('Best parameters: ', grid.best_params_)

X_test = vect.transform(text_test)
print(" test score: {:.2f}".format(grid.score(X_test,y_test)))

## a token that appears only in a single document is unlikely to appear in the test set and is therefore not helpful.

vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
print('X_train with min_df: {}'.format(repr(X_train)))

feature_name = vect.get_feature_names()
print("First 50 features: \n {}".format(feature_name[:50]))
print("Every 500th features: \n {}".format(feature_name[::500]))

## fitting model on modified train data

grid = GridSearchCV(LogisticRegression(),cv=5,param_grid=param_grid)
grid.fit(X_train,y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best Parameters {}".format(grid.best_params_))

## Next step : Get rid of stop-words
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
print('Number of stop words: {}'.format(len(ENGLISH_STOP_WORDS)))
print('Every 5th word in ENGLISH_STOP_WORDS list: {}'.format(list(ENGLISH_STOP_WORDS)[::10]))

vect = CountVectorizer(min_df =5, stop_words = 'english').fit(text_train)
X_train = vect.transform(text_train)
print('X_train with stop words: \n{}'.format(repr(X_train)))

grid = GridSearchCV(LogisticRegression(), param_grid=param_grid,cv=5)
grid.fit(X_train,y_train)

print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best Parameters {}".format(grid.best_params_))
