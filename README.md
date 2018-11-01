# test_analysis_python

If jupyter file fails to load, please copy the whole link and visit the following page:
https://nbviewer.jupyter.org/ and paste the link in the search bar

# Introduction

Text Analytics is the process of exploring and analyzing large amounts of unstructured/structured text data and  identify class, patterns, topics, keywords and other attributes in the data. For example, if we want to classify an email message as either spam or non-spam email, reviews of users about any movie on IMDB, classify text/document as per it's content.

Step by step approach:

 1. Get labelled training data (structured/unstructured text data with labels (sentiments, topics)
 2. Cleaning and pre-processing 
 3. Extract features from training text data (Tokenisation -  breaking up a sequence of strings into pieces such as words, keywords,         phrases, symbols and other elements called tokens)
 4. Train parameters of classifiers (I have tried two models Logistic Regression & Naives)
 5. Test the model on unseen data (Performance metrics calculation: Confusion Matrix and Classification Report)


In this repositories, I have tried to solve two problems:

 *1. Sentiment Analysis of Movie Reviews*
 
 *2. News headline classification*

# Sentiment Analysis of Movie Reviews

[DATASET] (http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.bib)

This is a two-class classification problem where user reviews are categorised into "positive" and "negative". Below should be the folder structure in order to run *imdb_movie_review_sentiment_analysis.ipynb*

# News headlines classification

[DATASET](https://www.kaggle.com/uciml/news-aggregator-dataset/home)

This dataset contains headlines, URLs, and categories for 422,937 news stories collected by a web aggregator between March 10th, 2014 and August 10th, 2014.
News categories are divided into business, science and technology, entertainment, and health.
This is a multiclass classification problem. 
Python code file: *news_headlines_classification*
