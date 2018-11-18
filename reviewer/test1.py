#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 12:45:46 2018
@author: simon
Neural Network Summary Creator for Amazon Reviews
"""
# Load LSTM network and generate text
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import os

current_dir = os.path.dirname(__file__)
filename = os.path.join(current_dir, 'data.csv')

# preprocessing functions for cleaning up and culling dataset
import pandas
def json_to_df(file_name):
    df = pandas.read_json(file_name, lines='true')
    return df
def filter_summary_len(df, min, max=9999999):
    mask = (df['summary'].str.len() > min) & (df['summary'].str.len() < max)
    return df.loc[mask]
def filter_review_len(df, min, max=9999999):
    mask = (df['reviewText'].str.len() > min) & (df['reviewText'].str.len() < max)
    return df.loc[mask]
def filter_stars(df, stars):
    mask = (df['overall'] == stars)
    return df.loc[mask]
def filter_helpful_min(df, up, down=0):
    mask = (df['helpful'][0] > up) & (df['helpful'][1] > down)
    return df.loc[mask]
def filter_helpful_max(df, up, down=0):
    mask = (df['helpful'][0] < up) & (df['helpful'][1] < down)
    return df.loc[mask]

stopWords = ("a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", \
             "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", \
             "could", "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from", \
             "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", \
             "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", "like", "likely", "may", \
             "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often", "on", \
             "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she", "should", \
             "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", \
             "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were", "what", \
             "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "yet", \
             "you", "your", "ain't", "aren't", "can't", "could've", "couldn't", "didn't", "doesn't", \
             "don't", "hasn't", "he'd", "he'll", "he's", "how'd", "how'll", "how's", "i'd", "i'll", \
             "i'm", "i've", "isn't", "it's", "might've", "mightn't", "must've", "mustn't", "shan't", \
             "she'd", "she'll", "she's", "should've", "shouldn't", "that'll", "that's", "there's", \
             "they'd", "they'll", "they're", "they've", "wasn't", "we'd", "we'll", "we're", "weren't", \
             "what'd", "what's", "when'd", "when'll", "when's", "where'd", "where'll", "where's", "who'd", "who'll", \
             "who's", "why'd", "why'll", "why's", "won't", "would've", "wouldn't", "you'd", "you'll", "you're",
             "you've")
unwanted_chars = ('\r', '"', "'", '(', ')', '*', ':', ';', '.', \
                  '[', ']', '_', '\xbb', '\xbf', '\xef', '®', '=', '\n')
def keywords(model_df, comment, limit=5):
    '''Given a model pandas dataframe and a comment string, returns the relevant keyword list'''
    summary_mask = ['summary']
    summaries = model_df[summary_mask]
    # review_mask = ['reviewText']
    # reviews = model_df[review_mask]
    mask = ['reviewText', 'overall', 'summary']
    df = model_df[mask]
    mask = (df['reviewText'].str.len() > 250) & (df['summary'].str.len() > 25)
    df = df[mask]
    reviews = df['reviewText'].str.lower()
    reviews = reviews.append(pd.Series([comment.lower()]), ignore_index=True)
    vect = TfidfVectorizer(sublinear_tf=True, analyzer='word', stop_words='english')
    # X = vect.fit_transform(df['reviewText']).toarray()
    X = vect.fit_transform(reviews).toarray()
    # r = df['summary'].copy()
    df = pd.DataFrame(X, columns=vect.get_feature_names())
    wantedWords = list(df.columns.values)
    review = str(reviews.values[-1]).lower()
    review = "".join(c for c in review if c not in unwanted_chars)
    review = "".join(c + " " for c in review.split(" ") if c not in stopWords)
    review = "".join(c + " " for c in review.split(" ") if c in wantedWords)
    num_words = min(limit, np.count_nonzero(df.values[-1]))
    top_n_indices = np.argsort(-(df.values[-1]))[0:num_words]
    top_n_words = [wantedWords[i] for i in top_n_indices]
    return top_n_words

def get_kws(sentence, limit=5):
    df = pd.read_csv(filename)
    kws = keywords(df, sentence, limit)
    return kws

