#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jake

Neural Network Summary Creator for Amazon Reviews - Sentence Level
"""

import tensorflow as tf
import tensorflow_hub as hub
import nltk.data
from lexrank import STOPWORDS, LexRank
from path import Path

embed = None
session = None
tokenizer = None
lxr = None

def init_session():
    '''Initialize TensorFlow universal-sentence-encoder, returning the session object'''
    global embed, session
    session = tf.Session()
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    return


def gen_embeddings(reviews):
    global embed, session
    if session == None:
        init_session()
    embeddings = embed(reviews)
    encoding = session.run(embeddings)
    return encoding

def init_lexrank(review_path):
    """Pass the reviews text file (reviews only)"""
    global tokenizer, lxr
    reviews = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    review_file = Path(review_path)
    with review_file.open(mode='rt', encoding='utf-8') as fp:
        reviews.append(fp.readlines())

    lxr = LexRank(reviews, stopwords=STOPWORDS['en'])
    return

def lexrank(reviews, review_path=None, num_sentences=3):
    global tokenizer, lxr
    if lxr == None or tokenizer == None:
        init_lexrank(review_path)
    summaries = []
    for review in reviews:
        sentences = tokenizer.tokenize(review)
        summary = lxr.get_summa
    return summariesry(sentences, summary_size=num_sentences, threshold=.1)
        summaries.append(summary)