import json
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import nltk
from nltk import *
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import sys
import codecs
import string
import random
import re

import requests
import urllib
import bs4
from bs4 import BeautifulSoup
from readability.readability import Document # https://github.com/buriy/python-readability. Tried Goose, Newspaper (python libraries on Github). Bad results.
from http.cookiejar import CookieJar #

import pycrunchbase
cb = pycrunchbase.CrunchBase('662e263576fe3e4ea5991edfbcfb9883')

def preprocessTweet(tweet): # preprocess tweets
    tweet = re.sub(r'\d+', '', str(tweet)) # remove numbers
    tweet = tweet.lower() # convert to lower case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # convert www.* or https?://* to URL
    tweet = re.sub('@[^\s]+','AT_USER',tweet) # convert @username to AT_USER
    tweet = re.sub('[\s]+', ' ', tweet) # remove additional white spaces
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # replace #word (hashtags) with word
    tweet = tweet.strip('\'"') # trim
    return tweet

def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)

def getStopWordList(stopWordListFileName): # get stopword list
    sw = [] # create list of stopwords

    nltk_stopwords = stopwords.words('english')
    for w in nltk_stopwords:
        sw.append(w)
    sw.append('AT_USER') # special stopwords from preprocessTweet function
    sw.append('URL')

    fp = open(stopWordListFileName, 'r') # load in any more custom stopwords from file
    line = fp.readline()
    while line:
        word = line.strip()
        sw.append(word)
        line = fp.readline()
    fp.close()
    return sw


stopword_list = getStopWordList('stopwords.txt')

def getFeatureVector(tweet):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:
        w = replaceTwoOrMore(w) # replace two or more with two occurrences
        w = w.strip('\'"?,.') # strip punctuation
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w) # check if the word starts with an alphabet
        if(w in stopword_list or val is None): # ignore if it is a stop word
            continue
        else:
            featureVector.append(w.lower())
    return featureVector

airline_train = pd.read_csv("Twitter-Get-Old-Tweets-Scraper/training.1600000.processed.noemoticon.csv",
                          header=None, encoding="ISO-8859-1") #latin1 encoding
airline_train.columns = ["sentiment", "id", "date", "query", "user", "text"]
# airline_train = airline_train.sample(frac=0.05, replace=True, random_state=1)
# airline_train = airline_train.reset_index(drop=True)
airline_tweets = []
airline_featurelist = []
for i in range(len(airline_train)):
    sentiment = airline_train['sentiment'][i]
    tweet = airline_train['text'][i]
    preprocessedTweet = preprocessTweet(tweet)
    featureVector = getFeatureVector(preprocessedTweet)
    airline_featurelist.extend(featureVector)
    airline_tweets.append((featureVector, sentiment))

def extract_features(tweet): # Determine if tweet contains a feature word
    tweet_words = set(tweet)
    features = {}
    for word in airline_featurelist:
        features['contains(%s)' % word] = (word in tweet_words)
    return features

airline_featurelist = list(set(airline_featurelist)) # remove airline_featurelist duplicates
airline_training_set = nltk.classify.util.apply_features(extract_features, airline_tweets)
airline_NBClassifier = nltk.NaiveBayesClassifier.train(airline_training_set)
from sklearn.externals import joblib
joblib.dump(airline_NBClassifier, 'csv_export/NBClassifier.pkl')
