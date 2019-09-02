#!/usr/bin/env python3

#dataset = train.csv
#rep name = text.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import tweepy
from tweepy import OAuthHandler
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


class SentimentalAnalyser:
  def __init__(self):
    self.ps =PorterStemmer()    
    self.dataset = pd.read_csv("train.csv")
    self.dataset['tweet'][0]
    self.clean_tweets = []

  def cleanTweets(self):
    for i in range(len(self.dataset)):
      #removing the @user as they don't impact the analyser
      tweet = re.sub('@[\w]*', ' ', self.dataset['tweet'][i])
      #removing all the emojis. Only taking the alphabets and numeric numbers into considerations
      tweet = re.sub('[^a-zA-Z#]', ' ', tweet)
      tweet = tweet.lower()
      tweet = tweet.split()
      
      tweet = [self.ps.stem(token) for token in tweet if not token in stopwords.words('english')]
      tweet = ' '.join(tweet)
      self.clean_tweets.append(tweet)

  def cleanTestTweets(self, uncleaned_tweets):
    test_tweets = []
    for tweets in uncleaned_tweets:
      tweet = re.sub('@[\w]*', ' ', tweets)
      tweet = re.sub('[^a-zA-Z#]', ' ', tweet)
      tweet = tweet.lower()
      tweet = tweet.split()
      tweet = [self.ps.stem(token) for token in tweet if not token in stopwords.words('english')]
      tweet = ' '.join(tweet)
      test_tweets.append(tweet)
    return test_tweets

  def buildModel(self):
    cv = CountVectorizer(max_features = 3000)
    X = cv.fit_transform(self.clean_tweets)
    X = X.toarray()
    y = self.dataset['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    gnb.predict(X_test)    
    print(gnb.score(X_test, y_test))
    return gnb

  def getTweets(self , topic):
    tweets = []
    consumer_key = "##########3"
    consumer_secret = "##########"
    access_token = "#################-########"
    access_token_secret = "##############################"
    try:
        auth = OAuthHandler(consumer_key,consumer_secret)
        auth.set_access_token(access_token , access_token_secret)
        api = tweepy.API(auth)
        fetched_tweets = api.search(q = topic , count = 200)
        count = 0
        
        for tweet in fetched_tweets:
            tweets.append(tweet.text)
            count += 1          
        return tweets

    except Exception as e:
      print(e)

#print(cv.get_feature_names())

def main():
  global np
  ob = SentimentalAnalyser()
  ob.cleanTweets()
  model = ob.buildModel()

  topic = input("Enter topic for sentimental analysis.\n")

  tweets = ob.getTweets(topic)

  test_tweets = ob.cleanTestTweets(tweets)
  cv = CountVectorizer(max_features = 3000)
  X = cv.fit_transform(test_tweets)
  import pdb
  pdb.set_trace()
  results = model.predict(X.toarray())

  print(results)



  


if __name__ == "__main__":
  main()

