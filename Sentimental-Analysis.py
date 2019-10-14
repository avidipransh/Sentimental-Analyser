#!/usr/bin/env python3
#dataset = train.csv
#rep name = text.py

a = '''
 _____            _   _                      _        _    ___              _                     
/  ___|          | | (_)                    | |      | |  / _ \            | |                    
\ `--.  ___ _ __ | |_ _ _ __ ___   ___ _ __ | |_ __ _| | / /_\ \_ __   __ _| |_   _ ___  ___ _ __ 
 `--. \/ _ \ '_ \| __| | '_ ` _ \ / _ \ '_ \| __/ _` | | |  _  | '_ \ / _` | | | | / __|/ _ \ '__|
/\__/ /  __/ | | | |_| | | | | | |  __/ | | | || (_| | | | | | | | | | (_| | | |_| \__ \  __/ |   
\____/ \___|_| |_|\__|_|_| |_| |_|\___|_| |_|\__\__,_|_| \_| |_/_| |_|\__,_|_|\__, |___/\___|_|   
                                                                               __/ |              
                                                                              |___/               
     '''

print(a)

import tqdm
import time
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

  def cleanTTweets(self,  uncleaned_tweets):
    tweets = []
    for i in range(len(uncleaned_tweets)):
      #removing the @user as they don't impact the analyser
      tweet = re.sub('@[\w]*', ' ', uncleaned_tweets[i])
      #removing all the emojis. Only taking the alphabets and numeric numbers into considerations
      tweet = re.sub('[^a-zA-Z#]', ' ', tweet)
      tweet = tweet.lower()
      tweet = tweet.split() 
      tweet = [self.ps.stem(token) for token in tweet if not token in stopwords.words('english')]
      tweet = ' '.join(tweet)
      tweets.append(tweet)
    return(tweets)

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
    print("Building Model\nStatus:")
    for i in tqdm.tqdm(range(len(X_train))):
      time.sleep(0.0001)
    print("Accuracy Of The Model = {0}%".format(gnb.score(X_test, y_test) * 100))
    return gnb

  def getTweets(self , topic):
    tweets = []
    consumer_key = "4HZz7KabLQiMfsfn8VsBjdp6z"
    consumer_secret = "VWj3sFWcKkVq8E2Gdr8fNsoGMeZCjhVwCXE8BTEWpy0LLOcxWI"
    access_token = "716622864741449728-bI6HI3YwTuCpZPZSRPqA7m00w7BDNTV"
    access_token_secret = "7QeDWNjMTnhBJt4mlS8HTMLHuwHOkSu5LJx8WKI7uhCvy"
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


  choice = int(input("Tweets Collected. Press 1 to display tweets"))

  if(choice == 1):  
    for i in range(0 , len(tweets)):  
      print(tweets[i])
  else:
    print("Invalid Input, Continuing.")

  sentence = input("Enter a sentence\n")

  s = []
  s.append(sentence)

  print("Analysing Sentence")

  cv = CountVectorizer(max_features = 3000)
  X = cv.fit_transform(s)
  X = X.toarray()
  y = model.predict(X)
  print("Polarity Of Tweet = {0}".format(y))

  for i in tqdm.tqdm(range(100)):
    time.sleep(0.01)

  print("Sentence is positive")




  


if __name__ == "__main__":
  main()

