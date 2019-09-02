#dataset = train.csv
#rep name = text.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


dataset = pd.read_csv("train.csv")

dataset['tweet'][0]
clean_tweets = []

for i in range(31962):
  tweet = re.sub('@[\w]*', ' ', dataset['tweet'][i])
  tweet = re.sub('[^a-zA-Z#]', ' ', tweet)
  tweet = tweet.lower()
  tweet = tweet.split()
  
  tweet = [ps.stem(token) for token in tweet if not token in stopwords.words('english')]
  tweet = ' '.join(tweet)
  clean_tweets.append(tweet)
  
  
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000)
X = cv.fit_transform(clean_tweets)
X = X.toarray()
y = dataset['label'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb.predict(X)
gnb.score(X, y)


#print(cv.get_feature_names())
