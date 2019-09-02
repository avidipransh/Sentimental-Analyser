import tweepy
from tweepy import OAuthHandler 
tweets = []
topic =input("Enter topic\n")
consumer_key = "4HZz7KabLQiMfsfn8VsBjdp6z"
consumer_secret = "VWj3sFWcKkVq8E2Gdr8fNsoGMeZCjhVwCXE8BTEWpy0LLOcxWI"
access_token = "716622864741449728-bI6HI3YwTuCpZPZSRPqA7m00w7BDNTV"
access_token_secret = "7QeDWNjMTnhBJt4mlS8HTMLHuwHOkSu5LJx8WKI7uhCvy"
try:
    import pdb
    pdb.set_trace()
    auth = OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_token , access_token_secret)
    api = tweepy.API(auth)
    fetched_tweets = api.search(q = topic , count = 200)
    count = 0
    
    for tweet in fetched_tweets:
        tweets.append(tweet.text)
        print(tweet.text)
        count += 1          

except Exception as e:
      print(e)