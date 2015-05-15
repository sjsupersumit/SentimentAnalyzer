import tweepy
import csv

#FILL IN YOUR KEYS
access_token = ""
access_token_secret = ""
consumer_key = ""
consumer_secret = ""

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Open/Create a file to append data
csvFile = open('tweets.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)


for tweet in tweepy.Cursor(api.search,
    q="healthkart",
    since="2015-05-08",
    until="2015-05-14",
    lang="en").items():


    print tweet.created_at, tweet.text
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])