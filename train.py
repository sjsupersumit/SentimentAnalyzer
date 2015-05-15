__author__ = 'sumit.jha'

import pandas as pd
from bs4 import BeautifulSoup
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier



train = pd.read_csv("C:\Users\sumit.jha\Downloads\labeledTrainData.tsv\labeledTrainData.tsv" , header=0, delimiter="\t", quoting=3 )

num_reviews = train["review"].size

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review).get_text()
    letter_only = re.sub("[^a-zA-Z]"," ", review_text )
    words = letter_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    pt = PorterStemmer()
    Lt = LancasterStemmer()
    meaningful_words = [Lt.stem(w) for  w in meaningful_words ]
    return (" ".join(meaningful_words))

# clean_review = review_to_words(train["review"][0])
# print clean_review

def remove_non_ascii(text):

    return ''.join(i for i in text if ord(i)<128)

print "cleaning and parsing the training set movie review.......\n"

clean_train_reviews = []
for i in xrange(0, num_reviews):
    if((i+1)%1000 == 0):
        print "Review %d of %d\n" %(i+1, num_reviews)
    clean_train_reviews.append(review_to_words(train["review"][i]))

print "Creating the bag of words......\n"

vectorizer = CountVectorizer(analyzer="word",tokenizer= None, stop_words=None, max_features=3000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)

train_data_features = train_data_features.toarray()

print "Model is trained now......\n"

# vocab = vectorizer.get_feature_names()
# print vocab

#dist = np.sum(train_data_features, axis=0)
# print "Displaying frequency of each vocab words.....\n"
#
# for tag, count in zip(vocab, dist):
#     print count , tag


print "Using RandomForestClassifier as classifier....\n..\n"

forest = RandomForestClassifier(n_estimators=100, n_jobs=1)

print "Initializing Random Forest Classifier with 100  tress....\n"

print "Fitting the forest to training set, using bag of words as features and sentiment label as response variable\n "

print "This may take few mins to run....\n"

forest = forest.fit(train_data_features, train["sentiment"])

print "All Set!!! Reading test Tweets  ....... \n"

test = pd.read_csv("C:\Users\sumit.jha\Desktop\piku_tweets.csv", header=0, delimiter=",")

#print test.columns.values

clean_test_reviews = []

print "Cleaning and parsing the test set movie reviews...\n"
# for i in xrange(0,num_reviews):
#     if( (i+1) % 1000 == 0 ):
#         print "Review %d of %d\n" % (i+1, num_reviews)
#     clean_review = review_to_words( test["review"][i] )
#     clean_rev = remove_non_ascii(clean_review)
#     clean_test_reviews.append( clean_rev )

fp1 = open("C:\Users\sumit.jha\Desktop\piku_tweets.csv", 'r')

line1 = fp1.readline()

while line1:

        clean_rev = review_to_words(line1)
        clean_rev = remove_non_ascii(clean_rev)
        clean_test_reviews.append(clean_rev)
        line1 = fp1.readline()


fp1.close()
# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column

output = pd.DataFrame(  data={"id":"##", "sentiment":result} )

print "ALL DONE..........PRINTING OUTPUT TO FILE>..."
# Use pandas to write the comma-separated output file
output.to_csv( "C:\Users\sumit.jha\Desktop\Output_Bag_of_Words_model.csv", index=False, quoting=3 )