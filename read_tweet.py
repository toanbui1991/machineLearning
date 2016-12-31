import csv,sys
import nltk, nltk.classify.naivebayes
from filtering_tweet_word import getStopWordList,getFeatureVector,processTweet


inpTweets = csv.reader(open('data/New_Traning_set.csv', 'rb'), delimiter=',', quotechar='|')
featureList = []
tweets = []

for row in inpTweets:
	sentiment = row[0]
	tweet = row[1]
	processedTweet = processTweet(tweet)
	featureVector = getFeatureVector(processedTweet)
	featureList.extend(featureVector)
	tweets.append((featureVector,sentiment));
	#The sentiment string is also called label
#end loop


# Remove featureList duplicates
featureList = list(set(featureList))

# Extract feature vector for all tweets in one shote
#start extract_features 
#inputs = 
#extract = extract_features()

def extract_features(tweet):
	tweet_words = set(tweet)
	features = {}
	
	for word in featureList:
		features['contains(%s)' % word] = (word in tweet_words)
	return features
#end


#we can apply the features to our classifier using the method apply_features. 
#We pass the feature extractor along with the tweets list defined above



training_set = nltk.classify.util.apply_features(extract_features, tweets)
# Train the classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)


# Test the classifier
voteHillary = 0
voteTrump = 0
i=0
testFile = csv.reader(open('data/New_Testing_set.csv', 'rb'))
for row in testFile:
	i = i+1
	print i
	testTweet = row[0]
	processedTestTweet = processTweet(testTweet)
	test_set = getFeatureVector(processedTestTweet)
	#print 'The result is '+ NBClassifier.classify(extract_features(test_set))+'\n'
	if (NBClassifier.classify(extract_features(test_set)) == 'PositiveHillary'):
		voteHillary = voteHillary+1
	if (NBClassifier.classify(extract_features(test_set)) == 'PositiveTrump'):
		voteTrump = voteTrump+1
print "Hillary = " + str(format((voteHillary*100.0/8070),'.2f') ) + '%   Trump = ' + str(format((voteTrump*100.0/8070),'.2f')) + '% '

try:
	person = input('Enter your name: ')
	print('Hello ', person)
except(EOFError):
	print('errer')



