#import regex
import re
import nltk
#initialze stopWords
stopWords = []
#start process tweet
def processTweet(tweet):
	#Convert to Lower case
	tweet = tweet.lower()
	#Convert www. * or https?://* to URL
	tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
	#Convert @username to AT_USER
	tweet = re.sub('@[^\s]+','AT_USER',tweet)
	#Remove additional white spaces
	tweet = re.sub('[\s]+', ' ', tweet)
	#Replace #word with word
	tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
	#trim
	tweet = tweet.strip('\'"')
	return tweet

 #end

#start replace TwoOrore
def replaceTwoOrMore(s):
	#look for 2 or more repetitions of character and replace with the character itself
	pattern = re.compile(r"(.)\1{1,}",re.DOTALL)
	return pattern.sub(r"\1\1", s)
#end

#start getStopWordList
def getStopWordList():
	#read the stopwords file and build a list
	stopWords = []
	stopWords.append('AT_USER')
	stopWords.append('URL')
	fp = open('data/stopwords.txt', 'r')
	line = fp.readline()
	while line:
		word = line.strip()
		stopWords.append(word)
		line = fp.readline()
	fp.close()
	return stopWords
#end

def getFeatureVector(tweet):
	stopWords1= getStopWordList()
	featureVector = []
	#split tweet into words
	words = tweet.split()
	for w in words:
		#replace two or more with two occurrences
		w = replaceTwoOrMore(w)
		#strip punctuation
		w = w.strip('\'",.!?')	
		#check if the word stats with an alphabet
		val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
		#ignore if it is a stop word
		if(w in stopWords1 or val is None):
			continue
		else:
			#print w
			featureVector.append(w.lower())
	return featureVector
#end