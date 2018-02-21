import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
from nltk.util import ngrams
from ngram import NGram
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

stops = set(stopwords.words("english"))                 
print stops
def save_csv(data):
	df1 = pd.read_csv('result1.csv')
	df2 = pd.DataFrame(data=data,columns=['test_id','is_duplicate'])
	frame = [df1,df2]
	df = pd.concat(frame)
	df.to_csv('result1.csv',header=True, index=False)
	print "saved"

def create_csv():
	df = pd.DataFrame(columns=['test_id','is_duplicate'])
	df.to_csv('result1.csv',header=True, index=False)
	print "csv created"

def tokernize_removestop(a):
	a = a.lower()
	# print a
	a = re.sub('[():?.,]',"",a)
	# print a
	a = word_tokenize(a)
	c= []
	for w in a:
		c.append(lemmatizer.lemmatize(w))
	a =  c
	a = [w for w in a if not w in stops]
	# print "token",a
	return a
	
def add_pos_tag(a):
	return nltk.pos_tag(a)

def jaccard_distance(a,b):
	# print a, b
	a = tokernize_removestop(a)
	b = tokernize_removestop(b)
	# a = add_pos_tag(a)
	# b = add_pos_tag(b)
	a = set(a)
	b = set(b)
	try:
		inter_len = float(len(list(a.intersection(b))))
		union_len = float(len(list(a.union(b))))
		return inter_len/union_len
	except Exception,e:
		print e
		print a,b
		return 0


def cosine_sim(string):
	for o in xrange(0,2):
		string[o] = " ".join(tokernize_removestop(string[o]))  
		# print "cosine ",string[o]
	tfidf = TfidfVectorizer(analyzer='word',stop_words = 'english',lowercase=True)
	y = tfidf.fit_transform(string)
	y_array = y.toarray() 
	return cosine_similarity(y_array)


# string =["Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?",
# "I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me?"
# ]
# print jaccard_distance(string[0].decode('utf-8'),string[1].decode('utf-8'))
# print cosine_sim(string)
# tokernize(unicode(string[0]),unicode(string[1]))
# str_input = raw_input('Enter the string')
df = pd.read_csv('/home/arpit/learning/machine learning/quora_dataset/test.csv')
# print df.head()
print df.is_duplicate.unique()
# a= input()
create_csv()
df=df.fillna('_')
print df.isnull().any()
string_sim = df.as_matrix()
print string_sim.shape
print string_sim[0][1]
duplicated = {}
duplicated['test_id'] = []
duplicated['is_duplicate'] = []
count = 0
i=1
print  datetime.datetime.now()
start =  datetime.datetime.now()
for o in string_sim[10:30]:
	string = []
	string.append(str(o[1]))
	string.append(str(o[2]))
	# print "string",string
	jaccard_distance_ans = jaccard_distance(string[0].decode('utf-8'),string[1].decode('utf-8'))
	# q = input()
	if (count == 10000):
		print i*10000
		i=i+1
		print  datetime.datetime.now()
		print  (datetime.datetime.now() - start)
		save_csv(duplicated)
		duplicated = {}
		duplicated['test_id'] = []
		duplicated['is_duplicate']= []
		count = 0
	duplicated['test_id'].append(o[0])
	ans = [0]*3
	try:
		sim_arry1 = cosine_sim(string)
		ans[0] = sim_arry1[0][1]
		ans[1] = jaccard_distance_ans
		ans[2] = max(sim_arry1[0][1],jaccard_distance_ans)
	except Exception,e:
		ans[0] = 0
		ans[1] = jaccard_distance_ans
		ans[2] = jaccard_distance_ans
		print e
		print o[1],o[2]
	count = count +1
	if (ans[2]>=0.75):
		result = 1
	else:
		result = 0
	duplicated['is_duplicate'].append(result)
	# print o[3]
	# print o[4]
	# print ans[0],ans[1],result , o[5]	
save_csv(duplicated)