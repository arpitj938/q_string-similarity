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


stops = set(stopwords.words("english"))                 
def jaccard_distance(a,b):
	# print a, b
	try:
		inter_len = float(len(list(a.intersection(b))))
		union_len = float(len(list(a.union(b))))
		return inter_len/union_len
	except Exception,e:
		print e
		print a,b
		return 0

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

def tokernize(a,b):
	a = a.lower()
	b = b.lower()
	# print a
	# print b
	a = re.sub('[():?.,]',"",a)
	b = re.sub('[?.,]',"",b)
	# b = b.encode('utf-8').translate(None,'?.,')
	# print a
	a = word_tokenize(a)
	b = word_tokenize(b)
	a = [w for w in a if not w in stops]
	b = [w for w in b if not w in stops]
	# print a
	# print b
	a = set(a)
	b = set(b)
	# print a
	# print b
	jaccard_distance_ans = jaccard_distance(a,b)
	return jaccard_distance_ans
# string =['WHAT is 2+2?.','WHAT is 2+2? sdhfh.']
# tokernize(unicode(string[0]),unicode(string[1]))

df = pd.read_csv('/home/arpit/learning/machine learning/quora_dataset/train.csv')
print df.head()
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
count =0
i=1
tfidf = TfidfVectorizer(analyzer='word',stop_words = 'english',lowercase=True)
n = NGram()
print  datetime.datetime.now()
start =  datetime.datetime.now()
for o in string_sim:
	string = []
	string.append(str(o[3]))
	string.append(str(o[4]))
	# ngram_array = [list(n.split(s)) for s in string]
	jaccard_distance_ans = tokernize(string[0].decode('utf-8'),string[1].decode('utf-8'))
	# print jaccard_distance_ans
	# print o[1],o[2]
	# print string
	# jaccard_distance_ans =0
	if (count == 10000):
		print i*10000
		i=i+1
		print  datetime.datetime.now()
		print  (datetime.datetime.now() - start)
		save_csv(duplicated)
		duplicated = {}
		duplicated['test_id'] = []
		duplicated['is_duplicate']= []
		count =0
	duplicated['test_id'].append(o[0])
	ans = [0]*3
	try:
		y = tfidf.fit_transform(string)
		y_array = y.toarray() 
		sim_arry1 = cosine_similarity(y_array)
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
		result =1
	else:
		result =0
	duplicated['is_duplicate'].append(result)
	print o[3]
	print o[4]
	print ans[0],ans[1],result , o[5]	
save_csv(duplicated)