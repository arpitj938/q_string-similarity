import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
from nltk.util import ngrams
from ngram import NGram


def jaccard_distance(a,b):
	# print a, b
	inter_len = float(len(list(a.intersection(b))))
	union_len = float(len(list(a.union(b))))
	return inter_len/union_len

df = pd.read_csv('/home/arpit/learning/machine learning/dataset/test.csv')
# print df.tail(),df.shape[0],df.shape[1]
#check for null element

print df.isnull().any()
# df1 = pd.DataFrame()
# # df1['qid'] = df['qid1']
# df1['questions']= df['question1']
# df2 = pd.DataFrame()
# # df2['qid'] = df['qid2']
# df2['questions'] = df['question2']
# frame = [df1,df2]
# #add columns to get all data under single column
# df = pd.concat(frame)
# print df.tail(),df.shape[0]
#drop null values
df=df.fillna('_')
print df.isnull().any().count()
# string_sim = string_sim.toarray()

string_sim = df.as_matrix()
print string_sim.shape
print string_sim[0][1]
duplicated = {}
duplicated['test_id'] = []
duplicated['is_duplicate'] = []
count =0
tfidf = TfidfVectorizer()
n = NGram()
print  datetime.datetime.now()
start =  datetime.datetime.now()
for o in string_sim:
	string = []
	count = count +1
	string.append(str(o[1]))
	string.append(str(o[2]))
	ngram_array = [ list(n.split(s)) for s in string]
	jaccard_distance_ans = [jaccard_distance(NGram(ngram_array[0]),NGram(s)) for s in ngram_array][1]
	# print string
	if (count == 10000):
		print 10000
		print  datetime.datetime.now()
		print  (datetime.datetime.now() - start)
		count =0
	duplicated['test_id'].append(o[0])
	try:
		y = tfidf.fit_transform(string)
		y_array = y.toarray() 
		sim_arry1 = cosine_similarity(y_array)
		duplicated['is_duplicate'].append((sim_arry1[0][1]+jaccard_distance_ans)/2)
	except Exception,e:
		duplicated['is_duplicate'].append((0+jaccard_distance_ans)/2)
		print e
		print o[1],o[2]
# print df.isnull().any()
# print string_sim[:10]
#make a tfidf matrix
# print duplicated
df1 = pd.DataFrame(data = duplicated, columns=['test_id','is_duplicate'])
print df1.head()
df1.to_csv('result.csv', sep=',', header=True, index=False)
