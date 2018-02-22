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
import sys 
import math
from collections import Counter 
import string
from scipy.stats import mode
from nltk.corpus import wordnet

reload(sys)  
sys.setdefaultencoding('utf8')

lemmatizer = WordNetLemmatizer()
fail =0
avg_c =0
one_count =0
jaccard_distance_avg =[]
cosine_sim_avg =[]
stops = set(stopwords.words("english"))                 


# POS tag uses treebank_tag eg: noun plural is NNP noun singular is NNS but lemmatizer don't accept those
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''     

#Saved CSV
def save_csv(data):
	df1 = pd.read_csv('result1.csv')
	df2 = pd.DataFrame(data=data,columns=['test_id','is_duplicate'])
	frame = [df1,df2]
	df = pd.concat(frame)
	df.to_csv('result1.csv',header=True, index=False)
	print "saved"


# Made CSV
def create_csv():
	df = pd.DataFrame(columns=['test_id','is_duplicate'])
	df.to_csv('result1.csv',header=True, index=False)
	print "csv created"


# Tokenize (task done are : lower, remove puncutation , tokenize , added tag , lemmatize , remove stop word)
def tokernize_removestop(a):
	a = a.lower()
	# print a
	a = re.sub('[():?.,]',"",a)
	# print a
	a = word_tokenize(a)
	# print "token: ", a
	b = add_pos_tag(a)[0]
	# print "pos_tag_sents: ",b
	c = []
	for w in b:
		first = w[0]
		second = w[1]
		# print first,second
		try:
			c.append(lemmatizer.lemmatize(first,get_wordnet_pos(second)))
		except Exception,e:
			c.append(first)
	a =  c
	# print "lemmatize: ", a
	c = a
	a = [w for w in a if not w in stops]
	if(len(a)==0):
		a=c
	# print "token",a
	return a

#Added Tag in sentances like noun, verb etc..	
def add_pos_tag(a):
	return nltk.pos_tag_sents([a])

#jaccard Distance
def jaccard_distance(a,b):
	global fail
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
		fail = fail+1
		print e
		print a,b
		return 0

# Cosine Distance 
def cosine_sim(string):
	tfidf = TfidfVectorizer(tokenizer=tokernize_removestop)
	y = tfidf.fit_transform(string)
	y_array = (y * y.T).toarray()
	return y_array[0][1]

# string =["Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?",
# "I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me?"
# ]
# print jaccard_distance(string[0].decode('utf-8'),string[1].decode('utf-8'))
# print cosine_sim(string)

df = pd.read_csv('/home/arpit/learning/machine learning/quora_dataset/train.csv')
# print df.head()
# print df.is_duplicate.unique()
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

# #Testing Purpose Init
# o_duplicated = {}
# o_duplicated['cosine_count'] = []
# o_duplicated['jaccard_count'] = []
# p_duplicated = {}
# p_duplicated['cosine_count'] = []
# p_duplicated['jaccard_count'] = []
count = 0
i=1
print  datetime.datetime.now()
start =  datetime.datetime.now()
for o in string_sim[:20000]:
	string = []
	string.append(str(o[3]))
	string.append(str(o[4]))
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
		ans[0] = sim_arry1
		ans[1] = jaccard_distance_ans
		ans[2] = max(sim_arry1,jaccard_distance_ans)
	except Exception,e:
		ans[0] = 0
		ans[1] = jaccard_distance_ans
		ans[2] = jaccard_distance_ans
		print e
		fail = fail+1
		print o[3],o[4]
	count = count +1
	if(o[5]==1.0):
		# #postive count
		# p_duplicated['cosine_count'].append(ans[0])
		# p_duplicated['jaccard_count'].append(ans[1])
		avg_c = avg_c + 1
		jaccard_distance_avg.append(ans[1])
		cosine_sim_avg.append(ans[0])
	else:
		# # negative count
		# o_duplicated['cosine_count'].append(ans[0])
		# o_duplicated['jaccard_count'].append(ans[1])
	if (ans[2]>=0.9):
		result = 1
		one_count = one_count + 1
	else:
		result = 0
	duplicated['is_duplicate'].append(result)
	# print o[3]
	# print o[4]
	# print ans[0],ans[1],result , o[5]	
save_csv(duplicated)

# #For Checking Purpose


# df2 = pd.DataFrame(data=o_duplicated,columns=['jaccard_count','cosine_count'])
# df2.to_csv('check_o.csv',header=True, index=False)
# df2 = pd.DataFrame(data=p_duplicated,columns=['jaccard_count','cosine_count'])
# df2.to_csv('check_p.csv',header=True, index=False)
# print "saved"
# print "fail", fail
# print "jaccard_distance_avg", mode(jaccard_distance_avg)
# print "cosine_sim_avg",mode(cosine_sim_avg)
# print "1 count" , avg_c
# print "result 1 count ",one_count