import pandas as pd

df = pd.read_csv('result1.csv')
df1 = pd.read_csv('/home/arpit/learning/machine learning/quora_dataset/train.csv')

orginal = df1.shape[0]
result = df.shape[0]
print orginal
print result
df = df.as_matrix()
df1 = df1.as_matrix()
count =0
for o in xrange(1,len(df)):
	if(df[o][1]==df1[o][5]):
		count = count+1
print "percentage", ((float(len(df)) - float(count))/float(len(df)))*100