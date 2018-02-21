import pandas as pd

df = pd.read_csv('result1.csv')
df1 = pd.read_csv('/home/arpit/learning/machine learning/quora_dataset/train.csv')

orginal = df1.shape[0]
result = df.shape[0]
print orginal
print result

orginal_ans = df1['is_duplicate'].value_counts()[1]
result_ans = df['is_duplicate'].value_counts()[1]
print "percentage", ((float(orginal_ans) - float(result_ans))/float(orginal_ans))*100