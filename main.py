# import libraries
import numpy as np
import pandas as pd
import re
from preprocess.compare_text import compare
from pathlib import Path
import os
from nltk.corpus import stopwords
from fuzzywuzzy import process
import itertools
import random
from itertools import combinations
from preprocess.compare_text import algorithm_validity
from preprocess.compare_text import tf_idf

# You will have to download the set of stop words the first time
# import nltk
# nltk.download('stopwords')
stop_words = stopwords.words('english')

# secret directory to the dataset
path = "../secret/data/"

# Hello world
def start_data():
	# start your code here
	discharge_summary = pd.read_csv(path+'discharge_summary.csv',index_col=0)
	# discharge_summary = discharge_summary.head(10)

def clean_data():
	#start your code here
	discharge_summary = pd.read_csv(path+'discharge_summary.csv',index_col=0)
	#discharge_summary = discharge_summary.head(10)

	discharge_summary['sum_note'] = discharge_summary['sum_note'].apply(lambda x: re.sub('&lt;br/&gt;',' ',str(x).lower()))
	discharge_summary['sum_note'] = discharge_summary['sum_note'].apply(lambda x: re.sub('&gt;',' ',str(x).lower()))
	discharge_summary['sum_note'] = discharge_summary['sum_note'].apply(lambda x: re.sub('<.*?>',' ',str(x).lower()))
	discharge_summary['sum_note'] = discharge_summary['sum_note'].apply(lambda x: re.sub('[^a-zA-Z0-9 ]','',str(x).lower()))
	discharge_summary['sum_note'] = discharge_summary['sum_note'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
	discharge_summary['sum_note'] = discharge_summary['sum_note'].apply(lambda x: re.sub('  +',' ',str(x).lower()))
	discharge_summary.to_csv(path+'snomed/discharge_clean.csv')

def convert(x):
	x = str(x).split('.')
	if len(x) > 1:
		return x[0]+x[1][:1]
	else:
		return x[0]

def get_related_icd10(text,compared_data,type,icd10_dict,icd10_name_dict):
	data = compare(text,compared_data.copy())
	data['icd10'] = data['concept_id'].apply(lambda x: icd10_dict[x] if x in icd10_dict else '')
	data['icd10_name'] = data['icd10'].apply(lambda x: icd10_name_dict[convert(x)] if convert(x) in icd10_name_dict else '')
	data = data[(data['icd10']!='') & (data['icd10'].notnull())]
	data['type'] = type
	return data

def map_icd10():
	discharge_summary = pd.read_csv(path+'snomed/discharge_clean.csv',index_col=0)
	icd10 = pd.read_csv(path+'snomed/conceptid_to_icd10.csv',index_col=0)
	icd10_dict = dict(zip(icd10.concept_id, icd10.icd10))
	icd10_name = icd10 = pd.read_csv(path+'snomed/icd10.csv',index_col=0)
	icd10_name_dict = dict(zip(icd10_name.code, icd10_name.cdesc))
	finding = pd.read_csv(path+'snomed/finding.csv',index_col=0)
	disorder = pd.read_csv(path+'snomed/disorder.csv',index_col=0)
	abnormality = pd.read_csv(path+'snomed/abnormality.csv',index_col=0)
	procedure = pd.read_csv(path+'snomed/procedure.csv',index_col=0)

	p = path+'result_100.csv'
	file = Path(p)
	if file.is_file():
		os.remove(p)

	for index,row in discharge_summary.iterrows():
		print(row['sum_note'])
		df = []
		df.append(get_related_icd10(row['sum_note'],finding.copy(),'finding',icd10_dict,icd10_name_dict))
		df.append(get_related_icd10(row['sum_note'],disorder.copy(),'disorder',icd10_dict,icd10_name_dict))
		df.append(get_related_icd10(row['sum_note'],abnormality.copy(),'abnormality',icd10_dict,icd10_name_dict))
		df.append(get_related_icd10(row['sum_note'],procedure.copy(),'procedure',icd10_dict,icd10_name_dict))
		result = pd.concat(df)
		result['sum_note'] = row['sum_note']
		result = result.sort_values(by='similarity', ascending=False)
		if len(result)>0:
			#print(row['sum_note'])
			result = result.head(100)
			result['id'] = index
			print(result[['id','term','icd10_name','type','similarity']])
			file = Path(p)
			if file.is_file():
				with open(p, 'a') as f:
					result.to_csv(f, header=False)
			else:
				result.to_csv(p)
		else:
			print('not found')

#clean_data()
#map_icd10()
#tf_idf()
#algorithm_validity()

def distance():
	df = pd.read_csv(path+'result_100.csv', index_col= 0)
	#print (df)
	data = df[['id','term', 'sum_note', 'icd10', 'icd10_name']]
	data = data.head(10000)
	data = data.reset_index()
	data['min_distance'] = 0
	data['dscaled'] = 10

	for index,y in data.iterrows():
		sn = str(y['sum_note'])
		tm = str(y['term'])
		tm = tm.lower()
		sn = sn.split()
		tm = tm.split()


		#print(tm)
		#print(sn)
		tp = []
		for t in tm:
			#print(t)
			result = process.extract(t,sn)
			#print(result)
			filter_result = [t for t in result if t[1] > 80]
			#print(filter_result)
			pos = []
			for w in filter_result:
				#print(w)
				values = np.array(sn)
				ii = np.where(values == w[0])
				#print(ii)
				#print(list(ii))
				for l in list(ii):
					pos.append(l.tolist())
			#print(pos)
			flat_pos = [y for x in pos for y in x]
			#print(flat_pos)
			flat_pos = np.unique(np.array(flat_pos)).tolist()
			#print (flat_pos)
			tp.append(flat_pos)
		#print (tp)
		sumst = []
		if len(tp) > 1:
			for x in itertools.product(*tp):
				#print(list(x))
				finalist = list(itertools.combinations(x, 2))
				#print(finalist)
				subtract = [abs(i[0] - i[1]) for i in finalist]
				#print(subtract)
				sumst.append(sum(subtract))
		elif len(tp) == 1:
			finalist = list(itertools.combinations(tp[0], 2))
			# print(finalist)
			subtract = [abs(i[0] - i[1]) for i in finalist]
			# print(subtract)
			sumst.append(sum(subtract))

		min_distance = 0
		if len(sumst)>0:
			min_distance = min(sumst)
			#print(min(sumst))
		data.at[index, 'min_distance'] = min_distance
		if len(tm) > 1:
			data.at[index, 'dscaled'] = min_distance/len(tm)
		if index == 1062:
			print (tm)
			print(sumst)
			print (min_distance)
			print(data[data.index == 1062])



		#return min(sumst)
	print(data[data.index==1062])
	data.to_csv(path+'distance.csv')

#distance()
def save_file(df,p):
	file = Path(p)
	if file.is_file():
		with open(p, 'a', encoding="utf-8") as f:
			df.to_csv(f, header=False)
	else:
		df.to_csv(p)
def remove_file(p):
	file = Path(p)
	if file.is_file():
		os.remove(p)
def machinelearn():
	df = pd.read_csv(path+'snomed/discharge_clean.csv', index_col= 0)
	#print (df)
	data = df[['sum_note','dx1_code']]
	#data['icd10'] = data['icd10'].apply(lambda x: convert(x))
	#data = data.head(10)
	data = data.reset_index()
	remove_file(path+'bag.csv')
	for index, y in data.iterrows():
		sn = str(y['sum_note'])
		sn = sn.split(' ')
		dx = str(y['dx1_code'])
		for j in range (3):
			sn1 = [sn[i:i+(j+1)] for i in range(len(sn) - (j+1))]
			hs = [' '.join(k) for k in sn1]
			kw = [[w,dx] for w in hs]
			bagframe = pd.DataFrame(kw, columns = ['keywords', 'icd10'])
			print (bagframe)
			save_file(bagframe, path+'bag.csv')
	df = pd.read_csv(path+'bag.csv', index_col = 0)
	df.to_csv(path+'bag.csv')
def get_row(col,val):
	dfs = []
	for d in pd.read_csv(path+'bag.csv', index_col = 0, chunksize = 10000):
		d = d[d[col] == val]
		dfs.append(d)
	x = pd.concat(dfs).sample()
	if col == 'keywords':
		return [[x['keywords'].iloc[0], x['keywords'].iloc[0]], [x['icd10'].iloc[0], x['keywords'].iloc[0]]], 'icd10', x['icd10'].iloc[0]
	else:
		return [[x['icd10'].iloc[0], x['icd10'].iloc[0]], [x['keywords'].iloc[0], x['icd10'].iloc[0]]], 'keywords', x['keywords'].iloc[0]
#machinelearn()
#n = 3944
#for df in pd.read_csv(path+'bag.csv', index_col = 0, chunksize = 10000):
#	n = n+1
#print (n)
data = []
#n = 0
#i = random.randrange(0, 3945, 1)
#i = 0
#row = None
#for df in pd.read_csv(path+'bag.csv', index_col = 0, chunksize = 10000):
#	#random , mai tong loop
#	if n == i:
#		row = df.sample()
#		break
#	n = n + 1
#data.append([row["keywords"].iloc[0], row["keywords"].iloc[0]])
#data.append([row["icd10"].iloc[0], row["keywords"].iloc[0]])
#print (data)
#row2 = get_row('icd10', row["icd10"].iloc[0])

#col = 'icd10'
#val = row["icd10"].iloc[0]
#row = None
#for x in range(10):
#	row, col, val = get_row(col,val)
#	data.append(row)
#	print(data)



filelist = [ f for f in os.listdir(path+'keywords/') if f.endswith(".csv") ]
for f in filelist:
    os.remove(os.path.join(path+'keywords/', f))
filelist = [ f for f in os.listdir(path+'icd10/') if f.endswith(".csv") ]
for f in filelist:
    os.remove(os.path.join(path+'icd10/', f))
for df in pd.read_csv(path+'bag.csv', index_col = 0, chunksize = 1):
	k = 'keyword_'+str(df['keywords'].iloc[0])
	i = 'icd10_'+str(df['icd10'].iloc[0])
	print (df)
	save_file(df, path+'keywords/'+str(k)+'.csv')
	save_file(df, path+'icd10/'+str(i)+'.csv')





