# import libraries
import numpy as np
import pandas as pd
import re
from preprocess.compare_text import compare
from pathlib import Path
import os
from nltk.corpus import stopwords
from preprocess.compare_text import algorithm_validity
from preprocess.compare_text import tf_idf

# You will have to download the set of stop words the first time
# import nltk
# nltk.download('stopwords')
stop_words = stopwords.words('english')

# secret directory to the dataset
path = "../secret/data/"

# Hello world
def clean_data():
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
#algorithm_validity()
tf_idf()

