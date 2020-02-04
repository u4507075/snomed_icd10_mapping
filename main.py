#import libraries
import numpy as np
import pandas as pd
import re
from preprocess.compare_text import compare

from nltk.corpus import stopwords

# You will have to download the set of stop words the first time
#import nltk
#nltk.download('stopwords')
stop_words = stopwords.words('english')

#secret directory to the dataset
path = '../secret/data/'

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

def map_icd10():
	discharge_summary = pd.read_csv(path+'snomed/discharge_clean.csv',index_col=0)
	icd10 = pd.read_csv(path+'snomed/conceptid_to_icd10.csv',index_col=0)
	icd10_dict = dict(zip(icd10.concept_id, icd10.icd10))
	icd10_name = icd10 = pd.read_csv(path+'snomed/icd10.csv',index_col=0)
	icd10_name_dict = dict(zip(icd10_name.code, icd10_name.cdesc))
	for index,row in discharge_summary.iterrows():
		#print(row['sum_note'])
		compared_data = pd.read_csv(path+'snomed/finding.csv',index_col=0)
		data = compare(row['sum_note'],compared_data.copy())
		data['icd10'] = data['concept_id'].apply(lambda x: icd10_dict[x] if x in icd10_dict else '')
		data['icd10_name'] = data['icd10'].apply(lambda x: icd10_name_dict[str(x)[:3]] if str(x)[:3] in icd10_name_dict else '')
		data = data[(data['icd10']!='') & (data['icd10'].notnull())]
		if len(data)>0:
			print(row['sum_note'])
			print(data.head(10))
		else:
			print('not found')

#clean_data()
map_icd10()
