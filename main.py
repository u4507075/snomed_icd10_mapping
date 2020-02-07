# import libraries
import numpy as np
import pandas as pd
import re
from preprocess.compare_text import compare

from nltk.corpus import stopwords

# You will have to download the set of stop words the first time
# import nltk
# nltk.download('stopwords')
stop_words = stopwords.words('english')

# secret directory to the dataset
path = "../secret/data/"
def clean_data():
	# start your code here
	discharge_summary = pd.read_csv(path+'discharge_summary.csv',index_col=0)
	# discharge_summary = discharge_summary.head(10)

	discharge_summary['sum_note'] = discharge_summary['sum_note'].apply(lambda x: re.sub('&lt;br/&gt;',' ',str(x).lower()))
	discharge_summary['sum_note'] = discharge_summary['sum_note'].apply(lambda x: re.sub('&gt;',' ',str(x).lower()))
	discharge_summary['sum_note'] = discharge_summary['sum_note'].apply(lambda x: re.sub('<.*?>',' ',str(x).lower()))
	discharge_summary['sum_note'] = discharge_summary['sum_note'].apply(lambda x: re.sub('[^a-zA-Z0-9 ]','',str(x).lower()))
	discharge_summary['sum_note'] = discharge_summary['sum_note'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
	discharge_summary['sum_note'] = discharge_summary['sum_note'].apply(lambda x: re.sub('  +',' ',str(x).lower()))
	discharge_summary.to_csv(path+'cleaned_data.csv')

def map_icd10():
	discharge_summary = pd.read_csv(path+'cleaned_data.csv', index_col= 0)
	snomed_finding = pd.read_csv(path + 'snomed/disorder.csv', index_col=0)
	concept_id = pd.read_csv(path + 'snomed/conceptid_to_icd10.csv', index_col=0)
	icd10_dict = dict(zip(concept_id['concept_id'],concept_id['icd10']))
	for index,row in discharge_summary.head(10).iterrows():
		print(row['sum_note'])
		compared_data = compare(row['sum_note'],snomed_finding.copy())
		discharge_summary['sum_note'] = discharge_summary['sum_note'].apply(lambda x: re.sub('  +', ' ', str(x).lower()))
		compared_data['icd10'] = compared_data['concept_id'].apply(lambda x: icd10_dict[x] if x in icd10_dict else '')
		# select unempty 'icd10' column
		compared_data = compared_data[compared_data['icd10']!='']
		print(compared_data)
		#for i,r in compared_data.head(100).iterrows():
		#	print(str(r['id'])+' '+r['term'])

# STEP 1
# clean_data()
# STEP 2
map_icd10()
