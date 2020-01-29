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

#hello
#start your code here
discharge_summary = pd.read_csv(path+'discharge_summary.csv',index_col=0)
discharge_summary = discharge_summary.head(10)

discharge_summary['sum_note'] = discharge_summary['sum_note'].apply(lambda x: re.sub('&lt;br/&gt;',' ',str(x).lower()))
discharge_summary['sum_note'] = discharge_summary['sum_note'].apply(lambda x: re.sub('&gt;',' ',str(x).lower()))
discharge_summary['sum_note'] = discharge_summary['sum_note'].apply(lambda x: re.sub('<.*?>',' ',str(x).lower()))
discharge_summary['sum_note'] = discharge_summary['sum_note'].apply(lambda x: re.sub('[^a-zA-Z0-9 ]','',str(x).lower()))
discharge_summary['sum_note'] = discharge_summary['sum_note'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
discharge_summary['sum_note'] = discharge_summary['sum_note'].apply(lambda x: re.sub('  +',' ',str(x).lower()))

for index,row in discharge_summary.head(10).iterrows():
	print(row['sum_note'])
	compared_data = pd.read_csv(path+'snomed/finding.csv',index_col=0)
	compare(row['sum_note'],compared_data.copy())
