from itertools import count

from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
import re
import textdistance

path = "../secret/data/"


def text_similarity(x, y):
    return fuzz.token_set_ratio(x.lower(), y.lower())


# return textdistance.hamming.normalized_similarity(x.lower(),y.lower())

def compare(text, compared_data):
    compared_data['similarity'] = compared_data['term'].apply(lambda x: text_similarity(str(text), str(x)))
    compared_data = compared_data.drop_duplicates()
    compared_data = compared_data.sort_values(by='similarity', ascending=False)
    # compared_data = compared_data[compared_data['similarity']>80]
    # print(compared_data)
    return compared_data


def convert(x):
    x = str(x).split('.')
    if len(x) > 1:
        return x[0] + x[1][:1]
    else:
        return x[0]


def calculate(dx_code, predicted_icd):
    if dx_code in predicted_icd:
        return 1
    else:
        return 0


def tf_idf():
	
	for df in pd.read_csv(path+'result_100.csv',index_col=0, chunksize=2000):
		df['d'] = df['id'].groupby(df['id']).transform('count')
		d = df.groupby(['id','term']).size().to_frame(name = 'f').reset_index()
		result = pd.merge(df, d, on =["id","term"], how = 'inner')
		result['tf'] = result['f']/result['d']
		print (result[result['id'] == 803616])
		#print (df[df['id'] == 803616][['id','term']])
		#print (d[d['id'] == 803616])
		break

    standard_result.to_csv(path + 'result_100_validity_checked.csv')
