from fuzzywuzzy import fuzz
import pandas as pd
import re
import textdistance

def text_similarity(x,y):
  	return fuzz.token_set_ratio(x.lower(),y.lower())
	#return textdistance.hamming.normalized_similarity(x.lower(),y.lower())
def compare(text,compared_data):
	compared_data['similarity'] = compared_data['term'].apply(lambda x: text_similarity(str(text),str(x)))
	compared_data = compared_data.drop_duplicates()
	compared_data = compared_data.sort_values(by='similarity', ascending=False)
	#compared_data = compared_data[compared_data['similarity']>80]
	#print(compared_data)
	return compared_data

