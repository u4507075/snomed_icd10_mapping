# Auto-mapping ICD-10 using SNOMED-CT

## Researchers
1. Natthanaphop Isaradech, forth year medical student, Faculty of Medicine, Chiang Mai University, Chiang Mai, Thailand
2. Assistant Professor Piyapong Khumrin, MD, Faculty of Medicine, Chiang Mai University, Chiang Mai, Thailand

## Duration
1 month (February 2020)

## Introduction
### Problem statement
Icd-10 stands for International Statistical Classification of Diseases and Related Health Problems which is a medical classification documented by World Health Organization (WHO). The list contains an international identification numbers for diagnosis, signs and symptoms, abnormla finding, procedures, etc. Physicians have to document ICD-10 numbers and its terms in discharge summary note when patients are discharged from hospital. Documenting icd-10 is a repetitive task and time-consuming for physicians. As a result, having an algorithm that could automatically match physician's terms into icd-10 terms, would save their time resuorces and focus as well as avoiding documentation errors from human. 
### Prior work
-SNOMED, csv files 
## Objective
To match string presenting in clinical document with SNOMED-CT to map ICD-10.

## Aim
Be able to 50% correctly map ICD-10.

## Materials and methods
### Meterials
5 csv files that were used to create the algorithm as following:
#### 1. discharge_summary.csv
This file is the collection of physicians' discharge dischrage summary in Maharaj Nakorn Chiang Mai Hospital. The file contains discharge summary notes, icd-10 codes and terms that were made by physicians. 
#### 2. abnormality.csv
This file was acquired from [...]. It contains a group of abnormality terms in medicine that are already matched to concept_id codes.
#### 3. disorder.csv
This file was acquired from [...]. It contains a group of disorder terms in medicine that are already matched to concept_id codes.
#### 4. finding.csv
This file was acquired from [...]. It contains a group of finding terms in medicine that are already matched to concept_id codes.
#### 5. procedure.csv
This file was acquired from [...]. It contains a group of procedures terms in medicine that are already matched to concept_id codes.

### Methods
In order to correctly match physician's notes in discharge summary to icd-10 terms, the algorithm must do the following 3 steps:
1. Cleaning data
2. Matching terms in discharge summary to icd-10 codes & terms by using SNOMED_concept_id codes
3. Improve matching accuracy by apply TF-IDF concepts
#### 1. Cleaning data
def clean_data() and def start_data() was used to clear strings that are not important and meaningless in physicians' discharge summary notes. In this dataset strings that are '&lt;br/&gt', '&gt;', and '<.*?>'  are considered not important

The function contains the following codes: 
```
stop_words = stopwords.words('english')
path = "../secret/data/"

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
```
#### 2. Matching terms in discharge summary to icd-10 codes & terms by using SNOMED_concept_id codes
In order to determine which terms are similar, the Levenshtein Distance concept was used. Levenshtein Distance was created by Vladimir Levenshtein in 1965. It is a formula used to measure how apart are two sequences of words by finding the minimum number of operations needed to change a word sequence into the other using insertions, deletions or substitutions. For example the Levenshtein distance between "MOCHA" and "MOCHI" is 1 (1 substitution, "A" to "I") or the Levenshtein distance between "LATTE" and "MANTLE" is 3. (2 insertions, add "M" and "N", and 1 deletion, "T") The greater the distance, the more different two words are. 

Fuzzy wuzzy package uses the concept of Levenshtein Distance by computing the standard Levenshtein distance similarity ratio between two sequences so that outcome yields in percentage. The higher the percent is, the more similar two sequences are. As a result, fuzz.token_set_ratio was applied. This function computes the standard Levenshtein distance similarity ratio by using 2 more conditions. One is to XXXXXXXXXXXXXXXXXXXXXXXXXXXX

by using fuzz.set_token_ratio function as following codes: 
```
##### in compare_text.py 

from itertools import count

from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
import re
import textdistance

path = "../secret/data/"


def text_similarity(x, y):
    return fuzz.token_set_ratio(x.lower(), y.lower())

def compare(text, compared_data):
    compared_data['similarity'] = compared_data['term'].apply(lambda x: text_similarity(str(text), str(x)))
    compared_data = compared_data.drop_duplicates()
    compared_data = compared_data.sort_values(by='similarity', ascending=False)
    # compared_data = compared_data[compared_data['similarity']>80]
    # print(compared_data)
    return compared_data
    
##### in main.py 

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
```
#### Improve matching accuracy by apply TF-IDF concepts

### Notes on How to use SSH server
#### Putty
1. Enter IP address in "xxx.xxx.xxx.xxx" host name (or IP address) -> click "open"
2. login as: (enter username)
3. (enter password)
4. ls = show infomation in directory
5. cd = change directory 
6. [Tab] = auto-correction
7. Change directory to root@CLIT000038_1001:~/icd10/secret/data
8. sudo nano (your file) = Edit your file 
9. in sudo nano, [ctrl] + [o] = save file
10. in sudo nano, [ctrl] + [X] = exit
11. screen -ls = check currently running screens 
12. screen -r (enter screen number) = enter the running screen
13. [Ctrl] + [A] + [D] = exit the screen
14. screen -S_(enter screen name) = create a new screen
15. screen -X -S (session # your want to kill) quit = delete a screen
16. python3 (file name) = run your python file
17. rm (file name) = to PERMANENTLY delete the file
##### Github 
1. git status = check status github (update or not)
2. git fetch = refresh status form server (check update)
3. git pull = pull update -> (enter username + password)
4. git add . = add all new updates to local git (similarly to 'stages')
5. git commit -m "(comment here)" = comment what you update to local git
6. git push = push code from local git to server git (! always pull the lastest update prior to git push) -> (enter username + password)
#### Download file from server on your computer command prompt 
C:\Users\ASUS>scp root@xxx.xxx.xxx.xxx:~/icd10/secret/data/result2.csv C:\Users\ASUS\Documents\GitHub\secret\data\result2.csv

### Results
500 discharge summary were used to run the algorithm. The algorithm could 100% correctly match 149 samples icd-10 terms compared to the physician's diagnosis. There were 1 80%-correctly-matched, 3 75%-correctly-matched, 8 66.67%-correctly-matched, 2 60%-correctly-matched, 86 50%-correctly-matched, 1 44.44%-correctly-matched, 3 42.86%-correctly-matched, 5 40%-correctly-matched, 2 37.5%-correctly-matched, 59 33.33%-correctly-matched, 2 30%-correctly-matched, 4 28.57-correctly-matched, 39 25%-correctly-matched, 1 23.08%-correctly-matched, 1 22.22%-correctly-matched, 29 20%-correctly-matched, 6 18.18%-correctly-matched, 16 16.67%-correctly-matched, 2 15.38%-correctly-matched, 13 14.29%-correctly-matched, 8 12.5%-correctly-matched, 4 11.11%-correctly-matched, 3 10%-correctly-matched, 1 9.09%-correctly-matched, 1 7.69%-correctly-matched

In conclusion, there 249 discharge summaries from 500 samples that the algorithm could 50% or more correctly match to icd-10 terms
### Discussion
String matching by using the concept of Levenshtein Distance could correctly match icd-10 terms from physician's discharge summary at 49.8% accuracy. The algorithm could be improved by using the concept of Term frequency–inverse document frequency or TF-IDF which is a way to communicate computers how important a word in a document is. In this project, we use calculate TF/IDF by having TF divided by IDF. 

TF or term frequency were calculated by the number of a term in a document divided by total amount of terms in that document. For example:
Given TF = f(term,document) and a document = "The sky is blue. The sky is beautiful", the values of TF for each term in these 2 documents are as follows:

TF1 = f("The",document1) = f(2,8) = 2/8 = 0.25
TF2 = f("sky",document1) = f(2,8) = 2/8 = 0.25
TF3 = f("is",document1) = f(2,8) = 2/8 = 0.25
TF4 = f("blue",document1) = f(1,8) = 2/8 = 0.125
TF5 = f("beautiful",document1) = f(2,8) = 2/8 = 0.125

The greater TF is, the more frequent the term appears in a document

IDF or inverse-document frequency is used to measure importance of a term in all documents. In this project, we calculate IDF by Total documents (N) divided by Total documents that the term appears (df(t))

So when a term in a discharge summary is likely to be the definitive diagnosis, the term should have high TF and low IDF.

By applying all of these concepts the algorithm should be able to correctly match physician's discharge summary notes to icd-10 terms if the definitive diagnosis was noted. 

For example, 
### Limitations
#### The algorithms are not able to comprehend abbrevations
For example, if the physician noted C/S in discharge summary instead of Cesarean section, the algorithm will not match cesarean section to icd-10 term. 
#### The algorithm do not understand tenses 
The algorithms are not able to distinguish past, present and future noted in discharge summary. If a patient had a history of ruptured appendicitis 10 years prior and his present disease is STEMI this time is dyspnea on exertion, the algorithm would comprehend ruptured appendicitis and STEMI as present problems which will lead to icd-10 mismatching. 
#### Word sequence were not counted for icd-10 matching in this algorithm 
The algorithm do not weigh the importance of term sequences. If a term was placed in the beginning of paragraph could match athe last term in the document meaningfully, the algorithms willcombine those terms and find a matched icd-10.

For example, given a document of a patient diagnosed with ruptured ectopic pregnancy: "PE: normal consciousness, no pale conjunctive, anicteric sclera,..., definitive diagnosis: ruptured ectopic pregnancy" The algorithms could combine 'normal' and 'pregnancy' to find an icd-10 match like "normal pregnancy" which is an incorrect principles diagnosis
#### The algorithms do not understand negation
Negative findings are commonly reported in physician's discharge summary notes such as no palpaple mass, no pale conjunctiva, no jaundice, no hepatosplenomagaly. Our algorithms do not yet understand negation phrases. It will comprehend "no palpable mass" as "palpable mass"
