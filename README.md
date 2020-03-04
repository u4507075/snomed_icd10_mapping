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
-SNOMED, excel files 
## Objective
To match string presenting in clinical document with SNOMED-CT to map ICD-10.

## Aim
Be able to 50% correctly map ICD-10.

## Materials and methods
### Meterials
5 csv files that were used to create the algorithm are as follows:
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
2. Matching discharge summary to SNOMED_concept_id codes
3. Matching terms in discharge summary to icd-10 codes & terms by using SNOMED_concept_id codes
#### Cleaning data
def clean_data() and def start_data() was used to clear strings that are not important and meaningless in physicians' discharge summary notes. The function contains the following codes: 

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
#### Matching discharge summary to SNOMED_concept_id codes

#### Matching terms in discharge summary to icd-10 codes & terms by using SNOMED_concept_id codes

### Target group

### Dataset

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

### Discussion

### Limitations
