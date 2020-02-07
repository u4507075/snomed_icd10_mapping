# Auto-mapping ICD-10 using SNOMED-CT

## Researchers
1. Natthanaphop Isaradech, fifth year medical student, Faculty of Medicine, Chiang Mai University, Chiang Mai, Thailand
2. Assistant Professor Piyapong Khumrin, MD, Faculty of Medicine, Chiang Mai University, Chiang Mai, Thailand

## Duration
1 month (February 2020)

## Introduction 

### Problem statement

### Prior work

## Objective
1. to match string presenting in clinical document with SNOMED-CT to map ICD-10.

## Aim
1. Be able to 50% correctly map ICD-10.

## Materials and methods
### Target group

### Dataset

### How to use SSH server
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
15. python3 (file name) = run your python file
16. rm (file name) = to PERMANENTLY delete the file
##### Github 
1. git status = check status github (update or not)
2. git fetch = refresh status form server (check update)
3. git pull = pull update -> (enter username + password)
4. git add . = add all new updates to local git (similarly to 'stages')
5. git commit -m "(comment here)" = comment what you update to local git
6. git push = push code from local git to server git (! always pull the lastest update prior to git push) -> (enter username + password)
### Download file from server on your computer command prompt 
C:\Users\ASUS>scp root@xxx.xxx.xxx.xxx:~/icd10/secret/data/result2.csv C:\Users\ASUS\Documents\GitHub\secret\data\result2.csv

### Results

### Discussion

### Limitations
