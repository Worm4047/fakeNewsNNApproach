import json
import sys
import csv
"""
Module to read 50000 news items from signal media dataset and
13000 items from kaggle fake news datset
"""
csv.field_size_limit(sys.maxsize)
f = open('complete_signal_dataset.csv', 'w')
i = 0
f.write('{%s{%s{%s{%s\n'%("id","title","content","fakeness"))

with open("sample-1M.json", "r") as ins:
    for line in ins:
    	i = i+1
    	obj = json.loads(line)
    	id = i
    	title = obj['title'].replace("\n"," ")
    	title = title.replace("{"," ")
    	content = obj['content'].replace("\n"," ")
    	content = content.replace("{"," ")
    	f.write('{%d{%s{%s{%d\n'%(id,title,content,0))
    	if(i>=50000):
    		break
print("Read signal Media dataset--50000 items")       
with open("fake.csv", "r") as l:
	reader = csv.reader(l)
	for row in reader:
		i = i+1
		if(i == 50001):
			continue
		id = i-1
		title = row[4].replace("\n"," ")
		title = title.replace("{"," ")
		content = row[5].replace("\n"," ")
		content = content.replace("{"," ")
		f.write('{%d{%s{%s{%d\n'%(id,title,content,1))
		print(i-1)
		if(i>=63000):
			break	
f.close()
print("Read Fakenews datset-kaggle--13000 items")  			

