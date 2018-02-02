import json
import csv

# news_parsed = json.load(open("sample-1M.jsonl"))

with open('sample-1M.jsonl', 'r') as myfile:
    x=myfile.read()

x = json.loads(x)

f = csv.writer(open("test.csv", "wb+"))    