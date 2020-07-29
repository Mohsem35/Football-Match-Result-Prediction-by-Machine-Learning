import csv
import random


file=open("CompleteDataset.csv", "r")
reader = csv.reader(file)
names = []
x = 0
for line in reader:
	if line[8] == "Coventry":
		names.append(line[8])

    # t=line[1],line[8]
    # print(t)

random.shuffle(names)

for name in names:
	print(name)
	x+=1
	if x>11: break
