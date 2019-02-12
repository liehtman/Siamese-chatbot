import pandas as pd

data = pd.read_csv('train_2.txt', sep='\t')
output = open('extra_train_data.txt', 'w')
id = 404351
qid = 789802
for i, row in data.iterrows():
	try:
		row[1] = ''.join([char for char in row[1] if char != ','])
		row[0] = ''.join([char for char in row[0] if char != ','])
	except: continue
	line = '''"{}","{}","{}","{}","{}","{}"'''.format(id, qid, qid+1, row[0], row[1], row[2])
	id += 1
	qid += 2
	# print(line)
	output.write(line + '\n')
	# print(row[1])
	# break