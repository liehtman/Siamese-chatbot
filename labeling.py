import sqlite3
from classifierFuncs import classify
from text_to_word_list import text_to_word_list
import pandas as pd

conn = sqlite3.connect('db/Bot_data.sqlite')
cursor = conn.cursor()


tables_small_set = ['NUS_SMS_CORPUS', 'BCF_CORPUS']
tables = ['dialogues_v2', 'dialogues_v3', 'NUS_SMS_CORPUS', 'QA', 'BCF_CORPUS']
tables_last = ['eng_subs', 'QA']

def prep_one(table, name):
	df = pd.read_sql_query('SELECT * FROM {0}'.format(table), conn)
	cursor.execute('''DELETE FROM {0}'''.format(table))
	cursor.execute("alter table {0} add column '{1}' 'text'".format(table, name))

	for i, row in df.iterrows():
		if row['request'] and row['answer']:
			clear_sent = ' '.join(text_to_word_list(row['request']))
			cl = classify(clear_sent)[0]
			cursor.execute('''INSERT INTO {0} VALUES ("{1}","{2}","{3}")'''.format(table, row['request'], row['answer'], cl))
		if i % 1000 == 0:
			print('{0}/{1} from table {2} done'.format(i, df.shape[0], table))
	conn.commit()

	cursor.execute("SELECT * FROM {} LIMIT 10".format(table))
	res = cursor.fetchall()
	for r in res: print(r)

# for table in tables_last:
# 	prep_one(table, 'Intent')
# 	print('one done')
prep_one('dialogues_sp', 'Intent')