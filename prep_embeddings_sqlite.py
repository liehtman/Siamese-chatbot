import sqlite3
import h5py
import pickle
from embPhrase import embPhrase
from classifierFuncs import classes

conn = sqlite3.connect('db/Bot_data.sqlite')
cursor = conn.cursor()
embedded_data = []

D = dict()
for cl in classes:
    D[cl] = []

def emb_table(name, intent):
    cursor.execute('''SELECT request, answer FROM {0} WHERE Intent = "{1}"'''.format(name, intent))
    data = cursor.fetchall()
    
    for pair in data:
        try:
            D[intent].extend([embPhrase(pair[0]), embPhrase(pair[1])])
        except KeyError as e:
            continue

tables = ['dialogues_v2', 'dialogues_v3', 'NUS_SMS_CORPUS', 'QA', 'BCF_CORPUS']
for table in tables:
    for cl in classes:
        emb_table(table, cl)
    print('One more done')
with open('db/embedded_sqlite_dict_big.pickle', 'wb') as handle:
    pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)

conn.close()













# import sqlite3
# import h5py
# import pickle
# from embPhrase import embPhrase
# from classifierFuncs import classes
# import pandas as pd
# import numpy as np

# conn = sqlite3.connect('db/Bot_data.sqlite')
# cursor = conn.cursor()
# embedded_data = []

# tables = ['BCF_CORPUS', 'dialogues_v2', 'dialogues_v3']

# for table in tables:
#     df = pd.read_sql_query('SELECT * FROM {0}'.format(table), conn)
#     cursor.execute('''DELETE FROM {0}'''.format(table))
#     for i, row in df.iterrows():
#         if i%2 != 0:
#             continue
#         else:
#             cursor.execute('''INSERT INTO {0} VALUES ("{1}","{2}","{3}")'''.format(table, row['request'], row['answer'], row['Intent']))
#     conn.commit()
#     # cursor.execute('''SELECT * FROM {0}'''.format(table))
#     # for line in cursor: print(line)

# df = pd.read_sql_query('SELECT * FROM QA', conn)
# c1 = df['request'].values[1:]
# c2 = df['answer'].values[:-1]
# c3 = df['Intent'].values[1:]
# df = pd.DataFrame(np.column_stack([c1, c2, c3]), columns=['request', 'answer', 'Intent'])
# cursor.execute('''DELETE FROM QA''')
# for i, row in df.iterrows():
#     cursor.execute('''INSERT INTO {0} VALUES ("{1}","{2}","{3}")'''.format('QA', row['request'], row['answer'], row['Intent']))
# conn.commit()
# conn.close()





# import pandas as pd
# import sqlite3
# from classifierFuncs import classify
# data = pd.read_csv('qa.txt', delimiter=';',error_bad_lines=False, names=['answer','request'])
# # print(data.head())
# conn = sqlite3.connect('db/Bot_data.sqlite')
# cursor = conn.cursor()
# cursor.execute('''DELETE FROM {0}'''.format('QA'))
# for i, row in data.iterrows():
#     try:
#         cl = classify(row['request'])[0]
#         cursor.execute('''INSERT INTO {0} VALUES ("{1}","{2}","{3}")'''.format('QA', row['request'], row['answer'], cl))
#     except Exception as e:
#         print(e)
#         continue
# conn.commit()
# cursor.execute('''SELECT * FROM {0}'''.format('QA'))
# for line in cursor: print(line)


