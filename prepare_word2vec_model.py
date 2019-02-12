import sqlite3
import pandas as pd 
from text_to_word_list import text_to_word_list
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords

conn = sqlite3.connect('db/Bot_data.sqlite')
cursor = conn.cursor()
stops = set(stopwords.words('english'))

sentences = []
tables = ['dialogues_v2', 'dialogues_v3', 'NUS_SMS_CORPUS', 'QA', 'BCF_CORPUS', 'eng_subs', 'quora_pairs']

def prep_one(name):
    pairs = cursor.execute('''SELECT request, answer FROM {0}'''.format(name)).fetchall()
    for pair in pairs:
        for sent in pair:
            sentences.append([word for word in text_to_word_list(sent) if word not in stops])

print('Reading...')
for table in tables:
    prep_one(table)
    print('One more done')
print('Done')
conn.close()
print('Start training...')    
w2v = Word2Vec(sentences, min_count=5, size=300, workers=4) # сразу обучится на данных предложениях
word_vectors = w2v.wv
word_vectors.save('word2vec/w2v_Q_model')
print('Done')















# cursor.execute('''CREATE TABLE quora_pairs (request, answer)''')
# train_df = pd.read_csv('train.csv')
# test_df = pd.read_csv('test.csv')
# cursor.execute('''DELETE FROM quora_pairs''')

# print('Running...')
# for i, row in train_df.iterrows():
# 	q1 = ' '.join(text_to_word_list(row['question1']))
# 	q2 = ' '.join(text_to_word_list(row['question2']))
# 	cursor.execute('''INSERT INTO quora_pairs VALUES("{}", "{}")'''.format(q1, q2))
# print('Train done')
# for i, row in test_df.iterrows():
# 	q1 = ' '.join(text_to_word_list(row['question1']))
# 	q2 = ' '.join(text_to_word_list(row['question2']))
# 	cursor.execute('''INSERT INTO quora_pairs VALUES("{}", "{}")'''.format(q1, q2))
# print('Done')
# conn.commit()