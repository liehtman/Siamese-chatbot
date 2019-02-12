import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from text_to_word_list import text_to_word_list
from gensim.models import KeyedVectors

EMBEDDING_FILE = 'word2vec/w2v_Q_model'
model = KeyedVectors.load(EMBEDDING_FILE)
index2word_set = set(model.index2word)

with open('db/vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f)

def notnull(L):
    for value in L:
        if value != 0: yield value

def embPhrase(string, maxlen=245):
    phraseVec = []
    splitted = text_to_word_list(string)
    for word in splitted:
        try:
            phraseVec.append(vocabulary[word])
        except KeyError:
            continue
    phraseVec = np.array(phraseVec).reshape((1, len(phraseVec)))
    phraseVec = pad_sequences(phraseVec, maxlen=maxlen)
    return phraseVec

def vecToPhrase(vec):
    words = []
    for value in notnull(vec[0]):
        words.append(list(vocabulary.keys())[list(vocabulary.values()).index(value)])
    return ' '.join(words)

def w2v_avg_vector(sentence, model, num_features=300):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def w2v_similarity(sentence1, sentence2):
    sim = model.wmdistance(sentence1, sentence2)
    return sim