import h5py
import _pickle as pickle
import numpy as np
from embPhrase import embPhrase, vecToPhrase, w2v_similarity
from build_model import build_model 
from time import time
from sklearn.metrics.pairwise import cosine_similarity as cos
from classifierFuncs import classify, classes
from text_to_word_list import text_to_word_list

model = build_model()
D = None
with open('db/embedded_sqlite_dict_big.pickle', 'rb') as handle:
    D = pickle.load(handle)
print()

def get_answer(sample, index):
    return vecToPhrase(sample[index*2+1])

def debug(maxsim, intent, similar_question, variants, t):
    print('----------')
    print('DEBUG INFO:')
    print('Classified intent: ', intent)
    print('Similar question:', similar_question)
    print('Total candidates: ', variants)
    print('Similarity: {}%(SNN)'.format(maxsim))
    print('Answering time: ', t)
    print('----------')

def talk_snn_only():
    while True:
        request = input(' Human: ')
        if request == '':
            continue
        request = ' '.join(text_to_word_list(request))
        intent = classify(request)[0]
        sample = np.array(D[intent])
        phraseVec = embPhrase(request)
        t = time()
        X, Y = [], []
        for i, question in enumerate(sample[::2]):
            X.append(phraseVec[0])
            Y.append(question[0])
        X, Y = np.array(X), np.array(Y)
        siamese_sims = np.transpose(model.predict([X,Y]))[0]
        # siamese_sims = [int(x*100) for x in siamese_sims]
        argmaxes = [i for i, x in enumerate(siamese_sims) if x == max(siamese_sims)]
        argmax = np.random.choice(argmaxes)
        # variants = len(argmaxes)
        # max_sim = max(siamese_sims)
        # similar_question = vecToPhrase(sample[argmax*2])
        answer = get_answer(sample, argmax)
        print(' Machine: {0}'.format(answer))
        # debug(max_sim, intent, similar_question, variants, time() - t)

talk_snn_only()