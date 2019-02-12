import _pickle as pickle
import numpy as np
import pandas as pd
from sklearn.semi_supervised import label_propagation
from embPhrase import embPhrase
from scipy import sparse
import gc
import pickle
from sklearn.metrics.pairwise import pairwise_kernels, cosine_similarity
from time import time

def my_kernel(X, Z):
	return pairwise_kernels(X, Z, metric="cosine") 

def prepare_all():
	with open('db/embedded_sqlite_dict_big.pickle', 'rb') as handle:
	    D = pickle.load(handle)

	training_data = pd.read_csv('intents_data.txt', '\t')
	training_data['class'] = pd.Categorical(training_data['class'])
	training_data['code'] = training_data['class'].cat.codes

	classes = list(training_data['class'].unique())
	X, Y = [], []
	for i, row in training_data.iterrows():
		X.append(embPhrase(row['sentence']))
		Y.append(row['code'])
	del training_data

	for cl in classes:
		for vec in np.array(D[cl]):
			X.append(vec)
	del D
	
	while len(Y) != len(X):
		Y.append(-1)

	X, Y = sparse.csr_matrix(np.array(X)[:,0]), sparse.csr_matrix(np.array(Y))
	print(X.shape, Y.shape)
	kernels = ['knn']
	
	for kernel in kernels:
		lp_model = label_propagation.LabelSpreading(kernel=kernel, n_neighbors=9, n_jobs=-1)
		gc.collect()

		print('Training...')
		s = time()
		lp_model.fit(X.toarray(), Y.toarray()[0])

		print('Trained in {0}'.format(time()-s))
		print('Score:', lp_model.score(X.toarray(), Y.toarray()[0]))
		print('Dumping')
		pickle.dump(lp_model, open('LabelSpreadingModel_{0}.sav'.format(kernel), 'wb'))
		# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
		preds = lp_model.transduction_
		np.savetxt('cluster_preds_{0}.txt'.format(kernel), preds)

prepare_all()
