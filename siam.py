import h5py
import pickle
import pandas as pd
import numpy as np
import re
import itertools
import datetime
from time import time
import keras.backend as K
from gensim.models import KeyedVectors, Word2Vec
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from text_to_word_list import text_to_word_list
from build_model import build_model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

def full_preparing():
    print('Load training and test set...')
    TRAIN_CSV = 'train.csv'
    TEST_CSV = 'test.csv'
    EMBEDDING_FILE = 'word2vec/w2v_Q_model'
    
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    stops = set(stopwords.words('english'))
    
    vocabulary = dict()
    inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
    word2vec = KeyedVectors.load(EMBEDDING_FILE)
    questions_cols = ['question1', 'question2']
    
    print('Iterate over the questions only of both training and test datasets...')
    for dataset in [train_df, test_df]:
        for index, row in dataset.iterrows():
            # Iterate through the text of both questions of the row
            for question in questions_cols:
                q2n = []  # q2n -> question numbers representation
                for word in text_to_word_list(row[question]):
                    # Check for unwanted words
                    if word in stops and word not in word2vec.vocab:
                        continue
                    if word not in vocabulary:
                        vocabulary[word] = len(inverse_vocabulary)
                        q2n.append(len(inverse_vocabulary))
                        inverse_vocabulary.append(word)
                    else:
                        q2n.append(vocabulary[word])
                # Replace questions as word to question as number representation
                dataset.set_value(index, question, q2n)
    print('Dataset is ready')
    
    embedding_dim = 300
    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored
    
    print('Building the embedding matrix...')
    for word, index in vocabulary.items():
       if word in word2vec.vocab:
           embeddings[index] = word2vec[word]
    del word2vec
    max_seq_length = max(train_df.question1.map(lambda x: len(x)).max(),
                          train_df.question2.map(lambda x: len(x)).max(),
                          test_df.question1.map(lambda x: len(x)).max(),
                          test_df.question2.map(lambda x: len(x)).max())
    
    print(max_seq_length)
    print('Split to train validation...')
    validation_size = 40000
    training_size = len(train_df) - validation_size
    
    X = train_df[questions_cols]
    Y = train_df['is_duplicate']
    
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)
    
    print('Split to dicts...')
    X_train = {'left': X_train.question1, 'right': X_train.question2}
    X_validation = {'left': X_validation.question1, 'right': X_validation.question2}
    X_test = {'left': test_df.question1, 'right': test_df.question2}
    
    print('Convert labels to their numpy representations...')
    Y_train = Y_train.values
    Y_validation = Y_validation.values
    
    print('Zero padding...')
    for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

    print('Saving the files...')
    h5f = h5py.File('db/data_XY_Qw2v.h5', 'w')
    h5f.create_dataset('X_right', data=X_train['right'])
    h5f.create_dataset('X_left', data=X_train['left'])
    h5f.create_dataset('Y_train', data=Y_train)
    h5f.create_dataset('X_validation_left', data=X_validation['left'])
    h5f.create_dataset('X_validation_right', data=X_validation['right'])
    h5f.create_dataset('Y_validation', data=Y_validation)
    h5f.create_dataset('Embeddings', data=embeddings)
    h5f.close()
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def load_weights_and_train():    
    malstm = build_model()
    
    hf = h5py.File('db/data_XY_Qw2v.h5', 'r')
    # with h5py.File('data_XY.h5', 'r') as hf:
    X_train_left = hf['X_left'][:]
    X_train_right = hf['X_right'][:]
    Y_train = hf['Y_train'][:]
    X_validation_left = hf['X_validation_left'][:]
    X_validation_right = hf['X_validation_right'][:]
    Y_validation = hf['Y_validation'][:]
    embeddings = hf['Embeddings'][:]
    hf.close()
    
    max_seq_length = 245
    batch_size = 64
    n_epoch = 20
    
    # The visible layer
    training_start_time = time()
    checkpoint = ModelCheckpoint('weights/weights.{epoch:02d}-{val_loss:.3f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    print('Done')
    malstm_trained = malstm.fit([X_train_left, X_train_right], Y_train, batch_size=batch_size, nb_epoch=n_epoch,
                                validation_data=([X_validation_left, X_validation_right], Y_validation), callbacks=callbacks_list)
    
    print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))

# full_preparing()
load_weights_and_train()
#model = load_model('MaLSTM.h5')
