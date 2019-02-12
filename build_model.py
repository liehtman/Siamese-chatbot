import h5py
import keras.backend as K
import _pickle as pickle
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge
from keras.optimizers import Adadelta, Adamax
from keras.models import model_from_json
from keras.utils import plot_model

def build_model():
    hf = h5py.File('db/data_XY_Qw2v.h5', 'r')
    embeddings = hf['Embeddings'][:]
    hf.close()
    
    max_seq_length = 245
    n_hidden = 64
    gradient_clipping_norm = 1.25
    embedding_dim = 300
    
    def exponent_neg_manhattan_distance(left, right):
        return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))
    
    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,),  dtype='int32')
    embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)
    shared_lstm = LSTM(n_hidden)
    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)
    malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
    malstm = Model([left_input, right_input], [malstm_distance])
    optimizer = Adadelta(clipnorm=gradient_clipping_norm)
    malstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    plot_model(malstm, to_file='model.png')
    malstm.load_weights('weights/best.hdf5')

    return malstm

