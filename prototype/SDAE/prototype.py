from keras.layers import Dense, Convolution2D, Dropout
from keras.layers.noise import GaussianNoise
from keras.models import Sequential
from keras.activations import selu, relu, linear, sigmoid
from keras.regularizers import l1
import numpy as np
import os, sys

cur_dir = os.path.dirname(__file__)
project_root = os.path.join(cur_dir, '..', '..')
model_path = os.path.join(project_root, 'models', 'nlp')
model_name = 'w2v.txt'
model_path = os.path.join(model_path, model_name) # chain assignment
print(model_path)



sentences = [
    'cat food poo',
    'dog cat'
]

def roll_out(corpus, AE, batch_size=1000):
    count = 0
    ds = []
    for sentence in corpus:
        words = sentence.split(' ')
        print(words)
        for i in range(0, len(words)-1):
            print(words[i], words[i+1])

        samples = words

        ds.append(samples)
        count += 1
        if count == batch_size:
            yield np.array(ds)
            ds = []
    return ds
for each in roll_out(sentences, None):
    pass
exit()
from gensim.models.word2vec import Word2Vec

w2v = Word2Vec.load_word2vec_format(model_path)

print(w2v.vector_size)

AE = Sequential()
encoder = Sequential()

i = Dense(w2v.vector_size * 2, input_shape=(w2v.vector_size * 2,))
AE.add(i)
encoder.add(i)
h = Dense(w2v.vector_size, activation=linear)
AE.add(h)
encoder.add(h)
o = Dense(w2v.vector_size*2, activation=sigmoid)
AE.add(o)
print('abb')
roll_out(sentences, AE)
print('aa')