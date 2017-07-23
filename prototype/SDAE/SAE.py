from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.metrics import mean_absolute_percentage_error


class SequentialAE(object):
    def __init__(self):
        pass

    def init_arch(self, input_dim=400):
        ae = Sequential()
        encoder = Sequential()
        decoder = Sequential()
        encode_input = Dense(input_dim, input_shape=(input_dim*2,))
        decode = Dense(input_dim*2, activation='linear')
        # auto encoder
        ae.add(encode_input)
        ae.add(decode)
        # encoder
        encoder.add(encode_input)
        # decoder
        decoder.add(decode)

        ae.compile(optimizer=Adam(), loss='mean_absolute_percentage_error', metrics=mean_absolute_percentage_error)


    def train(self, sentences):
        for sentence in sentences:
            pass
