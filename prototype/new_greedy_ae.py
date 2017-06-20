from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.losses import binary_crossentropy
input_dim = 2000
output_dim = input_dim
encoder_stack = [
    Dense(1000, activation='linear', input_shape=(input_dim,)),
    Dense(500, activation='linear'),
    Dense(100, activation='linear'),
]
decoder_stack = [
    Dense(500, activation='linear'),
    Dense(1000, activation='linear'),
    Dense(input_dim, activation='sigmoid')
]
# todo @charles need to work on the output data
X, Y = [], []
encoder = Sequential()
decoder = Sequential()
AE = Sequential()
AE.add(encoder_stack[0])
for layer in encoder_stack[1:]:
    # layer-wise training here, create a input layer of current stack, build a AE from the input layer
    # after training, set the weights of AE to the weights of trained greedy layer
    last_layer = AE.layers[-1]
    prev_input_dim = last_layer.input_shape[1]
    prev_output_dim = last_layer.output_shape[1]
    AE.add(layer)
    cur_input_dim = prev_output_dim
    cur_output_dim = AE.layers[-1].output_shape[1]
    tmp_encode = Dense(cur_input_dim, input_shape=(cur_output_dim,), activation='linear')
    tmp_decode = Dense(output_dim, activation='sigmoid')
    tmp_AE = Sequential()
    tmp_AE.add(tmp_encode)
    tmp_AE.add(tmp_decode)
    tmp_AE.compile(Adadelta, binary_crossentropy)
    # todo @charles, need to work on the input data
    _x = []
    tmp_AE.fit(_x, Y)

    # loads the weights from greedy layer
    AE.layers[-1].set_weights(tmp_encode.get_weights())
    exit()
    # tmp_layer = Dense(layer.input_shape)
    # OK the best way to do this is to define the input and output in config,
    # otherwise I have to read from previous layer (layer.output_shape)
    print(layer.input_shape)
    print(layer.output_shape)
    exit()
for layer in decoder_stack:
    AE.add(layer)
# fit AE here


# reload encoder
for layer in encoder_stack:
    encoder.add(layer)
# reload and decoder
encoded_input = Dense(500, input_shape=(100,))
encoded_input.set_weights(decoder_stack[0].get_weights())

decoder.add(encoded_input)
for layer in decoder_stack[1:]:
    decoder.add(layer)