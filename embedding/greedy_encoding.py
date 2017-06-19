from .embedding_base import EmbeddingBase
from keras.layers import Dense
from keras.models import Sequential, model_from_json
from keras.layers.noise import AlphaDropout
import os
import numpy as np

class GreedyEncoder(EmbeddingBase):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self._x_test_set = None
        self._y_test_set = None
        self._x_train_set = None
        self._y_train_set = None
        self.blue_print = None
        self.use_bias = True
        self.batch_size = 40
        self.encoder_layer_stack = []
        self.encoding_stack = []
        self.encoded_test_input_stack = []
        self.encoded_train_input_stack = []
        self.input_layer_stack = []
        self.encoder = None
        self.decoder = None
        self.encoder_decoder = None
        self.encoder_decoder_input_shape = None
        self.encoder_decoder_output_shape = None
        self.encoder_input_shape = None
        self.encoder_output_shape = None
        self.decoder_input_shape = None
        self.decoder_output_shape = None
        super().__init__()

    def _get_encoder(self, x_train, y_train, x_test, y_test, i_dim, o_dim, e_dim, activation_1, activation_2, loss_function, epoch, optimizer, kernel_initializer='lecun_normal', dropout=None, dropout_rate=None, **kwargs):
        c_1, c_2 = {}, {}
        if kwargs.get('regularizer_1', None) is not None:
            c_1['activity_regularizer'] = kwargs.get('regularizer_1')
        if kwargs.get('bias_1', None) is not None:
            c_1['use_bias'] = kwargs.get('bias_1')
        if kwargs.get('regularizer_2', None) is not None:
            c_2['activity_regularizer'] = kwargs.get('regularizer_2')
        if kwargs.get('bias_2', None) is not None:
            c_2['use_bias'] = kwargs.get('bias_2')
        input_layer = Dense(i_dim, input_shape=(i_dim,), activation=activation_1, kernel_initializer=kernel_initializer)
        encoding_layer = Dense(e_dim, activation=activation_1, kernel_initializer=kernel_initializer, **c_1)
        decoding_layer = Dense(o_dim, activation=activation_2, kernel_initializer=kernel_initializer, **c_2)
        autoencoder = Sequential()
        autoencoder.add(input_layer)
        if dropout is not None and dropout_rate is not None:
            autoencoder.add(dropout(dropout_rate))
        autoencoder.add(encoding_layer)
        if dropout is not None and dropout_rate is not None:
            autoencoder.add(dropout(dropout_rate))
        autoencoder.add(decoding_layer)
        if dropout is not None and dropout_rate is not None:
            autoencoder.add(dropout(dropout_rate))

        optimizer = optimizer
        autoencoder.compile(optimizer=optimizer, loss=loss_function)
        autoencoder.fit(x_train, y_train,
                        epochs=epoch,
                        batch_size=self.batch_size,
                        shuffle=True,
                        verbose=self.verbose,
                        validation_data=(x_test, y_test))

        # build the encoder and return the encoder
        encoder = Sequential()
        encoder.add(input_layer)
        encoder.add(encoding_layer)
        ret = {
            'model': encoder,
            'input': input_layer,
            'i_dim': i_dim,
            'encode': encoding_layer,
            'e_dim': e_dim,
            'decode': decoding_layer,
            'o_dim': o_dim
        }
        return ret

    def fit(self, *args, **kwargs):
        # clean buffer, rebuild network
        self.encoder_layer_stack = []
        self.encoding_stack = []
        self.encoded_test_input_stack = []
        self.encoded_train_input_stack = []
        self.input_layer_stack = []

        self.encoded_train_input_stack.append(self._x_train_set)
        self.encoded_test_input_stack.append(self._x_test_set)

        for model_config in self.blue_print['stack']:
            o_dim = model_config['output_dimension']
            # train layer wise from here
            i_dim = model_config['input_dimension']
            e_dim = model_config['embedding_dimension']

            activation_1 = model_config['activation_1']
            activation_2 = model_config['activation_2']
            loss_function = model_config['loss_function']
            epoch = model_config['epoch']
            layer_config = {}
            if model_config.get('bias_1', None) is not None:
                layer_config['bias_1'] = model_config.get('bias_1')
            if model_config.get('bias_2', None) is not None:
                layer_config['bias_2'] = model_config.get('bias_2')
            if model_config.get('activity_regularizer_1', None) is not None:
                layer_config['regularizer_1'] = model_config['activity_regularizer_1']
            if model_config.get('activity_regularizer_2', None) is not None:
                layer_config['regularizer_2'] = model_config['activity_regularizer_2']
            optimizer = model_config['optimizer']
            greedy_AE_object = self._get_encoder(
                self.encoded_train_input_stack[-1], self._y_train_set,
                self.encoded_test_input_stack[-1],
                self._y_test_set, i_dim, o_dim, e_dim, activation_1, activation_2, loss_function, epoch,
                optimizer,
                **layer_config
            )
            self.encoding_stack.append(greedy_AE_object['model']) # input-embed-input tensor
            self.encoder_layer_stack.append(greedy_AE_object['encode']) # input-embed tensor
            self.input_layer_stack.append(greedy_AE_object['input'])
            self.encoded_test_input_stack.append(self.encoding_stack[-1].predict(self.encoded_test_input_stack[-1]))
            self.encoded_train_input_stack.append(self.encoding_stack[-1].predict(self.encoded_train_input_stack[-1]))

        # the input layer of stacked encoder, do I need to read weights here?
        input_layer = Dense(self.blue_print['stack'][0]['input_dimension'],
                            input_shape=(self.blue_print['stack'][0]['input_dimension'],),
                            weights=self.input_layer_stack[0].get_weights() if self.use_bias else [self.input_layer_stack[0].get_weights()[0]], use_bias=self.use_bias)
        self.encoder_decoder = Sequential()
        self.encoder_decoder.add(input_layer)
        # todo @charles add dropout option
        # self.encoder_decoder.add(AlphaDropout(0.1))
        for i in range(len(self.encoder_layer_stack)):
            # weights = self.encoder_layer_stack[i].get_weights() if self.use_bias else [self.encoder_layer_stack[i].get_weights()[0], np.zeros((self.encoder_layer_stack[i].get_weights()[0].shape[1]))]
            weights = self.encoder_layer_stack[i].get_weights() if self.use_bias else [self.encoder_layer_stack[i].get_weights()[0]]
            encoded = Dense(self.blue_print['stack'][i]['embedding_dimension'], weights=weights, use_bias=self.use_bias)
            self.encoder_decoder.add(encoded)
            # todo @charles add dropout option
            dropout = self.blue_print['stack'][i].get('dropout', AlphaDropout)
            dropout_rate = self.blue_print['stack'][i].get('dropout_rate', 0.1)
            # self.encoder_decoder.add(dropout(dropout_rate))

        output_layer = Dense(
            self.blue_print['stack'][-1]['output_dimension'],
            activation=self.blue_print['stack'][-1]['activation_2']
        )
        self.encoder_decoder.add(output_layer)
        # todo @charles add dropout option
        # self.encoder_decoder.add(AlphaDropout(0.1))
        self.encoder_decoder.compile(
            optimizer=self.blue_print['optimizer'],
            loss=self.blue_print['loss_function']
        )

        # the fine tuning does not converge!
        self.encoder_decoder.fit(
            (self._x_train_set),
            (self._x_train_set),
            epochs=self.blue_print['epoch'],
            batch_size=self.batch_size,
            shuffle=True,
            verbose=self.verbose,
            validation_data=(self._x_test_set, self._y_test_set)
        )

        self.encoder = Sequential()
        self.encoder.add(input_layer)
        for i in range(len(self.encoder_layer_stack)):
            weights = self.encoder_layer_stack[i].get_weights() if self.use_bias else [self.encoder_layer_stack[i].get_weights()[0]]
            encoded = Dense(self.blue_print['stack'][i]['embedding_dimension'], weights=weights, use_bias=self.use_bias)
            self.encoder.add(encoded)

        self.decoder = Sequential()
        # this is the input layer
        self.decoder.add(
            Dense(self.blue_print['stack'][-1]['embedding_dimension'],
                  input_shape=(self.blue_print['stack'][-1]['embedding_dimension'],),
                  use_bias=self.use_bias
                  )
        )
        self.decoder.add(Dense(self.blue_print['stack'][-1]['output_dimension'], activation=self.blue_print['stack'][-1]['activation_2'], weights=output_layer.get_weights() if self.use_bias else [output_layer.get_weights()[0]], use_bias=self.use_bias))

    def encode(self, x):
        return self.encoder.predict(x)

    def decode(self, x):
        return self.decoder.predict(x)

    def _noising(self, X, noise_factor):
        X_noisy = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
        X_noisy = np.clip(X_noisy, 0., 1.)
        return X_noisy

    def load(self, folder_path, model_name):
        encoder_arch_file = os.path.join(folder_path, '.'.join(['_'.join([model_name, 'encoder']), 'json']))
        encoder_weights_file = os.path.join(folder_path, '.'.join(['_'.join([model_name, 'encoder']), 'hdf5']))
        decoder_arch_file = os.path.join(folder_path, '.'.join(['_'.join([model_name, 'decoder']), 'json']))
        decoder_weights_file = os.path.join(folder_path, '.'.join(['_'.join([model_name, 'decoder']), 'hdf5']))
        encoder_decoder_arch_file = os.path.join(folder_path, '.'.join(['_'.join([model_name, 'encoder_decoder']), 'json']))
        encoder_decoder_weights_file = os.path.join(folder_path, '.'.join(['_'.join([model_name, 'encoder_decoder']), 'hdf5']))
        # Load autoencoder architecture + weights + shapes
        json_file = open(encoder_decoder_arch_file, 'r')  # read architecture json
        autoencoder_json = json_file.read()
        json_file.close()
        self.encoder_decoder = model_from_json(autoencoder_json)  # convert json -> model architecture
        self.encoder_decoder.load_weights(encoder_decoder_weights_file)  # load model weights
        self.encoder_decoder_input_shape = self.encoder_decoder.input_shape  # set input shape from loaded model
        self.encoder_decoder_output_shape = self.encoder_decoder.output_shape  # set output shape from loaded model

        # Load encoder architecture + weights + shapes
        json_file = open(encoder_arch_file, 'r')  # read architecture json
        encoder_json = json_file.read()
        json_file.close()
        self.encoder = model_from_json(encoder_json)  # convert json -> model architecture
        self.encoder.load_weights(encoder_weights_file)  # load model weights
        self.encoder_input_shape = self.encoder.input_shape  # set input shape from loaded model
        self.encoder_output_shape = self.encoder.output_shape  # set output shape from loaded model

        # Load decoder architecture + weights + shapes
        json_file = open(decoder_arch_file, 'r')  # read architecture json
        decoder_json = json_file.read()
        json_file.close()
        self.decoder = model_from_json(decoder_json)  # convert json -> model architecture
        self.decoder.load_weights(decoder_weights_file)  # load model weights
        self.decoder_input_shape = self.decoder.input_shape  # set input shape from loaded model
        self.decoder_output_shape = self.decoder.output_shape  # set output shape from loaded model




    def save(self, folder_path, model_name):
        encoder_arch_file = os.path.join(folder_path, '.'.join(['_'.join([model_name, 'encoder']), 'json']))
        encoder_weights_file = os.path.join(folder_path, '.'.join(['_'.join([model_name, 'encoder']), 'hdf5']))
        decoder_arch_file = os.path.join(folder_path, '.'.join(['_'.join([model_name, 'decoder']), 'json']))
        decoder_weights_file = os.path.join(folder_path, '.'.join(['_'.join([model_name, 'decoder']), 'hdf5']))
        encoder_decoder_arch_file = os.path.join(folder_path, '.'.join(['_'.join([model_name, 'encoder_decoder']), 'json']))
        encoder_decoder_weights_file = os.path.join(folder_path, '.'.join(['_'.join([model_name, 'encoder_decoder']), 'hdf5']))
        # Save autoencoder model arch + weights
        with open(encoder_decoder_arch_file, "w+") as json_file:
            json_file.write(self.encoder_decoder.to_json())  # arch: json format
        self.encoder_decoder.save_weights(encoder_decoder_weights_file)  # weights: hdf5 format

        # Save encoder model arch + weights
        with open(encoder_arch_file, "w+") as json_file:
            json_file.write(self.encoder.to_json())  # arch: json format
        self.encoder.save_weights(encoder_weights_file)  # weights: hdf5 format

        # Save decoder model arch + weights
        with open(decoder_arch_file, "w+") as json_file:
            json_file.write(self.decoder.to_json())  # arch: json format
        self.decoder.save_weights(decoder_weights_file)  # weights: hdf5 format

    def compile(self, use_bias=True, *args, **kwargs):
        # we want to call compile, but do nothing, don't raise error here
        self.use_bias = use_bias

    def arch(self, blue_print):
        '''
        config = {
            'stack': [
                {
                'input_dimension': input_dim,
                'output_dimension': input_dim,
                'embedding_dimension': 128,
                'activation_1': 'linear',
                'activity_regularizer_1': regularizers.l2(10e-4),
                'bias_1': True,
                'activation_2': 'sigmoid',
                'activity_regularizer_2': regularizers.l2(10e-4),
                'bias_2': True,
                'optimizer': keras.optimizers.Adadelta(),
                'loss_function': 'binary_crossentropy',
                'epoch': 1
            }
            ]
        }
        # optimizers
        all_classes = {
            'sgd': SGD,
            'rmsprop': RMSprop,
            'adagrad': Adagrad,
            'adadelta': Adadelta,
            'adam': Adam,
            'adamax': Adamax,
            'nadam': Nadam,
            'tfoptimizer': TFOptimizer,
        }
        Parameters
        ----------
        blue_print

        Returns
        -------

        '''
        self.blue_print = blue_print

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_training_data(self, x, y):
        '''
        :param x: numpy ndarray
        :param y:  numpy ndarray
        :return:  None
        '''
        self._x_train_set = x
        self._y_train_set = y

    def set_test_data(self, x, y):
        '''
        :param x: numpy ndarray
        :param y:  numpy ndarray
        :return:  None
        '''
        self._x_test_set = x
        self._y_test_set = y

    def encode_decode(self, x):
        # if self.verbose:
            # for layer in self.encoder_decoder.layers:
                # print(dir(layer))
                # print(layer.get_weights()[0])
                # print(layer.get_weights()[1])
                # print(sum(abs(layer.get_weights()[0])), sum(abs(layer.get_weights()[1] if len(layer.get_weights()) > 1 else [0])))
        # exit()
        return self.encoder_decoder.predict(x)

    def get_encoder_input_shape(self, *args, **kwargs):
        return self.encoder_input_shape

    def get_encoder_output_shape(self, *args, **kwargs):
        return self.encoder_output_shape


    def get_decoder_input_shape(self, *args, **kwargs):
        return self.decoder_input_shape

    def get_decoder_output_shape(self, *args, **kwargs):
        return self.decoder_output_shape