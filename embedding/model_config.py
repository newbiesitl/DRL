import keras
from keras.layers import *


output_shape = (64, 40)
# output_shape = (28, 28)
input_dim = output_shape[1] * output_shape[0]
use_bias = True
greedy_epoch = 20
fine_tuning_epoch = 40
activation_1 = 'linear'
activation_2 = 'sigmoid'
stack_activation = 'sigmoid'
optimizer = keras.optimizers.Adadelta()
# optimizer = keras.optimizers.Adagrad()

input_shape=(64,40,3)
embedded_input_shape=()
c_toy = {
    'input_layer':
    {
        'conv': Conv2D(filters=32, kernel_size=(5, 5), strides=(2,2),
                       activation='relu', padding='same',
                       kernel_initializer=keras.initializers.lecun_uniform(),
                       input_shape=input_shape),
        'pooling': MaxPooling2D(pool_size=(2, 2)),
        'dropout': Dropout(0.25)
    },
    'embedded_input_layer':
    {
        'conv': Conv2D(filters=32, kernel_size=(5, 5), strides=(2,2),
                       activation='relu', padding='same',
                       kernel_initializer=keras.initializers.lecun_uniform(),
                       input_shape=input_shape),
        'pooling': MaxPooling2D(pool_size=(2, 2)),
        'dropout': Dropout(0.25)
    },
    'encoding_stack': [
        Conv2D(filters=16, kernel_size=(3, 3), strides=(2,2),
               activation='relu', padding='same',
               kernel_initializer=keras.initializers.lecun_uniform()),
    ],
    'decoding_stack': [
        Conv2D(filters=16, kernel_size=(3, 3), strides=(2,2),
               activation='relu', padding='same',
               kernel_initializer=keras.initializers.lecun_uniform()),
    ]
}


f_2000_1000_300 = {
    'stack': [
        {
            'input_dimension': input_dim,
            'output_dimension': input_dim,
            'embedding_dimension': 2048,
            'activation_1': activation_1,
            # 'activity_regularizer_1': regularizers.l2(10e-4),
            'bias_1': use_bias,
            'activation_2': activation_2,
            # 'activity_regularizer_2': regularizers.l2(10e-4),
            'bias_2': use_bias,
            'optimizer': optimizer,
            'loss_function': 'binary_crossentropy',
            'epoch': greedy_epoch,
            'encode': True
        },
        {
            'input_dimension': 2048,
            'output_dimension': input_dim,
            'embedding_dimension': 1024,
            'activation_1': activation_1,
            # 'activity_regularizer_1': regularizers.l2(10e-4),
            'bias_1': use_bias,
            'activation_2': activation_2,
            # 'activity_regularizer_2': regularizers.l2(10e-4),
            'bias_2': use_bias,
            'optimizer': optimizer,
            'loss_function': 'binary_crossentropy',
            'epoch': greedy_epoch,
            'encode': True
        },
        {
            'input_dimension': 1024,
            'output_dimension': input_dim,
            'embedding_dimension': 300,
            'activation_1': activation_1,
            # 'activity_regularizer_1': regularizers.l2(10e-4),
            'bias_1': use_bias,
            'activation_2': stack_activation,
            # 'activity_regularizer_2': regularizers.l2(10e-4),
            'bias_2': use_bias,
            'optimizer': optimizer,
            'loss_function': 'binary_crossentropy',
            'epoch': greedy_epoch,
            'encode': True
        },
    ],
    'epoch': fine_tuning_epoch,
    'optimizer': optimizer,
    'loss_function': 'binary_crossentropy'
}

f_2000_1000_500 = f_2000_1000_300
f_2000_1000_500['stack'][-1]['embedding_dimension'] = 500
f_2000_1000_500_1000_2000 = f_2000_1000_500



f_4000_2000_1000 = f_2000_1000_300
f_4000_2000_1000['stack'][0]['embedding_dimension'] = 4000
f_4000_2000_1000['stack'][1]['input_dimension'] = 4000
f_4000_2000_1000['stack'][1]['embedding_dimension'] = 2000
f_4000_2000_1000['stack'][-1]['input_dimension'] = 2000
f_4000_2000_1000['stack'][-1]['embedding_dimension'] = 1000


f_4000_2000_1000_2000_4000 = f_4000_2000_1000
f_4000_2000_1000_2000_4000['stack'].append(
        {
            'input_dimension': 1000,
            'output_dimension': input_dim,
            'embedding_dimension': 2000,
            'activation_1': activation_1,
            # 'activity_regularizer_1': regularizers.l2(10e-4),
            'bias_1': use_bias,
            'activation_2': stack_activation,
            # 'activity_regularizer_2': regularizers.l2(10e-4),
            'bias_2': use_bias,
            'optimizer': optimizer,
            'loss_function': 'binary_crossentropy',
            'epoch': greedy_epoch,
            'encode': False
        }
)

f_4000_2000_1000_2000_4000['stack'].append(
{
            'input_dimension': 2000,
            'output_dimension': input_dim,
            'embedding_dimension': 4000,
            'activation_1': activation_1,
            # 'activity_regularizer_1': regularizers.l2(10e-4),
            'bias_1': use_bias,
            'activation_2': stack_activation,
            # 'activity_regularizer_2': regularizers.l2(10e-4),
            'bias_2': use_bias,
            'optimizer': optimizer,
            'loss_function': 'binary_crossentropy',
            'epoch': greedy_epoch,
            'encode': False
        }
)