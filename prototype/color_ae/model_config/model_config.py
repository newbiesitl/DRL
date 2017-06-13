import keras
output_shape = (60, 40)
# output_shape = (28, 28)
color_img = True
if color_img:
    input_dim = output_shape[1] * output_shape[0] * 3
else:
    input_dim = output_shape[1] * output_shape[0]
use_bias = True
greedy_epoch = 10
fine_tuning_epoch = 20
activation_1 = 'linear'
activation_2 = 'sigmoid'
stack_activation = 'sigmoid'
optimizer = keras.optimizers.Adadelta()
# optimizer = keras.optimizers.Adagrad()


c_128 = {
    'stack':[
        {
            'input_dimension': input_dim,
            'output_dimension': input_dim,
            'embedding_dimension': 128,
            'activation_1': activation_1,
            # 'activity_regularizer_1': regularizers.l2(10e-4),
            'bias_1': use_bias,
            'activation_2': activation_2,
            # 'activity_regularizer_2': regularizers.l2(10e-4),
            'bias_2': use_bias,
            'optimizer': optimizer,
            'loss_function': 'binary_crossentropy',
            'epoch': greedy_epoch
        }
    ],
    'epoch': fine_tuning_epoch,
    'optimizer': optimizer,
    'loss_function': 'binary_crossentropy'
}

c_1000_128 = c_128
c_1000_128['stack'].insert(0,
{
    'input_dimension': input_dim,
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
    'epoch': greedy_epoch
}
)
c_1000_128['stack'][1]['input_dimension'] = 1024

c_2000_1000 = c_1000_128
c_2000_1000['stack'][0]['embedding_dimension'] = 2000
c_2000_1000['stack'][1]['embedding_dimension'] = 1000
c_2000_1000['stack'][1]['input_dimension'] = 2000

c_2000_1000_300 = {
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
            'epoch': greedy_epoch
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
            'epoch': greedy_epoch
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
            'epoch': greedy_epoch
        },
    ],
    'epoch': fine_tuning_epoch,
    'optimizer': optimizer,
    'loss_function': 'binary_crossentropy'
}

c_2000_1000_500 = c_2000_1000_300
c_2000_1000_500['stack'][-1]['embedding_dimension'] = 500


c_4000_2000_1000 = c_2000_1000_300
c_4000_2000_1000['stack'][0]['embedding_dimension'] = 4000
c_4000_2000_1000['stack'][1]['input_dimension'] = 4000
c_4000_2000_1000['stack'][1]['embedding_dimension'] = 2000
c_4000_2000_1000['stack'][-1]['input_dimension'] = 2000
c_4000_2000_1000['stack'][-1]['embedding_dimension'] = 1000


c_4000_2000_1000_2000_4000 = c_4000_2000_1000
c_4000_2000_1000_2000_4000['stack'].append(
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
            'epoch': greedy_epoch
        }
)

c_4000_2000_1000_2000_4000['stack'].append(
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
            'epoch': greedy_epoch
        }
)