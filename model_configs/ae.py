import keras
output_shape = (60, 40)
# output_shape = (28, 28)
color_img = True
if color_img:
    input_dim = output_shape[1] * output_shape[0] * 3
else:
    input_dim = output_shape[1] * output_shape[0]
use_bias = True
greedy_epoch = 2
fine_tuning_epoch = 6
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
    'loss_function': 'binary_crossentropy',
    'name': 'c_128'
}

c_1000_128 = {
    'stack':[
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
        },
        {
            'input_dimension': 1024,
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
        },
    ],
    'epoch': fine_tuning_epoch,
    'optimizer': optimizer,
    'loss_function': 'binary_crossentropy',
    'name': 'c_1000_128'
}



c_2000_1000 = {
    'name': 'c_2000_1000',
    'stack':[
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
    ],
    'epoch': fine_tuning_epoch,
    'optimizer': optimizer,
    'loss_function': 'binary_crossentropy'
}

c_2000_1000_300 = {
    'name': 'c_2000_1000_300',
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

c_2000_1000_500 = {
    'name': 'c_2000_1000_500',
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
            'embedding_dimension': 500,
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



c_2000_1000_300_1000_2000 = {
    'name': 'c_2000_1000_300_1000_2000',
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
        {
            'input_dimension': 300,
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

    ],
    'epoch': fine_tuning_epoch,
    'optimizer': optimizer,
    'loss_function': 'binary_crossentropy'
}


c_2000_1000_800_1000_2000 = {
    'name': 'c_2000_1000_800_1000_2000',
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
            'embedding_dimension': 800,
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
        {
            'input_dimension': 800,
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

    ],
    'epoch': fine_tuning_epoch,
    'optimizer': optimizer,
    'loss_function': 'binary_crossentropy'
}