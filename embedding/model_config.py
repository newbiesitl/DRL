import keras
output_shape = (60, 40)
# output_shape = (28, 28)
input_dim = output_shape[1] * output_shape[0]
use_bias = True
greedy_epoch = 20
fine_tuning_epoch = 40
activation_1 = 'relu'
activation_2 = 'sigmoid'
optimizer = keras.optimizers.Adadelta()

large_network_config = {
    'stack': [
        {
            'input_dimension': input_dim,
            'output_dimension': input_dim,
            'embedding_dimension': 4000,
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
            'input_dimension': 4000,
            'output_dimension': input_dim,
            'embedding_dimension': 2000,
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
            'input_dimension': 2000,
            'output_dimension': input_dim,
            'embedding_dimension': 300,
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
            'input_dimension': 300,
            'output_dimension': input_dim,
            'embedding_dimension': 2000,
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
            'input_dimension': 2000,
            'output_dimension': input_dim,
            'embedding_dimension': 4000,
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


mid_size_network_config = {
    'stack': [
        {
            'input_dimension': input_dim,
            'output_dimension': input_dim,
            'embedding_dimension': 2056,
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
            'input_dimension': 2056,
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
            'embedding_dimension': 2056,
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

toy_network_config = {
    'stack': [
        {
            'input_dimension': input_dim,
            'output_dimension': input_dim,
            'embedding_dimension': 512,
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
            'input_dimension': 512,
            'output_dimension': input_dim,
            'embedding_dimension': 256,
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
            'input_dimension': 256,
            'output_dimension': input_dim,
            'embedding_dimension': 512,
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

one_layer_network_config = {
    'stack': [
        {
            'input_dimension': input_dim,
            'output_dimension': input_dim,
            'embedding_dimension': 200,
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