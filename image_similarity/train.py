import os, sys
cur_dir = os.path.dirname(__file__)
project_root = os.path.join(cur_dir,'..','..')
sys.path.append(project_root)


from image_similarity.utils import  ImageTransformer
from unsupervised.greedy_encoding import GreedyEncoder
from keras.layers import regularizers
import numpy as np
import keras
from image_similarity.utils import visualize_result_ae, visualize_result_encode_decode
model_path = os.path.join(cur_dir, 'models')
# data_path = os.path.join(cur_dir, 'data', 'men')
data_path = os.path.join(cur_dir, 'data', 'women')

def main():
    output_shape = (60, 40)
    # output_shape = (28, 28)
    input_dim = output_shape[1] * output_shape[0]
    train = True
    test_on_training = False
    use_bias = True
    test_on_noise = False
    viz_result = False
    greedy_epoch = 40
    fine_tuning_epoch = 100
    model_name = '10K_sample_5000_3000_2000_3000_5000'
    random_sample = True
    training_set_size = 40000
    config = {
        'stack': [
            {
                'input_dimension': input_dim,
                'output_dimension': input_dim,
                'embedding_dimension': 5000,
                'activation_1': 'linear',
                # 'activity_regularizer_1': regularizers.l2(10e-4),
                'bias_1': use_bias,
                'activation_2': 'sigmoid',
                # 'activity_regularizer_2': regularizers.l2(10e-4),
                'bias_2': use_bias,
                'optimizer': keras.optimizers.Adam(),
                'loss_function': 'binary_crossentropy',
                'epoch': greedy_epoch
            },
            {
                'input_dimension': 5000,
                'output_dimension': input_dim,
                'embedding_dimension': 3000,
                'activation_1': 'linear',
                # 'activity_regularizer_1': regularizers.l2(10e-4),
                'bias_1': use_bias,
                'activation_2': 'sigmoid',
                # 'activity_regularizer_2': regularizers.l2(10e-4),
                'bias_2': use_bias,
                'optimizer': keras.optimizers.Adam(),
                'loss_function': 'binary_crossentropy',
                'epoch': greedy_epoch
            },
            {
                'input_dimension': 3000,
                'output_dimension': input_dim,
                'embedding_dimension': 2000,
                'activation_1': 'linear',
                # 'activity_regularizer_1': regularizers.l2(10e-4),
                'bias_1': use_bias,
                'activation_2': 'sigmoid',
                # 'activity_regularizer_2': regularizers.l2(10e-4),
                'bias_2': use_bias,
                'optimizer': keras.optimizers.Adam(),
                'loss_function': 'binary_crossentropy',
                'epoch': greedy_epoch
            },
            {
                'input_dimension': 2000,
                'output_dimension': input_dim,
                'embedding_dimension': 3000,
                'activation_1': 'linear',
                # 'activity_regularizer_1': regularizers.l2(10e-4),
                'bias_1': use_bias,
                'activation_2': 'sigmoid',
                # 'activity_regularizer_2': regularizers.l2(10e-4),
                'bias_2': use_bias,
                'optimizer': keras.optimizers.Adam(),
                'loss_function': 'binary_crossentropy',
                'epoch': greedy_epoch
            },
            {
                'input_dimension': 3000,
                'output_dimension': input_dim,
                'embedding_dimension': 5000,
                'activation_1': 'linear',
                # 'activity_regularizer_1': regularizers.l2(10e-4),
                'bias_1': use_bias,
                'activation_2': 'sigmoid',
                # 'activity_regularizer_2': regularizers.l2(10e-4),
                'bias_2': use_bias,
                'optimizer': keras.optimizers.Adam(),
                'loss_function': 'binary_crossentropy',
                'epoch': greedy_epoch
            },
        ],
        'epoch': fine_tuning_epoch,
        'optimizer': keras.optimizers.Adam(),
        'loss_function': 'binary_crossentropy'
    }

    # create an transformer
    t = ImageTransformer()
    # configure the transformer
    t.configure(output_shape)

    ae = GreedyEncoder(verbose=True)
    data = None
    for batch in t.transform_all(data_path, batch_size=training_set_size):
        print('read data...', end='')
        data = batch
        print('done')
        break

    divider = int(len(data)*0.8)
    x_train = data[:divider]
    x_test = data[divider:]
    if train:
        ae.set_training_data(x_train, x_train)
        ae.set_test_data(x_test, x_test)
        ae.set_batch_size(256)
        ae.compile(use_bias=use_bias)
        ae.arch(config)
        ae.fit()
        ae.save(model_path, model_name)
    else:
        ae.load(model_path, model_name)
    if test_on_training:
        x_test = x_train
    if test_on_noise:
        x_test = np.random.uniform(0, 1, (len(x_test), output_shape[0]*output_shape[1]))
        x_test = np.zeros((len(x_test), output_shape[0]*output_shape[1]))
        x_test = np.ones((len(x_test), output_shape[0]*output_shape[1]))
        # print(x_test)
    num_images = min(len(x_test), 10)
    if viz_result:
        visualize_result_ae(ae, x_test, output_shape, random_sample=random_sample, number_images=num_images)
        # visualize_result_encode_decode(ae, x_test, output_shape, random_sample=False)
        # visualize_result_ae(ae, x_train, output_shape, random_sample=True)

# Driver file
if __name__ == '__main__':
    main()