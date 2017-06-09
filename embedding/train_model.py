import os, sys
cur_dir = os.path.dirname(__file__)
project_root = os.path.join(cur_dir,'..','..')
sys.path.append(project_root)


from utils.utils import ImageTransformer
from embedding.greedy_encoding import GreedyEncoder
import numpy as np
from utils.utils import visualize_result_ae, visualize_result_encode_decode
from embedding.model_config import toy_network_config, mid_size_network_config, one_layer_network_config, output_shape, large_network_config
model_path = os.path.join(cur_dir, 'models')

def main():
    train = True
    test_on_training = True
    use_bias = False
    test_on_noise = False
    viz_result = True
    data_set = 'all'
    data_path = os.path.join(cur_dir, 'data', data_set)

    model_name = '_'.join(['toy_network', data_set])
    # model_name = '10K_sample_2056_1024_2056_40_100'
    random_sample = True
    training_set_size = 10000
    # config = toy_network_config
    # config = one_layer_network_config
    config = large_network_config

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
    # if True:
    #     from keras.datasets import mnist
    #     import numpy as np
    #     (x_train, _), (x_test, _) = mnist.load_data()
    #     x_train = x_train.astype('float32') / 255.
    #     x_test = x_test.astype('float32') / 255.
    #     x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))[:1000]
    #     x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))[:1000]
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