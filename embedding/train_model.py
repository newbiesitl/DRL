import os, sys
cur_dir = os.path.dirname(__file__)
project_root = os.path.join(cur_dir, '..', '..')
sys.path.append(project_root)


from utils.utils import ImageTransformer
from embedding.greedy_encoding import AutoEncoder
import numpy as np
from utils.utils import visualize_result_ae, visualize_result_encode_decode
from embedding.model_config import *
model_path = os.path.join(cur_dir, 'models')
data_path = os.path.join(cur_dir, 'data', 'all')

def main():
    train = False
    test_on_training = False
    test_on_noise = False
    viz_result = True
    model_name = 'c_4000_2000_1000_2000_4000'
    random_sample = True
    training_set_size = 50
    config = f_4000_2000_1000_2000_4000

    # create an transformer
    t = ImageTransformer()
    # configure the transformer
    t.configure(output_shape)

    ae = AutoEncoder(verbose=True)
    data = None
    for batch in t.transform_all(data_path, batch_size=training_set_size):
        print('read data...', end='')
        data = batch
        print('done')
        break

    divider = int(len(data)*0.9)
    x_train = data[:divider]
    x_test = data[divider:]
    if train:
        ae.set_training_data(x_train, x_train)
        ae.set_test_data(x_test, x_test)
        ae.set_batch_size(500)
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