import os, sys
cur_dir = os.path.dirname(__file__)
project_root = os.path.join(cur_dir,'..','..')
sys.path.append(project_root)


from utils.utils import ImageTransformer
from embedding.greedy_encoding import GreedyEncoder
import numpy as np
from utils.utils import visualize_result_ae, visualize_result_encode_decode
from embedding.model_config import *
model_path = os.path.join(cur_dir, 'models')


def init_model(config, model_name, train=False, show_results=False, test_on_training = False, test_on_noise = False, data_set ='all', shuffle_samples = True, batch_size=2000):
    data_path = os.path.join(cur_dir, 'data', data_set)
    model_name = '_'.join([model_name, 'with-bias' if use_bias else 'without-bias', data_set])

    # create an transformer
    t = ImageTransformer()
    # configure the transformer
    t.configure(output_shape)
    ae = GreedyEncoder(verbose=True)
    if train:
        ae.compile(use_bias=use_bias)
        ae.arch(config)
    else:
        ae.load(model_path, model_name)
    for batch in t.transform_all(data_path, batch_size=batch_size):
        print('read data...', end='')
        print('done')
        divider = int(len(batch)*0.9)
        x_train = batch[:divider]
        x_test = batch[divider:]
        if train:
            ae.set_training_data(x_train, x_train)
            ae.set_test_data(x_test, x_test)
            ae.set_batch_size(500)
            ae.fit()
        break
    ae.save(model_path, model_name)
    if test_on_training:
        x_test = x_train
    if test_on_noise:
        x_test = np.random.uniform(0, 1, (len(x_test), output_shape[0]*output_shape[1]))
        x_test = np.zeros((len(x_test), output_shape[0]*output_shape[1]))
        x_test = np.ones((len(x_test), output_shape[0]*output_shape[1]))
        # print(x_test)
    num_images = min(len(x_test), 10)
    if show_results:
        visualize_result_ae(ae, x_test, output_shape, random_sample=shuffle_samples, number_images=num_images)
        # visualize_result_encode_decode(ae, x_test, output_shape, random_sample=False)
        # visualize_result_ae(ae, x_train, output_shape, random_sample=True)

# Driver file
if __name__ == '__main__':
    for session in [
        # (c_4000_2000_1000, 'c_4000_2000_1000'),
        # (c_4000_2000_300_2000_4000, 'c_4000_2000_300_2000_4000'),
        (c_4000_2000_1000_2000_4000, 'c_4000_2000_1000_2000_4000'),
        # (c_4000_2000_300, 'c_4000_2000_300'),
    ]:
        model_name, config = session
        init_model(model_name, config, train=True, show_results=True, batch_size=10000, shuffle_samples=True)