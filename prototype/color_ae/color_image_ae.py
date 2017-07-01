from utils.utils import *
import os
from embedding.greedy_encoding import GreedyEncoder
from model_configs.model_config import *


cur_dir = os.path.dirname(__file__)
project_root = os.path.join(cur_dir, '..', '..')
data_folder = os.path.join(project_root, 'data', 'images', 'all')

'''
Trained:
c_2000_1000_300_1000_2000  # working with 40 k 60*40 , next step is to train 100 * 80
c_2000_1000_800_1000_2000  # compare this with 300 hidden
c_128   # for testing
c_1000_128   # for testing
c_2000_1000_300
'''

model_config = c_2000_1000_300
model_folder_name = model_config['name']
# model_folder_name = '_shape_'.join([model_config['name'], '_'.join([str(x) for x in output_shape])])
# model_folder_name = '_'.join([model_folder_name, activation_1])
model_name = model_config['name']
model_folder = os.path.join(project_root, 'models', 'vision', model_folder_name)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
# print(len(model_config['stack']))
# print(model_config)
# exit()
# train_model = True
train_model = True



ae = GreedyEncoder(verbose=True)
t =ImageTransformer()
t.configure(output_shape)
data = []

for dat in t.transform_all(data_folder, grey_scale=False, batch_size=5000 if train_model is False else 50000, flatten=True):
    data = dat # get the first batch from generator
    break
divider = int(len(data)*0.9)
test = data[divider:]
train = data[:divider]
ae.set_training_data(train, train)
ae.set_test_data(test, test)
ae.compile(use_bias=True)
ae.arch(model_config)
ae.set_batch_size(200)
if train_model:
    ae.fit()
    ae.save(model_folder, model_name)
else:
    ae.load(model_folder, model_name)
print('finish loading model.')

visualize_result_ae(ae, test, output_shape, color_img=True, random_sample=True, number_images=10)

