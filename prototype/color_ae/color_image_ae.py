from utils.utils import *
import os
from embedding.greedy_encoding import GreedyEncoder
from prototype.color_ae.model_config.model_config import *

cur_dir = os.path.dirname(__file__)
project_root = os.path.join(cur_dir, '..', '..')
data_folder = os.path.join(project_root, 'data', 'images', 'women')

'''
Trained:
c_2000_1000_300_1000_2000  #
c_128   # for testing
c_1000_128   # for testing
c_2000_1000_300
'''

model_config = c_2000_1000_300_1000_2000
model_folder_name = '_shape_'.join([model_config['name'], '_'.join([str(x) for x in output_shape])])
model_name = model_config['name']
model_folder = os.path.join(project_root, 'models', model_folder_name)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
# print(len(model_config['stack']))
# print(model_config)
# exit()
train_model = False



ae = GreedyEncoder(verbose=True)
t =ImageTransformer()
t.configure(output_shape)
data = []
for dat in t.transform_all(data_folder, grey_scale=False, batch_size=50000):
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
visualize_result_ae(ae, test, output_shape, color_img=True, random_sample=True, number_images=10)
# Next step is to visualize the result from prediction
# for batch in dat:
#     print(batch.shape)

