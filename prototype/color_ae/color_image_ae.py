from utils.utils import *
import os
from embedding.greedy_encoding import GreedyEncoder
from prototype.color_ae.model_config.model_config import *

cur_dir = os.path.dirname(__file__)
data_folder = os.path.join(cur_dir, 'data')

model_name = 'c_1000_128'
model_folder = os.path.join(cur_dir, 'models')
model_config = c_1000_128
train = True

ae = GreedyEncoder(verbose=True)
t =ImageTransformer()
t.configure(output_shape)
dat = t.transform_all(data_folder, grey_scale=False, batch_size=2)
dat = list(dat)[0] # get the first batch from generator
divider = int(len(dat)*0.9)
train = dat[:divider]
test = dat[divider:]
ae.set_training_data(train, train)
ae.set_test_data(test, test)
ae.compile(use_bias=True)
ae.arch(model_config)
ae.set_batch_size(20)
if train:
    ae.fit()
    ae.save(model_folder, model_name)
else:
    ae.load(model_folder, model_name)
visualize_result_ae(ae, dat, output_shape, color_img=True, random_sample=True, number_images=10)
# Next step is to visualize the result from prediction
# for batch in dat:
#     print(batch.shape)

