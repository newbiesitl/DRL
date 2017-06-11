from utils.utils import *
import os
from embedding.greedy_encoding import GreedyEncoder
from prototype.color_ae.model_config.model_config import *

cur_dir = os.path.dirname(__file__)
data_folder = os.path.join(cur_dir, 'data')

model_name = 'c_4000_2000_1000'
model_folder = os.path.join(cur_dir, 'models')
model_config = c_4000_2000_1000
ae = GreedyEncoder()
t =ImageTransformer()
t.configure(output_shape)
dat = t.transform_all(data_folder, grey_scale=False, batch_size=200)
dat = list(dat)[0]

ae.set_training_data(dat, dat)
ae.set_test_data(dat, dat)
ae.compile(use_bias=True)
ae.arch(model_config)
ae.set_batch_size(20)
ae.fit()
# for batch in dat:
#     print(batch.shape)

