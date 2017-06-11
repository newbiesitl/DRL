from utils.utils import *
import os
from embedding.greedy_encoding import GreedyEncoder

cur_dir = os.path.dirname(__file__)
data_folder = os.path.join(cur_dir, 'data')

model_name = 'c_4000_2000_1000'
model_folder = os.path.join(cur_dir, 'models')

ae = GreedyEncoder()
ae.load(os.path.join(model_folder, model_name), model_name)
t =ImageTransformer()
t.configure((80,60))
dat = t.transform_all(data_folder, grey_scale=False, batch_size=200)
for batch in dat:
    print(batch.shape)

