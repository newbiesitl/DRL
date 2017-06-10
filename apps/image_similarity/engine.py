from utils.utils import ImageTransformer
import numpy as np
import os
from os import listdir
from os.path import isfile, join

class DataManager(object):
    def __init__(self):
        self.encoder = None
        self.counter = 0
        self.output_shape = None
        self.bin_db_path = None
        self.db_name = None
        self.raw_db_paths = None
        self.f_idx = {}
        self.idx_f = {}

    def get_file_name(self, idx):
        return self.idx_f[idx]

    def get_index(self, f_name):
        return self.f_idx[f_name]

    def build_mapping(self):
        if self.raw_db_paths is None:
            raise Exception('Config before use')
        mat_file_list = []
        for each_path in self.raw_db_paths:
            mat_file_list += [os.path.join(each_path, f) for f in listdir(each_path) if f[-3:]=='jpg' and isfile(join(each_path, f))]
        self.f_idx = dict(zip(mat_file_list, range(len(mat_file_list))))
        self.idx_f = dict(zip(range(len(mat_file_list)), mat_file_list))


    def configure(self, config_dict):
        self.bin_db_path = config_dict['bin_db_path']
        self.output_shape = config_dict['output_shape']
        self.db_name = config_dict['db_name']
        self.raw_db_paths = config_dict['raw_db_paths']


    def register_encoder(self, encoder=None):
        self.encoder = encoder

    def load_raw_data(self, batch_size=5000):
        self.counter = 0
        if isinstance(self.raw_db_paths, str):
            self._img_to_numpy_array(self.raw_db_paths, self.output_shape, self.bin_db_path, self.db_name, batch_size)
        elif isinstance(self.raw_db_paths, list):
            for each_folder in self.raw_db_paths:
                self._img_to_numpy_array(each_folder, self.output_shape, self.bin_db_path, self.db_name,
                                         batch_size)
        else:
            raise Exception('Unsupported format, only supports str and list (list of paths)')

    def _img_to_numpy_array(self, folder_path, output_shape, save_folder_path, db_name, size=5000):
        t = ImageTransformer()
        t.configure(output_shape)
        for chunk in t.transform_all(folder_path, flatten=True, batch_size=size, multi_thread=True):
            if self.encoder:
                ret = self.encoder.encode(chunk)
            else:
                ret = chunk
            np.save(os.path.join(self.bin_db_path, '_'.join([db_name, str(self.counter)])), ret)
            self.counter += 1

    def load_dataset(self):
        mat_file_list = [os.path.join(self.bin_db_path, f) for f in listdir(self.bin_db_path) if isfile(join(self.bin_db_path, f))]
        all = None
        for mat_file in mat_file_list:
            ds = np.load(mat_file)
            if all is None:
                all = ds
            else:
                all = np.concatenate((all, ds))
        return all
