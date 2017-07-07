"""
**Authors:**
    yubo@cse.yorku.ca
"""
import scipy.misc, os
from os import listdir
from os.path import isfile, join
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool


class TransformerBase(ABC):
    @abstractmethod
    def transform_all(self, iterable):
        pass

    @abstractmethod
    def transform_one(self, single):
        pass

    @abstractmethod
    def configure(self, **kwargs):
        pass


class ImageTransformer(TransformerBase):
    def __init__(self):
        self.output_shape = None
        self.encoder = None
        super().__init__()

    def configure(self, output_shape):
        self.output_shape = output_shape

    def register_encoder(self, encoder=None):
        self.encoder = encoder

    def transform_one(self, file_path, grey_scale=True):
        '''
        Input is file name or numpy array
        :param args:
        :param kwargs:
        :return:
        '''
        if self.output_shape is None:
            raise Exception("output shape is not configured")
        ret = self._read_data_parallel([file_path], gray_scale=grey_scale)
        if self.encoder is not None:
            return self.encoder.encode(ret)
        return ret

    def transform_many(self, file_path_list, grey_scale=False, flatten=False):
        '''
        Input is file name or numpy array
        :param args:
        :param kwargs:
        :return:
        '''
        if self.output_shape is None:
            raise Exception("output shape is not configured")
        ret = self._read_data_parallel(file_path_list,  gray_scale=grey_scale, flatten=flatten)
        if self.encoder is not None:
            return self.encoder.encode(ret)
        return ret

    def transform_all(self, folder_name, grey_scale=False, batch_size=256, multi_thread=True, flatten=False):
        if self.output_shape is None:
            raise Exception("output shape is not configured")
        if multi_thread:
            onlyfiles = [os.path.join(folder_name, f) for f in listdir(folder_name) if isfile(join(folder_name, f))]
            for chunk in chunks(onlyfiles, batch_size):
                data = self._read_data_parallel(chunk, gray_scale=grey_scale, batch_size=batch_size, flatten=flatten)
                if self.encoder is None:
                    yield data
                else:
                    ret = self.encoder.encode(data)
                    yield ret


    def _read_data_parallel(self, onlyfiles, gray_scale=False, flatten=False, batch_size=256):
        patch = zip(onlyfiles, [gray_scale]*len(onlyfiles), [self.output_shape]*len(onlyfiles))
        p = Pool(8)
        batch = p.map(ImageTransformer._read_file_worker, patch)
        p.close()
        p.join()
        # nothing being removed ...
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            raise FileNotFoundError("No query file found, did you add query files?")
        batch = np.array(batch)
        if flatten:
            # if gray scale, flatten to 1-d array
            batch = batch.reshape((len(batch), np.prod(batch.shape[1:])))
        batch = batch.astype('float32') / 255.
        return batch

    @staticmethod
    def _read_file_worker(patch):
        file_path, gray_scale, output_shape = patch
        if file_path[-3:] not in ['jpg', 'png', 'jpeg']:
            return None
        img = ImageTransformer._read_img(file_path, gray_scale=gray_scale)
        img = np.array(scipy.misc.imresize(img, output_shape))
        return img

    # read_img: read image, and convert to numpy
    @staticmethod
    def _read_img(img_filename, gray_scale=False):
        if gray_scale:
            img = np.array(scipy.misc.imread(img_filename, flatten=gray_scale))
        else:
            img = np.array(scipy.misc.imread(img_filename, mode='RGB'))
        return img




def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def visualize_result_ae(ae, test, shape, random_sample=True, number_images=10, color_img=False):
    # print(dir(encoder[0]))
    # print(encoder[0].get_weights())
    # exit()
    # encode and decode some digits
    # note that we take them from the *test* set

    try:
        decoded_imgs = ae.encode_decode(test)
    except AttributeError:
        decoded_imgs = ae.predict(test)
    print(test.shape)
    print(decoded_imgs.shape)
    print(type(decoded_imgs), type(decoded_imgs[0]))
    # exit()

    # use Matplotlib (don't ask)

    n = number_images  # how many digits we will display
    if random_sample:
        indices = [random.randint(0, len(test)-1) for _ in range(n)]
    else:
        indices = range(n)
    plt.figure(figsize=(30, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        # plt.imshow(test[i].reshape(dim_x, dim_x))
        original_matrix = test[indices[i]]
        if color_img:
            original_matrix = np.resize(original_matrix, (shape[0], shape[1], 3))
            plt.imshow(original_matrix, interpolation='nearest')
        else:
            original_matrix = np.resize(original_matrix, shape)
            plt.imshow(original_matrix)
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        # threshold the data
        decoded_matrix = decoded_imgs[indices[i]]

        if color_img:
            decoded_matrix = np.resize(decoded_matrix, (shape[0], shape[1], 3))
            plt.imshow(decoded_matrix, interpolation='nearest')
        else:
            decoded_matrix = np.resize(decoded_matrix, shape)
            plt.imshow(decoded_matrix)

        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def visualize_result_encode_decode(ae, test, shape, random_sample=True):
    # print(dir(encoder[0]))
    # print(encoder[0].get_weights())
    # exit()
    # encode and decode some digits
    # note that we take them from the *test* set

    encoded_imgs = ae.encode(test)
    decoded_imgs = ae.decode(encoded_imgs)
    print(test.shape)
    print(decoded_imgs.shape)
    print(type(decoded_imgs), type(decoded_imgs[0]))
    # exit()

    # use Matplotlib (don't ask)

    n = 10  # how many digits we will display
    if random_sample:
        indices = [random.randint(0, len(test)-1) for _ in range(n)]
    else:
        indices = range(n)
    plt.figure(figsize=(30, 4))
    avg = []
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        # plt.imshow(test[i].reshape(dim_x, dim_x))
        original_matrix = test[indices[i]]
        original_matrix = np.resize(original_matrix, shape)
        plt.imshow(original_matrix)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        # threshold the data
        decoded_matrix = decoded_imgs[indices[i]]
        decoded_matrix = np.resize(decoded_matrix, (40,40))
        plt.imshow(decoded_matrix)

        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    ret = cosine_similarity(encoded_imgs[indices])
    print(ret)
    plt.show()