"""
**Authors:**
    yubo@cse.yorku.ca
"""
import scipy.misc
import os
from os import listdir
from os.path import isfile, join
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
import matplotlib.pyplot as plt

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
        super().__init__()

    def configure(self, output_shape):
        self.output_shape = output_shape

    def register_encoder(self, encoder):
        raise NotImplementedError()

    def transform_one(self, input):
        '''
        Input is file name or numpy array
        :param args:
        :param kwargs:
        :return:
        '''
        if type(input) is str:
            img = self._read_img(input)
        elif type(input) is np.array:
            img = input
        else:
            raise Exception('Unsupported img input type {0}, supported inputs are file_name & numpy.array')
        if img.shape == self.output_shape:
            return img
        return np.array(scipy.misc.imresize(img, self.output_shape))


    def transform_all(self, folder_name, flatten=True, batch_size=256, multi_thread=True):
        if self.output_shape is None:
            raise Exception('Please configure the transformer before use')
        if multi_thread:
            onlyfiles = [os.path.join(folder_name, f) for f in listdir(folder_name) if isfile(join(folder_name, f))]
            for chunk in chunks(onlyfiles, batch_size):
                data = self._read_data_parallel(chunk, gray_scale=flatten, batch_size=batch_size)
                yield data
        raise Exception('single thread is not supported')
        #return self._read_data(folder_name, gray_scale=flatten, batch_size=batch_size)


    def _read_data(self, onlyfiles, gray_scale=False, batch_size=256):
        batch = []
        count = 0
        for file_path in onlyfiles:
            # we are able to read png
            if file_path[-3:] not in  ['jpg', 'png', 'jpeg']:
                continue
            img = ImageTransformer._read_img(file_path, gray_scale=gray_scale)
            img = np.array(scipy.misc.imresize(img, self.output_shape))
            batch.append(img)
            count += 1
        buf = np.array(batch)
        buf = buf.reshape((len(buf), np.prod(buf.shape[1:])))
        buf = buf.astype('float32') / 255.
        return buf

    def _read_data_parallel(self, onlyfiles, gray_scale=False, batch_size=256):
        patch = zip(onlyfiles, [gray_scale]*len(onlyfiles), [self.output_shape]*len(onlyfiles))
        from multiprocessing import Pool
        p = Pool(8)
        batch = p.map(ImageTransformer._read_file_worker, patch)
        p.close()
        p.join()
        batch = [x for x in batch if x is not None]
        batch = np.array(batch)
        batch = batch.reshape((len(batch), np.prod(batch.shape[1:])))
        batch = batch.astype('float32') / 255.
        return batch

    @staticmethod
    def _read_file_worker(patch):
        file_path, gray_scale, output_shape = patch
        if file_path[-3:] not in ['jpg', 'png', 'jpeg']:
            return
        img = ImageTransformer._read_img(file_path, gray_scale=gray_scale)
        img = np.array(scipy.misc.imresize(img, output_shape))
        img = img.reshape((len(img), np.prod(img.shape[1:])))
        img = img.astype('float32') / 255.
        return img

    # read_img: read image, and convert to numpy
    @staticmethod
    def _read_img(img_filename, gray_scale=False):
        img = np.array(scipy.misc.imread(img_filename, flatten=gray_scale))
        return img




def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def visualize_result_ae(ae, test, shape, random_sample=True, number_images=10):
    # print(dir(encoder[0]))
    # print(encoder[0].get_weights())
    # exit()
    # encode and decode some digits
    # note that we take them from the *test* set


    decoded_imgs = ae.encode_decode(test)
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
        original_matrix = np.resize(original_matrix, shape)
        plt.imshow(original_matrix)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        # threshold the data
        decoded_matrix = decoded_imgs[indices[i]]
        decoded_matrix = np.resize(decoded_matrix, shape)
        plt.imshow(decoded_matrix)

        plt.gray()
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