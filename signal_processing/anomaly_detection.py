from signal_processing import auto_encoder

from itertools import islice
from sklearn.metrics import mean_absolute_error
import numpy as np
def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result



def serialize(seq, window_size, encoder, threshold):
    '''
    Convert inputs to number sequence, use sliding window to slide through signal and pass window to autoencoder, autoencoder reconstruct the inputs and report reconstruction error, compare reconstruction error with threshold, if error is bigger than threshold, mark the signal on the plot.
    :param seq: the input sequence
    :return:
    '''
    mark = []
    index = 0
    for x in window(seq, n=window_size):
        x_hat = encoder.predict(x)
        index += 1
        # re is reconstruction error
        re = mean_absolute_error(x, x_hat)
        avg_re = np.average(re)
        if avg_re > threshold:
            mark.append(index)



