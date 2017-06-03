from keras import backend as K
from keras.engine.training import _weighted_masked_objective

# def _weighted_masked_objective(fn):
#     """Adds support for masking and sample-weighting to an objective function.
#     It transforms an objective function `fn(y_true, y_pred)`
#     into a sample-weighted, cost-masked objective function
#     `fn(y_true, y_pred, weights, mask)`.
#     # Arguments
#         fn: The objective function to wrap,
#             with signature `fn(y_true, y_pred)`.
#     # Returns
#         A function with signature `fn(y_true, y_pred, weights, mask)`.
#     """
#     if fn is None:
#         return None
#
#     def weighted(y_true, y_pred, weights, mask=None):
#         """Wrapper function.
#         # Arguments
#             y_true: `y_true` argument of `fn`.
#             y_pred: `y_pred` argument of `fn`.
#             weights: Weights tensor.
#             mask: Mask tensor.
#         # Returns
#             Scalar tensor.
#         """
#         # score_array has ndim >= 2
#         score_array = fn(y_true, y_pred)
#         if mask is not None:
#             # Cast the mask to floatX to avoid float64 upcasting in theano
#             mask = K.cast(mask, K.floatx())
#             # mask should have the same shape as score_array
#             score_array *= mask
#             #  the loss per batch should be proportional
#             #  to the number of unmasked samples.
#             score_array /= K.mean(mask)
#
#         # reduce score_array to same ndim as weight array
#         ndim = K.ndim(score_array)
#         weight_ndim = K.ndim(weights)
#         score_array = K.mean(score_array, axis=list(range(weight_ndim, ndim)))
#
#         # apply sample weighting
#         if weights is not None:
#             score_array *= weights
#             score_array /= K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
#         return K.mean(score_array)
#     return weighted
