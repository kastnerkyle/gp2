import numpy as np
import scipy.spatial.distance as spdist
from sklearn.metrics.pairwise import euclidean_distances
from scipy import signal as spsig
from scipy import spatial as spat


# from mrmartin.ner/?p=223
def exponential_kernel(x1, x2):
    # Broadcasting tricks to get every pairwise distance.
    return np.exp(-(x1[np.newaxis, :, :] - x2[:, np.newaxis, :])[:, :, 0] ** 2).T


# From fast GP NASA paper
def ornstein_uhlenbeck_kernel(x1, x2):
    # Broadcasting tricks to get every pairwise distance.
    return np.exp(-(x1[:, np.newaxis, :] - x2[np.newaxis, :, :])[:, :, 0])


# Initialization based on initSMhypers from
# http://mlg.eng.cam.ac.uk/andrew/pattern/patterndemo2.zip
def init_spectral_mixture_kernel(x1, x2, n_components=10):
    raise ValueError("Very Broken(TM)")
    rng = np.random.RandomState(1999)
    n_features = x1.shape[1]
    weights = np.ones((1, n_components)) * x2.std() / n_components
    means = np.zeros((n_features, n_components))
    variances = np.zeros((n_features, n_components))
    for i in range(n_features):
        feature_distances = euclidean_distances(x1[:, i].reshape((-1, 1)))
        feature_distances[feature_distances == 0] = 1.
        means[i, :] = (.5 / feature_distances.min()) * rng.rand(
            *means[i, :].shape)
        variances[i, :] = 1. / np.abs(feature_distances.max() *
                                      rng.rand(*variances[i, :].shape))
    return weights, means, variances


# http://mlg.eng.cam.ac.uk/andrew/pattern/
# https://github.com/hadsed/PyGPML/blob/master/kernels.py
# https://github.com/marionmari/pyGPs/blob/master/pyGPs/Core/cov.py
def spectral_mixture_kernel(x, z, weights, means, variances):
    raise ValueError("Very Broken(TM)")
    n, D = x.shape
    Q = means.shape[1]
    w = weights
    m = means
    v = variances
    d2 = np.zeros((n, z.shape[0], D))
    for i in range(D):
        d2[:, :, i] = spdist.cdist(x[:, i].reshape((-1, 1)),
                                   z[:, i].reshape((-1, 1)),
                                   'sqeuclidean')
    d = np.sqrt(d2)

    def k(d2v, dm):
        return np.exp(-2 * np.pi ** 2 * d2v) * np.cos(2 * np.pi * dm)
    A = 0.
    for q in range(Q):
        C = w[0, q] ** 2
        for j in range(D):
            C = C * k(d2[:, :, j] * v[j, q], d[:, :, j] * m[j, q])
        A = A + C
    return A
