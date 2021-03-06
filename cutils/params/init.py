import warnings
import numpy
import theano
import theano.tensor as T

from cutils.numeric import numpy_floatX


def xavier_init(rng, n_in, n_out, activation, size=None):
    """
    Returns a matrix (n_in X n_out) based on the
    Xavier initialization technique
    """

    if activation not in [T.tanh, T.nnet.sigmoid, T.nnet.relu]:
        warnings.warn("You are using the Xavier init with an \
                       activation function that is not sigmoidal or relu")
    # Default value for size
    if size is None:
        size = (n_in, n_out)
    W_values = numpy_floatX(
        rng.uniform(
            low=-numpy.sqrt(6. / (n_in + n_out)),
            high=numpy.sqrt(6. / (n_in + n_out)),
            size=size,
        ))
    if activation == T.nnet.sigmoid:
        return W_values * 4
    if activation == T.nnet.relu:
        return W_values * numpy.sqrt(2.)
    return W_values


def norm_init(n_in, n_out, scale=0.01, ortho=True):
    """
    Initialize weights from a scaled standard normal distribution
    Falls back to orthogonal weights if n_in = n_out

    n_in : The input dimension
    n_out : The output dimension
    scale : Scale for the normal distribution
    ortho : Fall back to ortho weights when n_in = n_out
    """
    if n_in == n_out and ortho:
        return ortho_weight(n_in)
    else:
        return numpy_floatX(scale * numpy.random.randn(n_in, n_out))


def ortho_weight(ndim):
    """
    Returns an orthogonal matrix via SVD decomp
    Used for initializing weight matrices of an LSTM
    """
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(theano.config.floatX)
