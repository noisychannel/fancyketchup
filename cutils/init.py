import warnings
import numpy
import theano.tensor as T

from numeric import numpy_floatX


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
