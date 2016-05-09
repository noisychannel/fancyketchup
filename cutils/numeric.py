import numpy
import theano


def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)
