import numpy
import theano
import theano.tensor as T


def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)


def safe_log(x, min_val=0.0000000001):
    #return T.log(x + 1e-10)
    # TODO: Clip seems to trigger HostFromGPU
    return T.log(T.clip(x, numpy.float32(1e-10), numpy.float32(1e10)))
