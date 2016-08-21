import numpy
import theano

from cutils.numeric import numpy_floatX


def get_minibatches_idx(n, minibatch_size, shuffle=False, use_remaining=True):
    """
    Used to shuffle the dataset at each iteration
    """
    idx_list = numpy.arange(n, dtype="int32")
    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_size != n) and use_remaining:
        # Put the remaining samples in a minibatch
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def weight_decay(U, decay_c):
    """
    cost is a Theano expression
    U is a Theano variable
    decay_c is a scalar
    """
    #TODO: Assert the datatypes
    decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
    weight_decay = 0.
    weight_decay += (U ** 2).sum()
    weight_decay *= decay_c
