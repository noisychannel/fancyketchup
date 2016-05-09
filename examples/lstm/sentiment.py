"""
Build a tweet sentiment analyzer
"""

import numpy
import theano
import theano.tensor as T
from collections import OrderedDict

from cutils.numeric import numpy_floatX

import imdb


datasets = {'imdb': (imdb.load_data, imdb.prepare_data)}

SEED = 123
numpy.random.seed(SEED)

#TODO: move
def get_minibatches_idx(n, minibatch_size, shuffle=False):
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

    if (minibatch_size != n):
        # Put the remaining samples in a minibatch
        minibatches.append(idx_list[minibatch_start:])

def get_dataset(name):
    return datasets[name][0], datasets[name][1]

def zipp(params, tparams):
    """
    Reloading model, needed for the GPU
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)

def unzip(zipped):
    """
    When we pickle the model, needed for the GPU
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

def dropout_layer(state_before, use_noise, trng):
    #TODO: move
    #TODO: use the noise function that exists somewhere
    proj = T.switch(use_noise,
                    (state_before *
                     trng.binomial(state_before.shape,
                                   p=0.5, n=1,
                                   dtype=state_before.dtype)),
                     state_before * 0.5)
    return proj

def _p(pp, name):
    return '%s_%s' % (pp, name)

def init_params(options):
    """
    Embedding and classifier params
    """
    params = OrderedDict()
    # Embedding
    # TODO:These params can be auto initialized, via dicts
    randn = numpy.random.rand(options['n_words'],
                              options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype(theano.config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(theano.config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(theano.config.floatX)
    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp

    return params

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def get_layer(name):
    fns = layers[name]
    return fns

def ortho_weight(ndim):
    #TODO: Move
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(theano.config.floatX)

def param_init_lstm(options, params, prefix='lstm'):
    """
    Initialize the LSTM params
    """
    W = numpy.concatenate(
            [ortho_weight(options['dim_proj']),
             ortho_weight(options['dim_proj']),
             ortho_weight(options['dim_proj']),
             ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W

    #TODO: why is the axis 1, figure out
    U = numpy.concatenate(
            [ortho_weight(options['dim_proj']),
             ortho_weight(options['dim_proj']),
             ortho_weight(options['dim_proj']),
             ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U

    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(theano.config.floatX)

    return params

def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    # State below : steps x samples
    # Recurrence over dim 0
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = T.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = T.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = T.nnet.signoid(_slice(preact, 1, options['dim_proj']))
        o = T.nnet.signoid(_slice(preact, 2, options['dim_proj']))
        c = T.tanh(_slice(preact, 3, options['dim_proj']))
        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (T.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])
    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[T.alloc(numpy_floatX(0.),
                                                      n_samples,
                                                      dim_proj),
                                              T.alloc(numpy_floatX(0.),
                                                      n_samples,
                                                      dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]


# Create the layers (lstm layers)
layers = {'lstm': (param_init_lstm, lstm_layer)}




