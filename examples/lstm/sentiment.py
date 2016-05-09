"""
Build a tweet sentiment analyzer
"""

import numpy
import theano
import theano.tensor as T
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from cutils.numeric import numpy_floatX
from cutils.init import ortho_weight

import imdb


datasets = {'imdb': (imdb.load_data, imdb.prepare_data)}

SEED = 123
numpy.random.seed(SEED)


# TODO: move
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
    # TODO: move
    # TODO: use the noise function that exists somewhere
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
                                            options['ydim']) \
        .astype(theano.config.floatX)
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


def param_init_lstm(options, params, prefix='lstm'):
    """
    Initialize the LSTM params
    """
    rng = numpy.random.RandomState(SEED)
    W = numpy.concatenate([ortho_weight(options['dim_proj'], rng),
                           ortho_weight(options['dim_proj'], rng),
                           ortho_weight(options['dim_proj'], rng),
                           ortho_weight(options['dim_proj'], rng)], axis=1)
    params[_p(prefix, 'W')] = W

    # TODO: why is the axis 1, figure out
    U = numpy.concatenate([ortho_weight(options['dim_proj'], rng),
                           ortho_weight(options['dim_proj'], rng),
                           ortho_weight(options['dim_proj'], rng),
                           ortho_weight(options['dim_proj'], rng)], axis=1)
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


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout
    use_noise = T.matrix('x', dtype='int64')
    x = T.matrix('x', dtype='int64')
    mask = T.matrix('mask', dtype=theano.config.floatX)
    y = T.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = T.nnet.softmax(T.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -T.log(pred[T.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """
    Probabilities for new examples from a trained model
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(theano.config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print("%d/%d samples classified" % (n_done, n_samples))

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Errors for samples for a trained model
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err


def train_lstm(
    dim_proj=128,
    patience=10,
    max_epochs=5000,
    disp_freq=10,
    decay_c=0.,
    lrate=0.0001,
    n_words=10000,
    optimizer='adadelta',
    encoder='lstm',
    save_to='lstm_model.npz',
    valid_freq=370,
    save_freq=1110,
    maxlen=100,
    batch_size=16,
    valid_batch_size=64,
    dataset='imdb',
    noise_std=0.,
    use_dropout=True,
    reload_model=None,
    test_size=-1
):
    pass
