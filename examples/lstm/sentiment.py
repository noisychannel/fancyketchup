"""
Build a tweet sentiment analyzer
"""

import time
import pickle
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
    model_options = locals().copy()
    print("model options", model_options)

    load_data, prepare_data = get_dataset(dataset)
    train, valid, test = load_data(n_words=n_words, valid_portion=0.5,
                                   maxlen=maxlen)
    if test_size > 0:
        # Random shuffle of the test set
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = (test[0][n] for n in idx), (test[1][n] for n in idx)

    ydim = numpy.max(train[1]) + 1
    model_options['ydim'] = ydim

    print('Building model')
    # Create the initial parameters for the model
    params = init_params(model_options)

    if reload_model:
        load_params('lstm_model.npz', params)

    # Create the shared variables for the model
    tparams = init_tparams(params)
    (use_noise, x, mask,
     y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    # TODO: move this
    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y], cost, name='f_cost')
    grads = theano.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = theano.scalar('lr')
    f_grad_shared, f_update = optimizer([lr, tparams, grads, x, mask, y, cost])

    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print('%d train examples' % len(train[0]))
    print('%d valid examples' % len(valid[0]))
    print('%d test examples' % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    if valid_freq == -1:
        valid_freq = len(train[0]) // batch_size
    if save_freq == -1:
        save_freq = len(train[0]) // batch_size

    uidx = 0  # The number of updates done
    estop = False  # Early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0
            # Get shuffled index for the training set
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples in this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]

                # Convert to shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, y)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, disp_freq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if save_to and numpy.mod(uidx, save_freq) == 0:
                    print('Saving...')
                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(save_to, history_errs=history_errs, **params)
                    pickle.dump(model_options, open('%s.pkl' % save_to, 'wb'), -1)
                    print('Done')

                if numpy.mod(uidx, valid_freq) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
                    test_err = pred_error(f_pred, pred_error, test, kf_test)
                    history_errs.append([valid_err, test_err])

                    if (best_p is None or valid_err <= numpy.array([history_errs])[:0].min()):
                        best_p = unzip(tparams)
                        bad_counter = 0

                    if (len(history_errs) > patience and valid_err >= numpy.array(history_errs)[:-patience,0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print('Training Interrupted')

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, params)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, pred_error, test, kf_test)

    print('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err)

    if save_to:
        numpy.savez(save_to, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f epochs/sec' % ((eidx + 1), ((end_time - start_time) / (1. * (eidx + 1)))))
    print('Training took %.1fs' % (end_time - start_time))
    return train_err, valid_err, test_err

if __name__ == '__main__':
    train_lstm(
        max_epochs=100,
        test_size=500
    )
