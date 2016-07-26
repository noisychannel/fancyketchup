"""
Build a tweet sentiment analyzer
"""

from __future__ import print_function

import os
import sys
import time
import pickle
import numpy
import theano
import theano.tensor as T

from cutils.training.utils import get_minibatches_idx, weight_decay
from cutils.params.utils import zipp, unzip, load_params
from cutils.data_interface.utils import pad_and_mask
from cutils.training.trainer import adadelta

# Include current path in the pythonpath
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_path)

import imdb
from classifier import LSTM_CF

SEED = 123
numpy.random.seed(SEED)


def train_lstm(
    dim_proj=128,
    patience=10,
    max_epochs=5000,
    disp_freq=10,
    decay_c=0.,
    lrate=0.0001,
    n_words=10000,
    optimizer=adadelta,
    encoder='lstm',
    save_to='lstm_model.npz',
    valid_freq=370,
    save_freq=1110,
    maxlen=100,
    batch_size=16,
    valid_batch_size=64,
    dataset='../../data/aclImdb',
    noise_std=0.,
    use_dropout=True,
    reload_model=None,
    test_size=-1
):
    model_options = locals().copy()
    print("model options", model_options)

    imdb_data = imdb.IMDB(dataset, n_words=n_words,
                          emb_dim=model_options['dim_proj'])
    train, valid, test = imdb_data.load_data(valid_portion=0.05, maxlen=maxlen)

    if test_size > 0:
        # Random shuffle of the test set
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    ydim = numpy.max(train[1]) + 1
    model_options['ydim'] = ydim

    print('Building model')
    # Create the initial parameters for the model
    lstm_cf = LSTM_CF(model_options['dim_proj'], ydim,
                      imdb_data.dictionary, SEED)

    if reload_model:
        load_params('lstm_model.npz', lstm_cf.params)
        # Update the tparams with the new values
        zipp(lstm_cf.params, lstm_cf.tparams)

    # Create the shared variables for the model
    (use_noise, x, mask, y, cost) = lstm_cf.build_model()

    if decay_c > 0.:
        cost += weight_decay(cost, lstm_cf.tparams['U'], decay_c)

    f_cost = theano.function([x, mask, y], cost, name='f_cost')
    grads = theano.grad(cost, wrt=list(lstm_cf.tparams.values()))
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = T.scalar('lr')
    f_grad_shared, f_update = optimizer(lr, lstm_cf.tparams, grads, cost, x, mask, y)

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
                x, mask, y = pad_and_mask(x, y)
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
                        lstm_cf.params = best_p
                    else:
                        lstm_cf.params = unzip(lstm_cf.tparams)
                    numpy.savez(save_to, history_errs=history_errs, **lstm_cf.params)
                    pickle.dump(model_options, open('%s.pkl' % save_to, 'wb'),
                                -1)
                    print('Done')

                if numpy.mod(uidx, valid_freq) == 0:
                    use_noise.set_value(0.)
                    train_err = lstm_cf.pred_error(train, kf)
                    valid_err = lstm_cf.pred_error(valid, kf_valid)
                    test_err = lstm_cf.pred_error(test, kf_test)
                    history_errs.append([valid_err, test_err])

                    if (best_p is None or valid_err <=
                            numpy.array(history_errs)[:, 0].min()):
                        best_p = unzip(lstm_cf.tparams)
                        bad_counter = 0

                    print(('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err))

                    if (len(history_errs) > patience and valid_err
                            >= numpy.array(history_errs)[:-patience, 0].min()):
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
        zipp(best_p, lstm_cf.tparams)
    else:
        best_p = unzip(lstm_cf.tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = lstm_cf.pred_error(train, kf_train_sorted)
    valid_err = lstm_cf.pred_error(valid, kf_valid)
    test_err = lstm_cf.pred_error(test, kf_test)

    print('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err)

    if save_to:
        numpy.savez(save_to, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f epochs/sec' %
          ((eidx + 1), ((end_time - start_time) / (1. * (eidx + 1)))))
    print('Training took %.1fs' % (end_time - start_time))
    return train_err, valid_err, test_err

if __name__ == '__main__':
    train_lstm(
        max_epochs=100,
        test_size=500
    )
