"""
Training for the LSTM-LM
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

import ptb
from lm import LSTM_LM

SEED = 123
numpy.random.seed(SEED)


def train_lstm(
    dim_proj=650,
    patience=10,
    max_epochs=5000,
    disp_freq=10,
    decay_c=0.,
    lrate=0.0001,
    n_words=10000,
    optimizer=adadelta,
    encoder='lstm',
    save_to='lstm_model.npz',
    load_from='lstm_model.96.npz',
    valid_freq=370,
    save_freq=1110,
    maxlen=35,
    batch_size=20,
    valid_batch_size=64,
    dataset='../../data/simple-examples/data',
    noise_std=0.,
    use_dropout=True,
    reload_model=False,
):
    model_options = locals().copy()
    print("model options", model_options)

    print("... Loading data")
    ptb_data = ptb.PTB(dataset, n_words=n_words,
                       emb_dim=model_options['dim_proj'])
    train, valid, test = ptb_data.load_data()
    print("... Done loading data")

    ydim = ptb_data.dictionary.n_words
    model_options['ydim'] = ydim

    print('Building model')
    # Create the initial parameters for the model
    lstm_lm = LSTM_LM(model_options['dim_proj'], ydim,
                      ptb_data.dictionary, SEED)

    if reload_model:
        print('Reloading params from %s' % save_to)
        load_params(load_from, lstm_lm.params)
        # Update the tparams with the new values
        zipp(lstm_lm.params, lstm_lm.tparams)

    # Create the shared variables for the model
    (use_noise, x, mask, cost) = lstm_lm.build_model()

    if decay_c > 0.:
        cost += weight_decay(cost, lstm_lm.tparams['U'], decay_c)

    f_cost = theano.function([x, mask], cost, name='f_cost')
    grads = theano.grad(cost, wrt=list(lstm_lm.tparams.values()))
    f_grad = theano.function([x, mask], grads, name='f_grad')

    lr = T.scalar('lr')
    f_grad_shared, f_update = optimizer(lr, lstm_lm.tparams, grads, cost, x, mask)

    # Keep a few sentences to decode, to see how training is performing
    decode_use_noise, _, _, _ = lstm_lm.build_decode()
    decode_use_noise.set_value(1.)
    decode_sentences = ['with the', 'the cat', 'when the']
    decode_sentences = [ptb_data.dictionary.read_sentence(s) for s in decode_sentences]
    decode_sentences, decode_mask, _ = pad_and_mask(decode_sentences)

    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid), valid_batch_size)
    kf_test = get_minibatches_idx(len(test), valid_batch_size)

    print('%d train examples' % len(train))
    print('%d valid examples' % len(valid))
    print('%d test examples' % len(test))

    history_errs = []
    best_p = None
    bad_count = 0

    if valid_freq == -1:
        valid_freq = len(train) // batch_size
    if save_freq == -1:
        save_freq = len(train) // batch_size

    uidx = 0  # The number of updates done
    estop = False  # Early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0
            # Get shuffled index for the training set
            kf = get_minibatches_idx(len(train), batch_size, shuffle=True)
            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples in this minibatch
                x = [train[t] for t in train_index]

                # Convert to shape (minibatch maxlen, n samples)
                # Truncated backprop
                x, mask, _ = pad_and_mask(x, maxlen=maxlen)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, disp_freq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if save_to and numpy.mod(uidx, save_freq) == 0:
                    print('Saving...')
                    if best_p is not None:
                        lstm_lm.params = best_p
                    else:
                        lstm_lm.params = unzip(lstm_lm.tparams)
                    numpy.savez(save_to, history_errs=history_errs, **lstm_lm.params)
                    pickle.dump(model_options, open('%s.pkl' % save_to, 'wb'),
                                -1)
                    print('Done')

                if numpy.mod(uidx, valid_freq) == 0:
                    use_noise.set_value(0.)
                    valid_cost = lstm_lm.pred_cost(valid, kf_valid)
                    test_cost = lstm_lm.pred_cost(test, kf_test)
                    history_errs.append([valid_cost, test_cost])

                    if (best_p is None or valid_cost <=
                            numpy.array(history_errs)[:, 0].min()):
                        best_p = unzip(lstm_lm.tparams)
                        bad_counter = 0

                    print(('Valid ', valid_cost,
                           'Test ', test_cost))
                    print("Some sentences.. ")
                    print(ptb_data.dictionary.idx_to_words(lstm_lm.f_decode(decode_sentences, decode_mask, model_options['maxlen'])))

                    if (len(history_errs) > patience and valid_cost
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
        zipp(best_p, lstm_lm.tparams)
    else:
        best_p = unzip(lstm_lm.tparams)

    use_noise.set_value(0.)
    # Note that the training dataset is sorted by length.
    # This is for faster decoding, since padding will create smaller batch matrices
    kf_train_sorted = get_minibatches_idx(len(train), batch_size)
    train_cost = lstm_lm.pred_cost(train, kf_train_sorted)
    valid_cost = lstm_lm.pred_cost(valid, kf_valid)
    test_cost = lstm_lm.pred_cost(test, kf_test)

    print('Train ', train_cost, 'Valid ', valid_cost, 'Test ', test_cost)

    if save_to:
        numpy.savez(save_to, train_cost=train_cost,
                    valid_cost=valid_cost, test_cost=test_cost,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f secs/epoch' %
          ((eidx + 1), ((end_time - start_time) / (1. * (eidx + 1)))))
    print('Training took %.1fs' % (end_time - start_time))
    return train_cost, valid_cost, test_cost

if __name__ == '__main__':
    train_lstm(
        max_epochs=100,
        #reload_model=True,
    )
