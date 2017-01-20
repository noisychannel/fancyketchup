"""
Training for the LSTM-LM
"""
#TODO: Model options are not cleanly handled. Mix of direct args and
#      invocation of model options

from __future__ import print_function

import os
import sys
import time
import pickle
import argparse
import numpy
import theano
import theano.tensor as T

from cutils.training.utils import get_minibatches_idx, weight_decay
from cutils.params.utils import zipp, unzip, load_params
from cutils.data_interface.utils import pad_and_mask
from cutils.training.trainer import adadelta, sgd

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
    use_dropout=True,
    reload_model=False,
    decay_lr_after_ep=None,
    decay_lr_factor=1.
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
        print('Reloading params from %s' % load_from)
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
    decode_sentences = ['<BOS> with the', '<BOS> the cat', '<BOS> the meaning']
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
            kf = get_minibatches_idx(len(train), batch_size,
                                     shuffle=True, use_remaining=False)
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

                    # After patience expires, we will not tolerate #patience worse costs and then quit.
                    if (len(history_errs) > patience and valid_cost
                            >= numpy.array(history_errs)[:-patience, 0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)
            # Decay learning rate
            if (eidx + 1) >= decay_lr_after_ep:
                lrate = lrate / decay_lr_factor
                print('Learning rate is now : ', lrate)

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
    parser = argparse.ArgumentParser(description='LSTM Language Model \n \
        Eg. python train.py --dataset ../../data/simple_examples/data --save-to lstm_model.npz')

    parser.add_argument('--dim-proj', type=int, help='The size of the hidden states', default=650)
    parser.add_argument('--patience', type=int, help='The patience value for early stopping. \
        How many worse batch values are we willing to tolerate before we stop', default=200000)
    parser.add_argument('--max-epochs', type=int, help='Maximum number of epochs', default=100)
    parser.add_argument('--disp-freq', type=int, help='How often should we display the current cost \
        (batch)', default=10)
    parser.add_argument('--decay-c', type=float, help='Parameter for weight decay', default=0.)
    parser.add_argument('--lrate', type=float, help='The initial learning rate. Ignored for \
        adadelta', default=0.0001)
    parser.add_argument('--n-words', type=int, help='The number of top words to retain in the vocab. \
        Everything else is replaced by UNK', default=10000)
    parser.add_argument('--optimizer', type=str, help='The optimizer to use for learning.', \
        default=adadelta)
    parser.add_argument('--encoder', type=str, help='The encoder to use', default='lstm')
    parser.add_argument('--save-to', type=str, help='The location of the serialized learnt \
        params.', required=True)
    parser.add_argument('--load-from', type=str, help='To resume training, load params from \
        this location. Only used when reload_model is True', default='')
    parser.add_argument('--reload-model', type=bool, help='Resume training', default=False)
    parser.add_argument('--valid-freq', type=int, help='How often should we validate against the \
        valid set? Used for early stopping', default=370)
    parser.add_argument('--save-freq', type=int, help='How ofen should we save our best params?', \
        default=1110)
    parser.add_argument('--maxlen', type=int, help='What is the maximum length of a train sequence \
        we should accomodate? Equivalent to truncated backprop', default=35)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=20)
    parser.add_argument('--valid-batch-size', type=int, help='Valid and test batch sizes', default=64)
    parser.add_argument('--dataset', type=str, help='Location of the dataset', required=True)
    parser.add_argument('--use-dropout', type=bool, help='Should we use dropout?', default=True)
    parser.add_argument('--decay-lr-after-ep', type=int, help='After how many epochs should we \
        start decaying the learning rate? Useful for SGD only', default=10000)
    parser.add_argument('--decay-lr-factor', type=float, help='How much should we decay the learning \
        rate by? Useful for SGD only.', default=1.2)

    args = parser.parse_args()

    if type(args.optimizer) == str:
        if args.optimizer == 'sgd':
            args.optimizer = sgd
        elif args.optimizer == 'adadelta':
            args.optimzer = adadelta
        else:
            raise NotImplementedError

    train_lstm(
        dim_proj=args.dim_proj,
        patience=args.patience,
        max_epochs=args.max_epochs,
        disp_freq=args.disp_freq,
        decay_c=args.decay_c,
        lrate=args.lrate,
        n_words=args.n_words,
        optimizer=args.optimizer,
        encoder=args.encoder,
        save_to=args.save_to,
        load_from=args.load_from,
        valid_freq=args.valid_freq,
        save_freq=args.save_freq,
        maxlen=args.maxlen,
        batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size,
        dataset=args.dataset,
        use_dropout=args.use_dropout,
        reload_model=args.reload_model,
        decay_lr_after_ep=args.decay_lr_after_ep,
        decay_lr_factor=args.decay_lr_factor
    )
