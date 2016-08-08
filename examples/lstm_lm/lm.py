"""
Build a tweet sentiment analyzer
"""

from __future__ import print_function

import numpy
import theano
import theano.tensor as T
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from cutils.numeric import numpy_floatX
from cutils.layers.utils import dropout_layer
from cutils.layers.lstm import LSTM
from cutils.data_interface.utils import pad_and_mask
from cutils.params.utils import init_tparams


class LSTM_LM(object):
    def _p(self, pp, name):
        return '%s_%s' % (pp, name)


    def __init__(self, dim_proj, ydim, word_dict, random_seed):
        """
        Embedding and classifier params
        """
        self.layers = {}
        self.random_seed = random_seed
        self.dim_proj = dim_proj
        self.ydim = ydim
        self.rng = numpy.random.RandomState(self.random_seed)
        self.params = OrderedDict()
        self.tparams = OrderedDict()
        self.f_pred_prob = None
        self.f_pred = None

        def unpack(source, target):
            for kk, vv in source.items():
                target[kk] = vv

        # Add parameters from dictionary
        unpack(word_dict.params, self.params)
        unpack(word_dict.tparams, self.tparams)
        # Initialize LSTM and add its params
        self.layers['lstm'] = LSTM(dim_proj, self.rng)
        unpack(self.layers['lstm'].params, self.params)
        unpack(self.layers['lstm'].tparams, self.tparams)
        # Initialize other params
        other_params = OrderedDict()
        other_params['U'] = 0.01 * numpy.random.randn(dim_proj, ydim) \
            .astype(theano.config.floatX)
        other_params['b'] = numpy.zeros((ydim,)).astype(theano.config.floatX)
        other_tparams = init_tparams(other_params)
        unpack(other_params, self.params)
        unpack(other_tparams, self.tparams)


    def build_model(self, encoder='lstm', use_dropout=True):
        use_noise = theano.shared(numpy_floatX(0.))
        x = T.matrix('x', dtype='int64')
        # Since we are simply predicting the next word, the
        # following statement shifts the content of the x by 1
        # in the time dimension for prediction (axis 0, assuming TxNxV)
        y = T.roll(x, -1, 0)
        mask = T.matrix('mask', dtype=theano.config.floatX)

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        emb = self.tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                         n_samples,
                                                         self.dim_proj])
        proj = self.layers['lstm'].lstm_layer(emb, self.dim_proj, mask=mask)
        # Apply the mask to the final output to 0 out the time steps that are invalid
        proj = proj * mask[:, :, None]
        if use_dropout:
            trng = RandomStreams(self.random_seed)
            proj = dropout_layer(proj, use_noise, trng)

        pre_s = T.dot(proj, self.tparams['U']) + self.tparams['b']
        # Softmax works for 2-tensors (matrices) only. We have a 3-tensor
        # TxNxV. So we reshape it to (T*N)xV, apply softmax and reshape again
        # -1 is a proxy for infer dim based on input (numpy style)
        pre_s_r = T.reshape(pre_s, (pre_s.shape[0] * pre_s.shape[1], -1))
        # Softmax will receive all-0s for previously padded entries
        pred_r = T.nnet.softmax(pre_s_r)
        pred = T.reshape(pred_r, pre_s.shape)

        self.f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
        self.f_pred = theano.function([x, mask], pred.argmax(axis=2), name='f_pred')

        off = 1e-8
        if pred.dtype == 'float16':
            off = 1e-6

        # Note the use of flatten here. We can't directly index a 3-tensor
        # and hence we use the (T*N)xV view which is indexed by the flattened
        # label matrix, dim = (T*N)x1
        cost = -T.log(pred_r[T.arange(pred_r.shape[0]), y.flatten()] + off).mean()

        return use_noise, x, mask, cost


    def pred_probs(self, data, iterator, verbose=False):
        """
        Probabilities for new examples from a trained model
        """
        n_steps = len(data[0])
        n_samples = len(data[1])
        probs = numpy.zeros((n_steps, n_samples, self.ydim)).astype(theano.config.floatX)

        n_done = 0

        for _, valid_index in iterator:
            x, mask, _ = pad_and_mask([data[t] for t in valid_index])
            pred_probs = self.f_pred_prob(x, mask)
            probs[valid_index, :] = pred_probs

            n_done += len(valid_index)
            if verbose:
                print("%d/%d samples classified" % (n_done, n_samples))

        return probs


    def pred_error(self, data, iterator, verbose=False):
        """
        Errors for samples for a trained model
        """
        valid_err = 0
        for _, valid_index in iterator:
            x, mask, _ = pad_and_mask([data[t] for t in valid_index])
            # Preds is TxN
            preds = self.f_pred(x, mask)
            # Targets is TxN
            targets = np.roll(x, -1, 0)
            valid_err += (preds == targets).sum()
        valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

        return valid_err
