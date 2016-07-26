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


class LSTM_CF(object):
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
        mask = T.matrix('mask', dtype=theano.config.floatX)
        y = T.vector('y', dtype='int64')

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        emb = self.tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                         n_samples,
                                                         self.dim_proj])
        proj = self.layers['lstm'].lstm_layer(emb, self.dim_proj, mask=mask)
        # TODO: What happens when the encoder is not an LSTM
        # This should cleanly fall back to a normal hidden unit
        if encoder == 'lstm':
            #TODO: What the shit is happening here?
            proj = (proj * mask[:, :, None]).sum(axis=0)
            proj = proj / mask.sum(axis=0)[:, None]
        if use_dropout:
            trng = RandomStreams(self.random_seed)
            proj = dropout_layer(proj, use_noise, trng)

        pred = T.nnet.softmax(T.dot(proj, self.tparams['U'])
                              + self.tparams['b'])

        self.f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
        self.f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

        off = 1e-8
        if pred.dtype == 'float16':
            off = 1e-6

        cost = -T.log(pred[T.arange(n_samples), y] + off).mean()

        return use_noise, x, mask, y, cost


    def pred_probs(self, data, iterator, verbose=False):
        """
        Probabilities for new examples from a trained model
        """
        n_samples = len(data[0])
        probs = numpy.zeros((n_samples, 2)).astype(theano.config.floatX)

        n_done = 0

        for _, valid_index in iterator:
            x, mask, y = pad_and_mask([data[0][t] for t in valid_index],
                                      numpy.array(data[1])[valid_index],
                                      maxlen=None)
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
            x, mask, y = pad_and_mask([data[0][t] for t in valid_index],
                                      numpy.array(data[1])[valid_index],
                                      maxlen=None)
            preds = self.f_pred(x, mask)
            targets = numpy.array(data[1])[valid_index]
            valid_err += (preds == targets).sum()
        valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

        return valid_err
