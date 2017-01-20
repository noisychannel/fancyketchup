"""
An LSTM language model
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
from cutils.layers.logistic_regression import LogisticRegression
from cutils.data_interface.utils import pad_and_mask
from cutils.params.utils import init_tparams


class LSTM_LM(object):
    def _p(self, pp, name):
        return '%s_%s' % (pp, name)


    def __init__(self, dim_proj, ydim, word_dict, random_seed, use_dropout=True):
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
        self.f_cost = None
        self.f_decode = None
        self.use_dropout = use_dropout

        def unpack(source, target):
            for kk, vv in source.items():
                target[kk] = vv

        # Add parameters from dictionary
        unpack(word_dict.params, self.params)
        unpack(word_dict.tparams, self.tparams)
        # Initialize LSTM and add its params
        # Layer 1
        self.layers['lstm_1'] = LSTM(dim_proj, prefix='lstm_1')
        unpack(self.layers['lstm_1'].params, self.params)
        unpack(self.layers['lstm_1'].tparams, self.tparams)
        # Layer 2
        self.layers['lstm_2'] = LSTM(dim_proj, prefix='lstm_2')
        unpack(self.layers['lstm_2'].params, self.params)
        unpack(self.layers['lstm_2'].tparams, self.tparams)
        # Logit : hidden state to output
        self.layers['logit_lstm'] = LogisticRegression(dim_proj, dim_proj, prefix='logit_lstm', ortho=False)
        unpack(self.layers['logit_lstm'].params, self.params)
        unpack(self.layers['logit_lstm'].tparams, self.tparams)
        # Logit : raw input to output
        self.layers['logit_prev_word'] = LogisticRegression(dim_proj, dim_proj, prefix='logit_prev_word', ortho=False)
        unpack(self.layers['logit_prev_word'].params, self.params)
        unpack(self.layers['logit_prev_word'].tparams, self.tparams)
        # Logit : Softmax
        self.layers['logit'] = LogisticRegression(ydim, dim_proj, prefix='logit', ortho=False)
        unpack(self.layers['logit'].params, self.params)
        unpack(self.layers['logit'].tparams, self.tparams)
        ## Initialize other params
        #other_params = OrderedDict()
        #other_params['U'] = 0.01 * numpy.random.randn(dim_proj, ydim) \
            #.astype(theano.config.floatX)
        #other_params['b'] = numpy.zeros((ydim,)).astype(theano.config.floatX)
        #other_tparams = init_tparams(other_params)
        #unpack(other_params, self.params)
        #unpack(other_tparams, self.tparams)


    def build_model(self):
        trng = RandomStreams(self.random_seed)
        use_noise = theano.shared(numpy_floatX(0.))
        x = T.matrix('x', dtype='int64')
        # Since we are simply predicting the next word, the
        # following statement shifts the content of the x by 1
        # in the time dimension for prediction (axis 0, assuming TxN)
        y = T.roll(x, -1, 0)
        mask = T.matrix('mask', dtype=theano.config.floatX)

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        # Convert word indices to their embeddings
        # Resulting dims are (T x N x dim_proj)
        emb = self.tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                         n_samples,
                                                         self.dim_proj])
        # Dropout input if necessary
        if self.use_dropout:
            emb = dropout_layer(emb, use_noise, trng)

        # Compute the hidden states
        # Note that these contain hidden states for elements which were
        # padded in input. The cost for these time steps are removed
        # before the calculation of the cost.
        proj_1 = self.layers['lstm_1'].lstm_layer(emb, mask=mask, restore_final_to_initial_hidden=True)
        # Use dropout on non-recurrent connections (Zaremba et al.)
        if self.use_dropout:
            proj_1 = dropout_layer(proj_1, use_noise, trng)
        proj = self.layers['lstm_2'].lstm_layer(proj_1, mask=mask, restore_final_to_initial_hidden=True)
        if self.use_dropout:
            proj = dropout_layer(proj, use_noise, trng)

        pre_s_lstm = self.layers['logit_lstm'].logit_layer(proj)
        pre_s_input = self.layers['logit_prev_word'].logit_layer(emb)
        pre_s = self.layers['logit'].logit_layer(T.tanh(pre_s_lstm + pre_s_input))
        # Softmax works for 2-tensors (matrices) only. We have a 3-tensor
        # TxNxV. So we reshape it to (T*N)xV, apply softmax and reshape again
        # -1 is a proxy for infer dim based on input (numpy style)
        pre_s_r = T.reshape(pre_s, (pre_s.shape[0] * pre_s.shape[1], -1))
        pred_r = T.nnet.softmax(pre_s_r)

        off = 1e-8
        if pred_r.dtype == 'float16':
            off = 1e-6

        # Note the use of flatten here. We can't directly index a 3-tensor
        # and hence we use the (T*N)xV view which is indexed by the flattened
        # label matrix, dim = (T*N)x1
        # Also, the cost (before calculating the mean) is multiplied (element-wise)
        # with the mask to eliminate the cost of elements that do not really exist.
        # i.e. Do not include the cost for elements which are padded
        cost = -T.sum(T.log(pred_r[T.arange(pred_r.shape[0]), y.flatten()] + off) * mask.flatten()) / T.sum(mask)

        self.f_cost = theano.function([x, mask], cost, name='f_cost')

        return use_noise, x, mask, cost


    def build_decode(self):
        # Input to start the recurrence with
        trng = RandomStreams(self.random_seed)
        use_noise = theano.shared(numpy_floatX(0.))
        x = T.matrix('x', dtype='int64')
        # Number of steps we want the recurrence to run for
        n_timesteps = T.iscalar('n_timesteps')
        n_samples = x.shape[1]

        # The mask for the first layer has to be all 1s.
        # It does not make sense to complete a sentence for which
        # The mask is 1 1 0 (because it's already complete).
        mask = T.matrix('mask', dtype=theano.config.floatX)
        # This is a dummy mask, we want to consider all hidden states for
        # the second layer when decoding
        mask_2 = T.alloc(numpy_floatX(1.),
                         n_timesteps,
                         n_samples)
        emb = self.tparams['Wemb'][x.flatten()].reshape([x.shape[0],
                                                         x.shape[1],
                                                         self.dim_proj])

        def output_to_input_transform(output, emb):
            """
            output : The previous hidden state (Nxd)
            """
            # N X V
            pre_soft_lstm = self.layers['logit_lstm'].logit_layer(output)
            pre_soft_input = self.layers['logit_prev_word'].logit_layer(emb)
            pre_soft = self.layers['logit'].logit_layer(T.tanh(pre_s_lstm + pre_s_input))
            pred = T.nnet.softmax(pre_soft)
            # N x 1
            pred_argmax = pred.argmax(axis=1)
            # N x d (flatten is probably redundant)
            new_input = self.tparams['Wemb'][pred_argmax.flatten()].reshape([n_samples,
                                                                       self.dim_proj])
            return new_input

        proj_1 = self.layers['lstm_1'].lstm_layer(emb, self.dim_proj, mask=mask, n_steps=n_timesteps,
                                               output_to_input_func=output_to_input_transform)
        if self.use_dropout:
            proj_1 = dropout_layer(proj_1, use_noise, trng)
        proj = self.layers['lstm_2'].lstm_layer(proj_1, self.dim_proj, mask=mask_2)
        if self.use_dropout:
            proj = dropout_layer(proj, use_noise, trng)

        pre_s_lstm = self.layers['logit_lstm'].logit_layer(proj)
        pre_s_input = self.layers['logit_prev_word'].logit_layer(emb)
        pre_s = self.layers['logit'].logit_layer(T.tanh(pre_s_lstm + pre_s_input))
        # Softmax works for 2-tensors (matrices) only. We have a 3-tensor
        # TxNxV. So we reshape it to (T*N)xV, apply softmax and reshape again
        # -1 is a proxy for infer dim based on input (numpy style)
        pre_s_r = T.reshape(pre_s, (pre_s.shape[0] * pre_s.shape[1], -1))
        # Softmax will receive all-0s for previously padded entries
        # (T*N) x V
        pred_r = T.nnet.softmax(pre_s_r)
        # T x N
        pred = (T.reshape(pred_r, pre_s.shape)[:,:,2:]).argmax(axis=2) + 2
        self.f_decode = theano.function([x, mask, n_timesteps], pred, name='f_decode')

        return use_noise, x, mask, n_timesteps


    def pred_cost(self, data, iterator, verbose=False):
        """
        Probabilities for new examples from a trained model

        data : The complete dataset. A list of lists. Each nested list is a sample
        iterator : A list of lists. Each nested list is a batch with idxs to the sample in data
        """
        # Total samples
        n_samples = len(data)
        running_cost = []
        samples_seen = []

        n_done = 0

        # valid_index is a list containing the IDXs of samples for a batch
        for _, valid_index in iterator:
            x, mask, _ = pad_and_mask([data[t] for t in valid_index])
            # Accumulate running cost
            samples_seen.append(len(valid_index))
            running_cost.append(self.f_cost(x, mask))
            n_done += len(valid_index)
            if verbose:
                print("%d/%d samples classified" % (n_done, n_samples))

        return sum([samples_seen[i] * running_cost[i] for i in range(len(samples_seen))]) / sum(samples_seen)
