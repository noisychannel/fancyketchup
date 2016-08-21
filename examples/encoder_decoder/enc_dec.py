"""
The Encoder-decoder class
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


class ENC_DEC(object):
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
        # Encoder - Layer 1
        self.layers['enc_lstm_1'] = LSTM(dim_proj, self.rng, prefix='enc_lstm_1')
        unpack(self.layers['enc_lstm_1'].params, self.params)
        unpack(self.layers['enc_lstm_1'].tparams, self.tparams)
        # Encoder - Layer 2
        self.layers['enc_lstm_2'] = LSTM(dim_proj, self.rng, prefix='enc_lstm_2')
        unpack(self.layers['enc_lstm_2'].params, self.params)
        unpack(self.layers['enc_lstm_2'].tparams, self.tparams)
        # Decoder - Layer 1
        self.layers['dec_lstm_1'] = LSTM(dim_proj, self.rng, prefix='dec_lstm_1')
        unpack(self.layers['dec_lstm_1'].params, self.params)
        unpack(self.layers['dec_lstm_1'].tparams, self.tparams)
        # Decoder - Layer2
        self.layers['dec_lstm_2'] = LSTM(dim_proj, self.rng, prefix='dec_lstm_2')
        unpack(self.layers['dec_lstm_2'].params, self.params)
        unpack(self.layers['dec_lstm_2'].tparams, self.tparams)
        # Initialize other params
        other_params = OrderedDict()
        other_params['U'] = 0.01 * numpy.random.randn(dim_proj, ydim) \
            .astype(theano.config.floatX)
        other_params['b'] = numpy.zeros((ydim,)).astype(theano.config.floatX)
        other_tparams = init_tparams(other_params)
        unpack(other_params, self.params)
        unpack(other_tparams, self.tparams)


    def build_model(self):
        trng = RandomStreams(self.random_seed)
        use_noise = theano.shared(numpy_floatX(0.))
        # Simply encode this
        x = T.matrix('x', dtype='int64')
        y = T.matrix('y', dtype='int64')
        y_prime = T.roll(y, -1, 0)
        # Since we are simply predicting the next word, the
        # following statement shifts the content of the x by 1
        # in the time dimension for prediction (axis 0, assuming TxN)
        mask_x = T.matrix('mask_x', dtype=theano.config.floatX)
        mask_y = T.matrix('mask_y', dtype=theano.config.floatX)

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        # Convert word indices to their embeddings
        # Resulting dims are (T x N x dim_proj)
        emb = self.tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                         n_samples,
                                                         self.dim_proj])
        # Compute the hidden states
        # Note that these contain hidden states for elements which were
        # padded in input. The cost for these time steps are removed
        # before the calculation of the cost.
        enc_proj_1 = self.layers['enc_lstm_1'].lstm_layer(emb, self.dim_proj, mask=mask)
        # Use dropout on non-recurrent connections (Zaremba et al.)
        if self.use_dropout:
            proj_1 = dropout_layer(enc_proj_1, use_noise, trng)
        enc_proj_2 = self.layers['enc_lstm_2'].lstm_layer(enc_proj_1, self.dim_proj, mask=mask)
        if self.use_dropout:
            enc_proj_2 = dropout_layer(enc_proj_2, use_noise, trng)

        # Use the final state of the encoder as the initial hidden state of the decoder
        src_embedding = enc_proj_2[-1]
        # Run decoder LSTM
        dec_proj_1 = self.layers['enc_lstm_1'].lstm_layer(emb, self.dim_proj, mask=mask)
        # Use dropout on non-recurrent connections (Zaremba et al.)
        if self.use_dropout:
            proj_1 = dropout_layer(enc_proj_1, use_noise, trng)
        enc_proj_2 = self.layers['enc_lstm_2'].lstm_layer(enc_proj_1, self.dim_proj, mask=mask)
        if self.use_dropout:
            enc_proj_2 = dropout_layer(enc_proj_2, use_noise, trng)

        pre_s = T.dot(proj, self.tparams['U']) + self.tparams['b']
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

        def output_to_input_transform(output):
            """
            output : The previous hidden state (Nxd)
            """
            # N X V
            pre_soft = T.dot(output, self.tparams['U']) + self.tparams['b']
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

        pre_s = T.dot(proj, self.tparams['U']) + self.tparams['b']
        # Softmax works for 2-tensors (matrices) only. We have a 3-tensor
        # TxNxV. So we reshape it to (T*N)xV, apply softmax and reshape again
        # -1 is a proxy for infer dim based on input (numpy style)
        pre_s_r = T.reshape(pre_s, (pre_s.shape[0] * pre_s.shape[1], -1))
        # Softmax will receive all-0s for previously padded entries
        # (T*N) x V
        pred_r = T.nnet.softmax(pre_s_r)
        # T x N
        pred = T.reshape(pred_r, pre_s.shape).argmax(axis=2)
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
