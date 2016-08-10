import warnings

import numpy
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from collections import OrderedDict

from cutils.params.init import ortho_weight
from cutils.params.utils import init_tparams
from cutils.numeric import numpy_floatX


class LSTM(object):
    def __init__(self, dim_proj, rng, prefix='lstm'):
        """
        Initialize the LSTM params

        dim_proj : The embedding dimension of the input
        """
        self.param_names = []
        params = OrderedDict()
        W = numpy.concatenate([ortho_weight(dim_proj),
                               ortho_weight(dim_proj),
                               ortho_weight(dim_proj),
                               ortho_weight(dim_proj)], axis=1)
        params[_p(prefix, 'W')] = W
        self.param_names.append(_p(prefix, 'W'))

        # axis=1 will concat horizontally.
        # ie., the resulting shape is dim_proj x (dim_proj*4)
        U = numpy.concatenate([ortho_weight(dim_proj),
                               ortho_weight(dim_proj),
                               ortho_weight(dim_proj),
                               ortho_weight(dim_proj)], axis=1)
        params[_p(prefix, 'U')] = U
        self.param_names.append(_p(prefix, 'U'))

        b = numpy.zeros((4 * dim_proj))
        params[_p(prefix, 'b')] = b.astype(theano.config.floatX)
        self.param_names.append(_p(prefix, 'b'))

        self.prefix = prefix
        self.params = params
        self.tparams = init_tparams(params)

    def set_tparams(self, tparams):
        for p in self.param_names:
            self.tparams[p] = tparams[p]

    def lstm_layer(self, state_below, dim_proj, mask=None,
                   n_steps=None, output_to_input_func=None):
        """
        Recurrence with an LSTM hidden unit

        state_below : Is the input. This may be a single sample with
                      multiple timesteps, or a batch
        dim_proj : The dimensionality of the hidden units (projection)
        mask : The mask applied to the input for batching
        n_steps : The number of steps for which this recurrence should be run
                  This is only required with partial input. For any step
                  where no input is available, the output_to_input_func is
                  applied to the previous output and is then used as input
        output_to_input_func : The function to be applied to generate input
                               when partial input is available
        """
        # Make sure that we've initialized the tparams
        assert len(self.tparams) > 0
        # State below : steps x samples
        # Recurrence over dim 0
        if n_steps is None:
            nsteps = state_below.shape[0]
        else:
            nsteps = n_steps

        if n_steps is not None and output_to_input_func is None:
            raise Exception('n_steps was given to the LSTM but no output \
                             to input function was specified')

        # Hack to make sure that the theano ifelse compiles
        if output_to_input_func is None:
            output_to_input_func = dummy_func

        # Check if the input is a batch or a single sample
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1

        if mask is None:
            warnings.warn("You seem to be supplying single samples for \
                           recurrence. You may see speedup gains with using \
                           batches instead. If your samples are equal length, \
                           you may safely ignore this warning")

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        #def _step(m_, x_, h_, c_):
        def _step(t_, h_, c_, mask, state_below):
            """
            m_ is the mask for this timestep (N x 1)
            x_ is the input for this time step (pre-multiplied with the
              weight matrices). ie.
              x_ = (X.W + b)[t]
            h_ is the previous hidden state
            c_ is the previous LSTM context
            """
            preact = T.dot(h_, self.tparams[_p(self.prefix, 'U')])
            x_ = ifelse(T.lt(t_, state_below.shape[0]),
                             state_below[t_],
                             T.dot(output_to_input_func(h_), self.tparams[_p(self.prefix, 'W')])
                                   + self.tparams[_p(self.prefix, 'b')]
                            )
            preact += x_

            # The input to the sigmoid is preact[:, :, 0:d]
            # Similar slices are used for the rest of the gates
            i = T.nnet.sigmoid(_slice(preact, 0, dim_proj))
            f = T.nnet.sigmoid(_slice(preact, 1, dim_proj))
            o = T.nnet.sigmoid(_slice(preact, 2, dim_proj))
            c = T.tanh(_slice(preact, 3, dim_proj))
            c = f * c_ + i * c
            h = o * T.tanh(c)
            # None adds a dimension to the mask (N,) -> (N, 1)
            # Where the mask value is 1, use the value of the current
            # context, otherwise use the one from the previous
            # context when the mask value is 0
            # This will ensure that values generated for absent
            # elements marked with <PAD> will not be used
            # Similarrly, Where the mask value is 1, use the value of the current
            # hidden state, otherwise use the one from the previous
            # state when the mask value is 0
            c = ifelse(T.lt(t_, state_below.shape[0]),
                       mask[t_][:, None] * c + (1. - mask[t_])[:, None] * c_,
                       c)
            h = ifelse(T.lt(t_, state_below.shape[0]),
                       mask[t_][:, None] * h + (1. - mask[t_])[:, None] * h_,
                       c)
            #c = m_[:, None] * c + (1. - m_)[:, None] * c_
            #h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c

        state_below = (T.dot(state_below, self.tparams[_p(self.prefix, 'W')]) +
                       self.tparams[_p(self.prefix, 'b')])
        rval, updates = theano.scan(_step,
                                    sequences=[T.arange(nsteps)],
                                    outputs_info=[T.alloc(numpy_floatX(0.),
                                                          n_samples,
                                                          dim_proj),
                                                  T.alloc(numpy_floatX(0.),
                                                          n_samples,
                                                          dim_proj)],
                                    non_sequences=[mask, state_below],
                                    name=_p(self.prefix, '_layers'),
                                    n_steps=nsteps)
        # Returns a list of the hidden states (t elements of N x dim_proj)
        return rval[0]


def _p(pp, name):
    return '%s_%s' % (pp, name)


def dummy_func(_input):
    return _input
