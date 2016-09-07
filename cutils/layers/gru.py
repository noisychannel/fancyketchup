import warnings

import numpy
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from collections import OrderedDict

from cutils.params.init import norm_init, ortho_weight
from cutils.params.utils import init_tparams
from cutils.numeric import numpy_floatX


class GRU(object):
    def __init__(self, dim_proj, dim_input=None, prefix='lstm'):
        """
        Initialize the GRU params

        dim_proj : The embedding dimension of the hidden layer
        dim_input : The embedding dimension of the input
        """
        self.param_names = []
        params = OrderedDict()

        if dim_input is None:
            dim_input = dim_proj

        # Initialize weights using a scaled standard normal distribution
        # which will fall back to orthogonal weights if dim_proj = dim_input
        # These weights transform the input to the dimensionality of the hidden states
        W = numpy.concatenate([norm_init(dim_input, dim_proj),
                               norm_init(dim_input, dim_proj)], axis=1)
        params[_p(prefix, 'W')] = W
        self.param_names.append(_p(prefix, 'W'))

        # axis=1 will concat horizontally.
        # ie., the resulting shape is dim_proj x (dim_proj*4)
        # Recurrence weights
        U = numpy.concatenate([ortho_weight(dim_proj),
                               ortho_weight(dim_proj)], axis=1)
        params[_p(prefix, 'U')] = U
        self.param_names.append(_p(prefix, 'U'))

        b = numpy.zeros((2 * dim_proj))
        params[_p(prefix, 'b')] = b.astype(theano.config.floatX)
        self.param_names.append(_p(prefix, 'b'))

        # Parameters for the calculating the candidate hidden state
        W_h = norm_init(dim_input, dim_proj)
        params[_p(prefix, 'W_h')] = W_h
        self.param_names.append(_p(prefix, 'W_h'))

        U_h = ortho_weight(dim_proj)
        params[_p(prefix, 'U_h')] = U_h
        self.param_names.append(_p(prefix, 'U_h'))

        b_h = numpy.zeros(dim_proj)
        params[_p(prefix, 'b_h')] = b_h.astype(theano.config.floatX)
        self.param_names.append(_p(prefix, 'b_h'))

        # Memory of the last final hidden states
        # Not archived
        self.h_final = None

        self.prefix = prefix
        self.params = params
        self.tparams = init_tparams(params)
        # TODO: Make this change to the LSTM module. dim_proj can be stored
        self.dim_proj = dim_proj
        self.dim_input = dim_input

    def set_tparams(self, tparams):
        for p in self.param_names:
            self.tparams[p] = tparams[p]

    def gru_layer(self, state_below, mask=None,
                   n_steps=None, output_to_input_func=None,
                   restore_final_to_initial_hidden=False):
        """
        Recurrence with an LSTM hidden unit

        state_below : Is the input. This may be a single sample with
                      multiple timesteps, or a batch
        mask : The mask applied to the input for batching
        n_steps : The number of steps for which this recurrence should be run
                  This is only required with partial input. For any step
                  where no input is available, the output_to_input_func is
                  applied to the previous output and is then used as input
        output_to_input_func : The function to be applied to generate input
                               when partial input is available
        restore_final_to_initial_hidden : Use the final hidden state as the initial
                                  hidden state for the next batch
                                  WARNING : Assumes that batches are of the
                                  same size since the size of the initial
                                  state is fixed to Nxd
                                  TODO: Possibly think about averaging
                                  final states to make this number of sample
                                  independent
        """
        # Make sure that we've initialized the tparams
        assert len(self.tparams) > 0
        # State below : steps x samples x dim_proj
        # If n_steps is not provided, infer it
        if n_steps is None:
            nsteps = state_below.shape[0]
        else:
            # If n_steps is provided, this is the incomplete input setting
            # Make sure that a function is provided to transform output
            # from previous time step to input
            # TODO: This output function may require input from several time
            # steps instead of just the previous one. Make this modification
            nsteps = n_steps
            if output_to_input_func is None:
                raise Exception('n_steps was given to the GRU but no output \
                                 to input function was specified')

        # Hack to make sure that the theano ifelse compiles
        if output_to_input_func is None:
            output_to_input_func = dummy_func

        # Check if the input is a batch or a single sample
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1
            warnings.warn("You seem to be supplying single samples for \
                           recurrence. You may see speedup gains with using \
                           batches instead.")

        # Initialize initial hidden state if not specified
        # Restore final hidden state to new initial hidden state
        if restore_final_to_initial_hidden and self.h_final is not None:
            h0 = self.h_final
        else:
            h0 = T.alloc(numpy_floatX(0.),
                         n_samples,
                         self.dim_proj)

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        # TODO: Initialize mask if it is none
        # TODO; Make the same change to the LSTM module
        def _step(t_, h_, mask, state_below, state_below_h_c):
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
            r = T.nnet.sigmoid(_slice(preact, 0, self.dim_proj))
            z = T.nnet.sigmoid(_slice(preact, 1, self.dim_proj))

            # The proposal hidden state
            preact_h = T.dot(h_, self.tparams[_p(self.prefix, 'U_h')])
            preact_h = preact_h * r

            h_c_ = ifelse(T.lt(t_, state_below_h_c.shape[0]),
                               state_below_h_c[t_],
                               T.dot(output_to_input_func(h_), self.tparams[_p(self.prefix, 'W_h')])
                                     + self.tparams[_p(self.prefix, 'b_h')]
                            )
            # TODO : xx_
            preact_h = preact_h + h_c_
            h = T.tanh(preact_h)

            h = z * h_ + (1 - z) * h
            # None adds a dimension to the mask (N,) -> (N, 1)
            # Where the mask value is 1, use the value of the current
            # context, otherwise use the one from the previous
            # context when the mask value is 0
            # This will ensure that values generated for absent
            # elements marked with <PAD> will not be used
            # Similarly, Where the mask value is 1, use the value of the current
            # hidden state, otherwise use the one from the previous
            # state when the mask value is 0
            h = ifelse(T.lt(t_, state_below.shape[0]),
                       mask[t_][:, None] * h + (1. - mask[t_])[:, None] * h_,
                       h)

            return h

        state_below = (T.dot(state_below, self.tparams[_p(self.prefix, 'W')]) +
                       self.tparams[_p(self.prefix, 'b')])
        # Transformation to calculate the candidate hidden state
        state_below_h_c = (T.dot(state_below, self.tparams[_p(self.prefix, 'W_h')]) +
                           self.tparams[_p(self.prefix, 'b_h')])
        rval, updates = theano.scan(_step,
                                    sequences=[T.arange(nsteps)],
                                    outputs_info=[h0],
                                    non_sequences=[mask, state_below, state_below_h_c],
                                    name=_p(self.prefix, '_layers'),
                                    n_steps=nsteps)
        # Save the final state to be used as the next initial hidden state
        if restore_final_to_initial_hidden:
            self.h_final = rval[0][-1]

        # Returns a list of the hidden states (t elements of N x dim_proj)
        return rval[0]


def _p(pp, name):
    return '%s_%s' % (pp, name)


def dummy_func(_input):
    return _input
