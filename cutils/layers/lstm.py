import warnings

import numpy
import theano
import theano.tensor as T
from collections import OrderedDict

from cutils.params.init import ortho_weight
from cutils.params.utils import init_tparams
from cutils.numeric import numpy_floatX


class LSTM(object):
    def __init__(self, dim_proj, rng, prefix='lstm'):
        """
        Initialize the LSTM params
        """
        self.param_names = []
        params = OrderedDict()
        W = numpy.concatenate([ortho_weight(dim_proj),
                               ortho_weight(dim_proj),
                               ortho_weight(dim_proj),
                               ortho_weight(dim_proj)], axis=1)
        params[_p(prefix, 'W')] = W
        self.param_names.append(_p(prefix, 'W'))

        # TODO: why is the axis 1, figure out
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

    def lstm_layer(self, state_below, dim_proj, mask=None):
        """
        Recurrence with an LSTM hidden unit

        state_below : Is the input. This may be a single sample with
                      multiple timesteps, or a batch
        dim_proj : The dimensionality of the hidden units (projection)
        mask : The mask applied to the input for batching
        """
        # Make sure that we've initialized the tparams
        assert len(self.tparams) > 0
        # State below : steps x samples
        # Recurrence over dim 0
        nsteps = state_below.shape[0]
        # Check if the input is a batch or a single sample
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1

        if mask is None:
            warnings.warn("You seem to be supplying single samples for \
                           recurrence. You may see speedup gains with using \
                           batches instead.")

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _step(m_, x_, h_, c_):
            preact = T.dot(h_, self.tparams[_p(self.prefix, 'U')])
            preact += x_

            i = T.nnet.sigmoid(_slice(preact, 0, dim_proj))
            f = T.nnet.sigmoid(_slice(preact, 1, dim_proj))
            o = T.nnet.sigmoid(_slice(preact, 2, dim_proj))
            c = T.tanh(_slice(preact, 3, dim_proj))
            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * T.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c

        state_below = (T.dot(state_below, self.tparams[_p(self.prefix, 'W')]) +
                       self.tparams[_p(self.prefix, 'b')])
        rval, updates = theano.scan(_step,
                                    sequences=[mask, state_below],
                                    outputs_info=[T.alloc(numpy_floatX(0.),
                                                          n_samples,
                                                          dim_proj),
                                                  T.alloc(numpy_floatX(0.),
                                                          n_samples,
                                                          dim_proj)],
                                    name=_p(self.prefix, '_layers'),
                                    n_steps=nsteps)
        return rval[0]


def _p(pp, name):
    return '%s_%s' % (pp, name)
