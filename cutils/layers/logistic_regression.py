import theano
import theano.tensor as T
import numpy

from cutils.loss_functions import negative_log_likelihood, zero_one_loss, \
    nce_binary_conditional_likelihood
from cutils.params.init import norm_init
from cutils.params.utils import init_tparams
from cutils.regularization import L1, L2

class LogisticRegression(object):
    """
    Multi-class logistic regression
    """

    def __init__(self, dim_proj, dim_input, prefix='logit', ortho=True):
        """
        Initializes the parameters of a Logistic regression model

        :type input: theano.tensor.TensorType
        :param input: symbolic variable for one input batch, one row per sample

        :type n_in: int
        :param n_int: The dimensionality of the input layer

        :type n_out
        :param n_out: The dimensionality of the output (label) layer
        """
        # Initialize weight matrix with 0s. Size is n_in X n_out
        self.param_names = []
        params = OrderedDict()

        W = norm_init(dim_input, dim_proj, ortho)
        params[_p(prefix, 'W')] = W
        self.param_names.append(_p(prefix, 'W'))

        b = numpy_floatX(numpy.zeros(dim_proj,))
        params[_p(prefix, 'b')] = b
        self.param_names.append(_p(prefix, 'b'))

        # batch normalization params
        gamma = numpy_floatX(numpy.ones((n_out,)))
        params[_p(prefix, 'gamma')] = gamma
        self.param_names.append(_p(prefix, 'gamma'))

        beta = numpy_floatX(numpy.ones((n_out,)))
        params[_p(prefix, 'beta')] = beta
        self.param_names.append(_p(prefix, 'beta'))

        self.prefix = prefix
        self.params = params
        self.tparams = init_tparams(params)

        # Legacy params
        # Softmax components computed on demand
        self.p_y_given_x = None
        self.y_pred = None
        self.lin_output = None


    def logit_layer(self, input, batch_normalize=False):
        # Compute (symbolic) : softmax(x.W + b)
        lin_output = T.dot(input, self.tparams[_p(self.prefix, 'W')]) + self.tparams[_p(self.prefix, 'b')]

        # Parameters for batch normalization follow
        if batch_normalize:
            # Apply batch normalization to the linear output
            bn_output = T.nnet.batch_normalization(
                inputs=lin_output,
                gamma=self.gamma,
                beta=self.beta,
                mean=lin_output.mean((0,), keepdims=True),
                std=T.ones_like(lin_output.var((0,), keepdims=True)),
                mode='high_mem'
            )
            lin_output = bn_output

        self.lin_output = lin_output
        return lin_output


    def loss(self, y):
        """
        Returns the loss (negative-log-likelihood) over the mini-batch

        :type y: theano.tensor.TensorType
        :param y: The true vectors correspoding to the input examples in this
            batch
        """
        # Compute the softmax on the final layer on demand
        if self.p_y_given_x is None:
            self.p_y_given_x = T.nnet.softmax(self.lin_output)
        return negative_log_likelihood(self.p_y_given_x, y)

    def nce_loss(self, y, y_flat, noise_samples, noise_dist, k):
        """
        Returns the binary NCE loss for examples

        :type y: theano.tensor.TensorType
        :param y: The true vectors correspoding to the input examples in this
            batch

        :type noise_samples: theano.tensor.TensorType
        :param noise_samples: The noisy samples drawn from the vocab

        :type noise_dist: theano.tensor.TensorType
        :param noise_dist: The noise distribution for NCE
        """
        return nce_binary_conditional_likelihood(
            self.lin_output, y, y_flat, noise_samples, noise_dist, k)

    def errors(self, y):
        """
        Returns the 0-1 loss over the size of the mini-batch

        :type y: theano.tensor.TensorType
        :param y: The true vectors correspoding to the input examples in this
            batch
        """
        # Compute the softmax on the final layer on demand
        if self.p_y_given_x is None:
            self.p_y_given_x = T.nnet.softmax(self.lin_output)
        # Symbolic prediction (argmax over columns)
        if self.y_pred is None:
            self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # Make sure that y has the same dimensionality as y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y has the correct datatype
        if y.dtype.startswith('int'):
            return zero_one_loss(self.y_pred, y)
        else:
            raise NotImplementedError()

def _p(pp, name):
    return '%s_%s' % (pp, name)
