import theano
import theano.tensor as T
import numpy

from cutils.loss_functions import negative_log_likelihood, zero_one_loss, \
    nce_binary_conditional_likelihood


from cutils.regularization import L1, L2

class LogisticRegression(object):
    """
    Multi-class logistic regression
    """

    def __init__(self, input, n_in, n_out):
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
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX,
            ),
            name='W',
            borrow=True
        )
        # Initialize the bias vector to 0s. Size is n_out
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX,
            ),
            name='b',
            borrow=True
        )

        # Compute (symbolic) : softmax(x.W + b)
        lin_output = T.dot(input, self.W) + self.b
        # Parameters for batch normalization follow
        self.gamma = theano.shared(
            value=numpy.ones(
                (n_out,),
                dtype=theano.config.floatX), name='gamma')
        self.beta = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX), name='beta')

        # Apply batch normalization to the linear output
        bn_output = T.nnet.batch_normalization(
            inputs=lin_output,
            gamma=self.gamma,
            beta=self.beta,
            mean=lin_output.mean((0,), keepdims=True),
            std=T.ones_like(lin_output.var((0,), keepdims=True)),
            mode='high_mem'
        )

        lin_output.std((0,))

        self.lin_output = bn_output

        # Softmax components computed on demand
        self.p_y_given_x = None
        self.y_pred = None

        # Params of the model
        self.params = [self.W, self.b, self.gamma, self.beta]

        self.input = input

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
