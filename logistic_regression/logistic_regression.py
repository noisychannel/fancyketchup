import theano
import theano.tensor as T
import numpy

from loss_functions import negative_log_likelihood, zero_one_loss


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
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # Symbolic prediction (argmax over columns)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # Params of the model
        self.params = [self.W, self.b]

        self.input = input

    def loss(self, y):
        """
        Returns the loss (negative-log-likelihood) over the mini-batch

        :type y: theano.tensor.TensorType
        :param y: The true vectors correspoding to the input examples in this
            batch
        """
        return negative_log_likelihood(self.p_y_given_x, y)

    def errors(self, y):
        """
        Returns the 0-1 loss over the size of the mini-batch

        :type y: theano.tensor.TensorType
        :param y: The true vectors correspoding to the input examples in this
            batch
        """
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
