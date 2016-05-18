import os
import theano
import theano.tensor as T

import cutils.regularization as reg
from cutils.layers.dense_layer import DenseLayer
from cutils.layers.logistic_regression import LogisticRegression
from cutils.layers.utils import dropout_layer
from cutils.numeric import numpy_floatX

# Include logistic_regressioncurrent path in the pythonpath
script_path = os.path.dirname(os.path.realpath(__file__))


class NPLM(object):
    """
    Neural Network Language Model

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_h1, n_h2, n_out,
                 use_dropout=False, trng=None, dropout_p=0.5,
                 use_noise=theano.shared(numpy_floatX(0.))):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # This first hidden layer
        # The input is the concatenated word embeddings for all
        # words in the context input and the batch
        self.h1 = DenseLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_h1,
            activation=T.nnet.relu
        )

        # Use dropout if specified
        h2_input = self.h1.output
        if (use_dropout):
            assert trng is not None
            h2_input = dropout_layer(self.h1.output, use_noise,
                                     trng, dropout_p)

        # The second hidden layer
        self.h2 = DenseLayer(
            rng=rng,
            input=h2_input,
            n_in=n_h1,
            n_out=n_h2,
            activation=T.nnet.relu
        )

        # Apply dropout if specified
        log_reg_input = self.h2.output
        if (use_dropout):
            log_reg_input = dropout_layer(self.h2.output, use_noise,
                                          trng, dropout_p)

        # The logistic regression layer
        self.log_regression_layer = LogisticRegression(
            input=log_reg_input,
            n_in=n_h2,
            n_out=n_out
        )

        # Use L2 regularization, for the log-regression layer only
        self.L2 = reg.L2([self.log_regression_layer.W])
        # Get the NLL loss function from the logistic regression layer
        self.loss = self.log_regression_layer.loss

        # Bundle params (to be used for computing gradients)
        self.params = self.h1.params + self.h2.params + \
            self.log_regression_layer.params

        # Keeo track of the input (For debugging only)
        self.input = input
