import numpy
import theano
import theano.tensor as T

from cutils.params.init import xavier_init


class DenseLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, w_init="xavier", b_init="zero"):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer

        :type init: string
        :param init: The type of initialization to use
        """
        self.input = input
        if W is None:
            if w_init == "xavier":
                W_values = xavier_init(rng, n_in, n_out, activation)
                W = theano.shared(value=W_values, name='W', borrow=True)
            else:
                raise NotImplementedError()

        if b is None:
            if b_init == "zero":
                b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
                b = theano.shared(value=b_values, name='b', borrow=True)
            else:
                raise NotImplementedError()

        self.W = W
        self.b = b
        # Parameters for batch normalization follow
        self.gamma = theano.shared(
            value=numpy.ones(
                (n_out,),
                dtype=theano.config.floatX), name='gamma')
        self.beta = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX), name='beta')

        self.params = [self.W, self.b, self.gamma, self.beta]

        lin_output = T.dot(input, self.W) + self.b
        # Apply batch normalization to the linear output
        bn_output = T.nnet.batch_normalization(
            inputs=lin_output,
            gamma=self.gamma,
            beta=self.beta,
            mean=lin_output.mean((0,), keepdims=True),
            std=lin_output.std((0,), keepdims=True),
            mode='low_mem'
        )

        self.output = (
            bn_output if activation is None
            else activation(bn_output)
        )
