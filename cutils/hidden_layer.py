import numpy
import theano
import theano.tensor as T
import warnings


class HiddenLayer(object):
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
                W_values = self.xavier_init(rng, n_in, n_out, activation)
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
        self.params = [self.W, self.b]

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

    def xavier_init(self, rng, n_in, n_out, activation):
        """
        Returns a matrix (n_in X n_out) based on the
        Xavier initialization technique
        """
        if activation not in [T.tanh, T.nnet.sigmoid]:
            warnings.warn("You are using the Xavier init with an \
                           activation function that is not tanh \
                           or sigmoid.")
        W_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out),
            ),
            dtype=theano.config.floatX
        )
        if activation == T.nnet.sigmoid:
            return W_values * 4
        return W_values
