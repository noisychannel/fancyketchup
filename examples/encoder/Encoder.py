import theano
import numpy
import theano.tensor as T
from cutils.utils import xavier_init


class Encoder:
    """
    Implements an RNN encoder
    """

    def __init__(
        self,
        numpy_rng,
        n_hidden=50,
        n_input=50,
        activation=T.nnet.relu
    ):
        '''
        h_t = f(W_x * x_t + W_h * h_t + b)
        W_x \in R_{hs X w_emb}
        x_t \in R_{w_emb}
        '''
        self.activation = activation
        self.n_hidden = n_hidden
        self.n_input = n_input

        W_x_inital = xavier_init(numpy_rng, n_hidden, n_input, activation)
        W_x = theano.shared(value=W_x_inital, name='W_x', borrow=True)
        W_h_initial = xavier_init(numpy_rng, n_hidden, n_hidden, activation)
        W_h = theano.shared(value=W_h_initial, name='W_h', borrow=True)
        if activation == T.nnet.relu:
            # Small bias initialization
            b = theano.shared(
                value=numpy.ones(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            # Zero initialize
            b = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        self.W_x = W_x
        self.W_h = W_h
        self.b = b
        self.params = [self.W_x, self.W_h, self.b]

    def get_hidden_states(self, input):
        assert input.shape[0] == self.n_input

        def recurrence(x_t, h_tm1):
            h_t = self.activation(T.dot(self.W_x, x_t) +
                                  T.dot(self.W_h, h_tm1) +
                                  self.b)
            return h_t

        h, _ = theano.scan(fn=recurrence,
                           sequences=input,
                           outputs_info=[self.h0, None],
                           n_steps=input.shape[1])
