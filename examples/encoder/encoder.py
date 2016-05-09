import theano
import numpy
import theano.tensor as T
from cutils.init import xavier_init


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
        h0 = theano.shared(
            value=numpy.zeros(
                n_hidden,
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        self.h0 = h0
        self.W_x = W_x
        self.W_h = W_h
        self.b = b
        self.params = [self.W_x, self.W_h, self.b]

        # Create tensorvariables to accept input
        x = T.tensor3('x')

        def recurrence(x_t, h_tm1):
            h_t = self.activation(T.dot(self.W_x, x_t) +
                                  T.dot(self.W_h, h_tm1) +
                                  self.b)
            return h_t

        h, _ = theano.scan(fn=recurrence,
                           sequences=x,
                           outputs_info=[self.h0],
                           n_steps=x.shape[1])
        self.h = h
        return self.h

    def get_hidden_states(self):
        if self.h is None:
            raise Exception("The hidden states for this RNN have not \
                             been computed yet")
        return self.h

    def get_context_embedding(self):
        if self.h is None:
            raise Exception("The hidden states for this RNN have not \
                             been computed yet")
        get_hidden_states = theano.function(
            inputs=[index],
            outputs=encoder.compute_hidden_states_no_output(x),
            givens=[
                (x, test_set[10][0][index * batch_size: (index + 1) * batch_size]),
                (m, test_set[10][2][index * batch_size: (index + 1) * batch_size])
            ]
        )

        context = T.sum(get_hidden_states(index))
        log_regression_layer = LogisticRegression(
            input=context,
            n_in=n_hidden,
            n_out=2
        )

