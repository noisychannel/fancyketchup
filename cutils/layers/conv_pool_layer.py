import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d
from init import xavier_init


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        assert image_shape[1] == filter_shape[1]
        self.input = input

        # No. of inputs to a hidden unit =
        # input_feature_maps * filter_height * filter_width
        fan_in = numpy.prod(filter_shape[1:])
        # Each unit in the lower layer recieves gradients
        # from num_output_feature_maps * filter_height * filter_width
        # / pooling_size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        W_values = xavier_init(rng, fan_in, fan_out, T.tanh, filter_shape)
        self.W = theano.shared(W_values, borrow=True)
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(b_values, borrow=True)

        # Convolution operation (input feature maps with filters)
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # Apply max-pooling (downsample)
        # Notice that instead of padding 0s, the border is ignored
        pooled_out = pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # Dimshuffle bias vector to allow one bias term per filter
        # The rest will be handled via broadcasting
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

        self.input = input
