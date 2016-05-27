import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from cutils.layers.dense_layer import DenseLayer
from cutils.layers.logistic_regression import LogisticRegression
from cutils.training.trainer import simple_sgd
from autoencoder import AutoEncoder


class SdA(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10,
        corruption_levels=[0.1, 0.1]
    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """
        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # symbolic variables for the data
        self.x = T.matrix('x')  # rasterized images
        self.y = T.ivector('y')  # labels (int)

        # Create n_layers sigmoid layers and n_layers denoising autoencoders
        for i in range(self.n_layers):
            # Construct the sigmoidal layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # The input to this layer is either the activation of the hidden
            # layer below, or the input of the SdA if this is the firts layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = DenseLayer(
                rng=numpy_rng,
                input=layer_input,
                n_in=input_size,
                n_out=hidden_layers_sizes[i],
                activation=T.nnet.sigmoid
            )

            self.sigmoid_layers.append(sigmoid_layer)
            # Since we are pretraining the encoders, the only
            # parameters that will be fine-tuned in the stacked autoencoder
            # MLP facade are the parameters of the stacked autoencoder
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder which shares weights with this
            # layer. bvis is a parameter of the autoencoder only and is not
            # shared
            dA_layer = AutoEncoder(
                numpy_rng=numpy_rng,
                theano_rng=theano_rng,
                input=layer_input,
                n_visible=input_size,
                n_hidden=hidden_layers_sizes[i],
                W=sigmoid_layer.W,
                bhid=sigmoid_layer.b
            )

            self.dA_layers.append(dA_layer)

        # Add a logistic layer on top of the MLP
        self.log_layer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.log_layer.params)

        self.fine_tune_cost = self.log_layer.loss(self.y)
        self.errors = self.log_layer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''
        # Index to a mini-batch
        index = T.lscalar('index')
        corruption_level = T.scalar('corruption')  # % corruption to use
        learning_rate = T.scalar('learning_rate')  # learning rate

        pretrain_fns = []
        for dA in self.dA_layers:
            cost = dA.loss(corruption_level)
            updates = simple_sgd(cost, dA.params, learning_rate)
            fn = theano.function(
                inputs=[
                    index,
                    theano.In(corruption_level, value=0.2),
                    theano.In(learning_rate, value=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens=[
                    (self.x,
                     train_set_x[index * batch_size: (index + 1) * batch_size])
                ]
            )
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        ''' 
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # Notice that get_value is called with borrow
        # so that a deep copy of the input is not created
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

        print("... Building the model")

        index = T.lscalar()  # index to a mini-batch

        # Compile function that measures test performance wrt the 0-1 loss
        test_model = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens=[
                (self.x, test_set_x[index * batch_size: (index + 1) * batch_size]),
                (self.y, test_set_y[index * batch_size: (index + 1) * batch_size])
            ]
        )
        validate_model = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens=[
                (self.x, valid_set_x[index * batch_size: (index + 1) * batch_size]),
                (self.y, valid_set_y[index * batch_size: (index + 1) * batch_size])
            ]
        )

        # Stochastic Gradient descent
        updates = simple_sgd(self.fine_tune_cost, self.params, learning_rate)

        train_model = theano.function(
            inputs=[index],
            outputs=self.fine_tune_cost,
            updates=updates,
            givens=[
                (self.x, train_set_x[index * batch_size: (index + 1) * batch_size]),
                (self.y, train_set_y[index * batch_size: (index + 1) * batch_size])
            ]
        )

        def valid_scores():
            return [validate_model(i) for i in range(n_valid_batches)]

        def test_scores():
            return [test_model(i) for i in range(n_test_batches)]

        return train_model, valid_scores, test_scores
