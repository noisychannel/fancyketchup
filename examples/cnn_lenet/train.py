import os
import sys
import theano
import theano.tensor as T
import cPickle
import gzip
import numpy
import timeit

from cutils.trainer import sgd
from cutils.hidden_layer import HiddenLayer
from cutils.lenet_conv_pool_layer import LeNetConvPoolLayer

# Include current path in the pythonpath
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_path + "/../logistic_regression")

from logistic_regression import LogisticRegression


def load_data(dataset_location):
    # Load the dataset
    dataset_location = sys.argv[1]
    f = gzip.open(dataset_location, "rb")
    try:
        train_set, valid_set, test_set = cPickle.load(f, encoding="latin1")
    except:
        train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        """ Load the dataset into shared variables """
        data_x, data_y = data_xy
        assert len(data_x) == len(data_y)
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # Cast the labels as int32, so that they can be used as indices
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def sgd_optimization_mnist_mlp(learning_rate=0.1, nkerns=[20, 50],
                               n_epochs=200, dataset='mnist.pkl.gz',
                               batch_size=500, n_hidden=500):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # Notice that get_value is called with borrow
    # so that a deep copy of the input is not created
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print("... Building the model")

    index = T.lscalar()  # index to a mini-batch

    # Symbolic variables for input and output for a batch
    x = T.matrix('x')
    y = T.ivector('y')

    rng = numpy.random.RandomState(23455)

    # Reshape input to fit the shape of the image_shape
    # (batch_size, num_input_feature_maps, image_height, image_width)
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # First convolution layer
    # Applies 5x5 filters to the input image
    # Reduces the size of the image to (28-5+1, 28-5+1) = (24,24)
    # Maxpooling reduces this to (12/12)
    # Output size is then (batch_size, nkerns[0], 12, 12)
    # Note: For each image, multiple feature maps are extracted
    # Each feature map is of size f_height * f_width
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # Second conv layer
    # Each feature map convolves input from each input
    # feature map and applies pooling.
    # Output size = (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # Flatten the output of layer1 for a standard fully
    # connected hidden layer (batch_size x input)
    # input = nkerns[1] * 4 * 4
    # Flatten retains the leading ndim-1 dimensions and merges the rest
    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=n_hidden,
        activation=T.tanh
    )

    # Apply logistic regression to perform classification
    classifier = LogisticRegression(input=layer2.output,
                                    n_in=n_hidden, n_out=10)

    # Cost to minimize
    cost = classifier.loss(y)

    # Compile function that measures test performance wrt the 0-1 loss
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens=[
            (x, test_set_x[index * batch_size: (index + 1) * batch_size]),
            (y, test_set_y[index * batch_size: (index + 1) * batch_size])
        ]
    )
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens=[
            (x, valid_set_x[index * batch_size: (index + 1) * batch_size]),
            (y, valid_set_y[index * batch_size: (index + 1) * batch_size])
        ]
    )

    # Stochastic Gradient descent
    updates = sgd(cost, classifier.params, learning_rate)

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens=[
            (x, train_set_x[index * batch_size: (index + 1) * batch_size]),
            (y, train_set_y[index * batch_size: (index + 1) * batch_size])
        ]
    )

    ################
    # TRAIN MODEL  #
    ################
    print("... Training the model")
    # Early stopping parameters
    patience = 10000  # Look at these many parameters regardless
    # Increase patience by this quantity when a best score is achieved
    patience_increase = 2
    improvement_threshold = 0.995  # Minimum significant improvement
    validation_frequency = min(n_train_batches, patience // 2)
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # Iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            # Check if validation needs to be performed
            if (iter + 1) % validation_frequency == 0:
                # Compute average 0-1 loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # Check if this is the best validation score
                if this_validation_loss < best_validation_loss:
                    # Increase patience if gain is gain is significant
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    # Get test scores
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        'epoch %i, minibatch %i/%i, test error of'
                        ' best model %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )
                    # Save the best model
                    #with open(script_path + '/best_model_mlp.pkl', 'wb') as f:
                        #cPickle.dump(classifier, f)

        if patience <= iter:
            done_looping = True
            break
    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation error of %f %%,'
            'with test error of %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print ('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))


if __name__ == '__main__':
    sgd_optimization_mnist_mlp(dataset=sys.argv[1])
