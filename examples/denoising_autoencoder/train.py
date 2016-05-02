import os
import sys
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import cPickle
import gzip
import numpy
import timeit
from PIL import Image

from cutils.trainer import sgd
from cutils.utils import tile_raster_images

# Include current path in the pythonpath
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_path)

from autoencoder import AutoEncoder


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


def sgd_optimization_mnist_da(learning_rate=0.1, n_epochs=15,
                              dataset='mnist.pkl.gz', batch_size=20):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # Notice that get_value is called with borrow
    # so that a deep copy of the input is not created
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    print("... Building the model")

    index = T.lscalar()  # index to a mini-batch

    # Symbolic variables for input and output for a batch
    x = T.matrix('x')

    def sgd_da(corruption_level):
        # Initialize RNGs
        rng = numpy.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))

        # Build the logistic regression class
        # Images in MNIST are 28*28, there are 10 output classes
        da = AutoEncoder(
            numpy_rng=rng,
            theano_rng=theano_rng,
            input=x,
            n_visible=28 * 28,
            n_hidden=500
        )

        # Cost to minimize
        cost = da.loss(corruption_level=corruption_level)

        # Stochastic Gradient descent
        updates = sgd(cost, da.params, learning_rate)

        train_da = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens=[
                (x, train_set_x[index * batch_size: (index + 1) * batch_size]),
            ]
        )

        ################
        # TRAIN MODEL  #
        ################
        start_time = timeit.default_timer()
        print("... Training the model")
        for epoch in range(n_epochs):
            c = []
            for batch_index in range(n_train_batches):
                c.append(train_da(batch_index))

            print('Training epoch {0}, cost {1}'.format(epoch, numpy.mean(c)))

        end_time = timeit.default_timer()
        training_time = (end_time - start_time)

        print(('The {0} corruption code for file ' +
              os.path.split(__file__)[1] +
              ' ran for {1:.2f}m').format(corruption_level * 100,
                                     (training_time) / 60.))

        image = Image.fromarray(
            tile_raster_images(X=da.W.get_value(borrow=True).T,
                               img_shape=(28, 28), tile_shape=(10, 10),
                               tile_spacing=(1, 1))
        )
        image.save('filters_corrpution_' + str(int(corruption_level * 100))
                   + '.png')

    # Train model for no corrution
    sgd_da(0.)
    # Train a model with 30% corruption
    sgd_da(0.3)



if __name__ == '__main__':
    sgd_optimization_mnist_da(dataset=sys.argv[1])
