import sys
import os
import theano
import theano.tensor as T
import numpy

from cutils.dict import Dict
from cutils.data_utils import bucket_and_pad
from cutils.logistic_regression import LogisticRegression
from cutils.numeric import numpy_floatX
from cutils.trainer import sgd

# Include current path in the pythonpath
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_path)

from encoder import Encoder


def load_data(fileloc, word_dict):
    train_x, train_y, valid_x, valid_y, test_x, test_y = \
        ([] for i in range(6))
    for ext in [("pos", 1), ("neg", 0)]:
        data_x, data_y = ([] for i in range(2))
        with open(fileloc + "." + ext[0]) as f:
            for line in f:
                data_x.append(word_dict.read_sentence(line))
                data_y.append(ext[1])
        train_x.extend(data_x[:4700])
        train_y.extend(data_y[:4700])
        valid_x.extend(data_x[4700:5000])
        valid_y.extend(data_y[4700:5000])
        test_x.extend(data_x[5000:])
        test_y.extend(data_y[5000:])

    # Bucket data: x, y, masks
    train_buckets = {x: [[], [], []] for x in range(10, 60, 10)}
    valid_buckets = {x: [[], [], []] for x in range(10, 60, 10)}
    test_buckets = {x: [[], [], []] for x in range(10, 60, 10)}
    bucket_and_pad(train_x, train_y, train_buckets, consolidate=True)
    bucket_and_pad(valid_x, valid_y, valid_buckets, consolidate=True)
    bucket_and_pad(test_x, test_y, test_buckets, consolidate=True)
    print "Distribution of entries in the training buckets"
    print [train_buckets[x][0].shape for x in range(10, 60, 10)]
    print "Distribution of entries in the valid buckets"
    print [valid_buckets[x][0].shape for x in range(10, 60, 10)]
    print "Distribution of entries in the test buckets"
    print [test_buckets[x][0].shape for x in range(10, 60, 10)]

    def shared_dataset(dataset, borrow=True):
        """ Load the dataset into shared variables """
        shared_bucket = {}
        for b, b_data in dataset.iteritems():
            # Make sure we have the same number of entries
            assert b_data[0].shape[-1] == b_data[1].shape[-1] == \
                b_data[2].shape[-1]
            # Make sure the batch size is correct
            assert b_data[0].shape[1] == b
            shared_x = theano.shared(numpy_floatX(b_data[0]),
                                     borrow=borrow)
            shared_y = theano.shared(numpy_floatX(b_data[1]),
                                     borrow=borrow)
            shared_m = theano.shared(numpy_floatX(b_data[2]),
                                     borrow=borrow)
            shared_bucket[b] = [shared_x, T.cast(shared_y, 'int32'), shared_m]
        return shared_bucket

    train_set = shared_dataset(train_buckets)
    valid_set = shared_dataset(valid_buckets)
    test_set = shared_dataset(test_buckets)
    r_val = [train_set, valid_set, test_set]
    return r_val

if __name__ == '__main__':
    batch_size = 10
    n_input = 3
    n_hidden = 50
    learning_rate = 0.01
    rng = numpy.random.RandomState(12321)
    word_dict = Dict(n_input, rng)
    datasets = load_data(sys.argv[1], word_dict)
    train_set, valid_set, test_set = datasets
    n_train_batches = [train_set[b][0].get_value(borrow=True).shape[-1] // batch_size for b in train_set.keys()]
    n_valid_batches = [valid_set[b][0].get_value(borrow=True).shape[-1] // batch_size for b in valid_set.keys()]
    n_test_batches = [test_set[b][0].get_value(borrow=True).shape[-1] // batch_size for b in test_set.keys()]
    encoder = Encoder(rng, n_hidden=n_hidden, n_input=n_input)

    # Holds indices for the batch
    index = T.lscalar()
    # Emb_dim x sequence_len X batch_size
    x = T.tensor3('x')
    # Batch_size x 1
    y = T.ivector('y')
    # sequence_len x batch_size
    m = T.matrix('m')

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

    cost = log_regression_layer.loss(y)
    #TODO: add dict to the params
    params = encoder.params + log_regression_layer.params
    updates = sgd(cost, params, learning_rate)
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates
    )
