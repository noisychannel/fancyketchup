import sys
import os
import theano
import theano.tensor as T
import numpy
import random

from cutils.dict import Dict

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

    def bucket_and_pad(x, y, buckets):
        """
        Assumes x to be in a list of
        emb_size X len_sent arrays

        """
        for sample, label in zip(x, y):
            length_sample = sample.shape[1]
            b = length_sample - length_sample % 10 + 10
            # Create mask
            mask = numpy.ones(length_sample)
            # Pad sample and mask to bucket length
            padded_sample = numpy.pad(sample, ((0, 0), (0, b - length_sample)),
                                      'constant', constant_values=(0))
            padded_mask = numpy.pad(mask, ((0, b - length_sample)),
                                    'constant', constant_values=(0))
            if b in buckets:
                buckets[b][0].append(padded_sample)
                buckets[b][1].append(label)
                buckets[b][2].append(padded_mask)

    # Bucket data: x, y, masks
    train_buckets = {x: [[], [], []] for x in range(10, 60, 10)}
    valid_buckets = {x: [[], [], []] for x in range(10, 60, 10)}
    test_buckets = {x: [[], [], []] for x in range(10, 60, 10)}
    bucket_and_pad(train_x, train_y, train_buckets)
    bucket_and_pad(valid_x, valid_y, valid_buckets)
    bucket_and_pad(test_x, test_y, test_buckets)
    print "Distribution of entries in the training buckets"
    print [len(train_buckets[x][0]) for x in range(10, 60, 10)]
    print "Distribution of entries in the valid buckets"
    print [len(valid_buckets[x][0]) for x in range(10, 60, 10)]
    print "Distribution of entries in the test buckets"
    print [len(test_buckets[x][0]) for x in range(10, 60, 10)]

    # Shuffle the training buckets
    for b in range(10, 60, 10):
        combined_training = zip(train_buckets[b][0], train_buckets[b][1])
        random.shuffle(combined_training)
        train_buckets[b][0][:], train_buckets[b][1][:] = zip(*combined_training)

    def shared_dataset(dataset_x, dataset_y, borrow=True):
        """ Load the dataset into shared variables """
        assert len(dataset_x) == len(dataset_y)
        shared_x = [theano.shared(x, borrow=borrow) for x in dataset_x]
        shared_y = theano.shared(numpy.asarray(dataset_y),
                                 borrow=borrow)
        # Cast the labels as int32, so that they can be used as indices
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_x, train_y)
    assert numpy.array_equal(train_x[0], train_set_x[0].get_value())
    valid_set_x, valid_set_y = shared_dataset(valid_x, valid_y)
    test_set_x, test_set_y = shared_dataset(test_x, test_y)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

if __name__ == '__main__':
    embedding_size = 3
    rng = numpy.random.RandomState(12321)
    word_dict = Dict(embedding_size, rng)
    datasets = load_data(sys.argv[1], word_dict)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    encoder = Encoder(rng)
