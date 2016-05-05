import sys
import theano
import theano.tensor as T
import numpy
import random


class Dict:
    def __init__(self, embedding_size, rng):
        self.locked = False
        self.rng = rng
        self.embedding_size = embedding_size
        self.words = {"UNK": 0}
        self.embeddings = [self.get_new_embedding()]
        self.next_word_index = 1

    def add_word(self, word):
        if not self.locked:
            if word not in self.words:
                self.words[word] = self.next_word_index
                self.embeddings.append(self.get_new_embedding())
                self.next_word_index += 1
                return self.embeddings[-1]
        else:
            raise Exception("Trying to add a word to a locked dictionary")

    def lock(self):
        self.lock = True

    def get_new_embedding(self):
        return self.rng.uniform(low=-1., high=1., size=(self.embedding_size)
                                ).astype(theano.config.floatX)

    def read_sentence(self, line):
        line = line.strip().strip()
        sequence = []
        for word in line:
            if word != "":
                sequence.append(self.add_word(word))
        return sequence

    def num_words(self):
        return self.next_word_index


def load_data(fileloc, word_dict):
    train_x = train_y = []
    valid_x = valid_y = []
    test_x = test_y = []
    for ext in [("pos", 1), ("neg", 0)]:
        data_x = data_y = []
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
    combined_training = zip(train_x, train_y)
    random.shuffle(combined_training)
    train_x[:], train_y[:] = zip(*combined_training)

    def shared_dataset(data_x, data_y, borrow=True):
        """ Load the dataset into shared variables """
        assert len(data_x) == len(data_y)
        shared_x = theano.shared(numpy.vstack(data_x),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.vstack(data_y),
                                 borrow=borrow)
        # Cast the labels as int32, so that they can be used as indices
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_x, train_y)
    valid_set_x, valid_set_y = shared_dataset(valid_x, valid_y)
    test_set_x, test_set_y = shared_dataset(test_x, test_y)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

if __name__ == '__main__':
    embedding_size = 50
    rng = numpy.random.RandomState(12321)
    word_dict = Dict(embedding_size, rng)
    load_data(sys.argv[1], word_dict)
