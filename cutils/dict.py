"""
Contains the dictionary class which is responible for maintaining the
vocabulary and word embeddings
"""
from __future__ import print_function
from collections import OrderedDict

import numpy
import theano

from cutils.params.utils import init_tparams
from cutils.numeric import numpy_floatX


class Dict(object):
    """
    The dictionary is responsible for reading text and converting them into
    integer tokens (creating a vocabulary).
    It will also initialize random word embeddings.

    Helper functions are available to get unigram noise distributions for the
    vocabulary (to be used with NCE).
    """
    def __init__(self, sentences, n_words, emb_dim):
        """
        Initializes a dictionary.

        :type sentences: list(strings)
        :param sentences: A list of sentences (text) to
            initialize the vocabulary

        :type n_words : int
        :param n_words : The number of words to retain in the vocab. Less
            frequent words that are below this threshold will be replaced
            with UNK

        :type emb_dim: int
        :param emb_dim: The dimension for the word embeddings
        """
        self.locked = False
        wordcount = dict()
        for ss_ in sentences:
            words = ss_.strip().split()
            for word in words:
                if word not in wordcount:
                    wordcount[word] = 0
                wordcount[word] += 1
        counts = wordcount.values()
        keys = wordcount.keys()
        self.worddict = dict()
        self.reverse_worddict = dict()
        self.worddict['<UNK>'] = 1
        self.reverse_worddict[1] = '<UNK>'
        self.worddict['<PAD>'] = 0
        self.reverse_worddict[0] = '<PAD>'
        # Reverse and truncate at max_words
        sorted_idx = numpy.argsort(counts)[::-1][:n_words]
        for idx_, ss_ in enumerate(sorted_idx):
            if keys[ss_] == '<UNK>':
                continue
            self.worddict[keys[ss_]] = idx_ + 2
            self.reverse_worddict[idx_ + 2] = keys[ss_]

        self.n_words = len(self.worddict)

        self.noise_distribution = None
        self.create_unigram_noise_dist(wordcount)

        self.locked = True

        print("Total words read by dict = %d" % numpy.sum(counts))
        print("Total unique words read by dict = %d" % len(keys))
        print("Total words retained = %d" % len(self.worddict))

        self.embedding_size = emb_dim
        # TODO: Remove self.Wemb at some point, it should be part of params
        w_emb = self.initialize_embedding()
        params = OrderedDict()
        params['Wemb'] = w_emb
        self.params = params
        self.tparams = init_tparams(params)

    def create_unigram_noise_dist(self, wordcount):
        """
        Creates a Unigram noise distribution for NCE

        :type wordcount: dict
        :param wordcount: A dictionary containing frequency counts for words
        """
        counts = numpy.sort(wordcount.values())[::-1]
        # Don't count the UNK and PAD symbols in the second count
        freq = [0, sum(counts[self.n_words:])] \
            + list(counts[:(self.n_words-2)])
        assert len(freq) == self.n_words
        sum_freq = sum(freq)
        noise_distribution = [float(k) / sum_freq for k in freq]
        self.noise_distribution = init_tparams(
            OrderedDict([('noise_d', numpy_floatX(noise_distribution)
                          .reshape(self.n_words,))])
        )['noise_d']

    def initialize_embedding(self):
        """
        Initializes the word embeddings from a uniform distribution
        """
        # TODO: Which random seed is used here?
        randn = numpy.random.rand(self.n_words, self.embedding_size)
        w_emb = (0.01 * randn).astype(theano.config.floatX)
        return w_emb

    def read_sentence(self, line):
        """
        Reads a sentence (text) and converts in into a list of int tokens

        :type line: string
        :param line: The string to be read
        """
        line = line.strip().split()
        return [self.worddict[w] if w in self.worddict else 1 for w in line]

    def num_words(self):
        """
        Returns the number of words in the vocabulary
        """
        return self.n_words

    def idx_to_words(self, idx_arr):
        """
        idx_array is TxN. Each col is a sentence
        """
        results = []
        for col in idx_arr:
            sentence = []
            for k in col:
                sentence.append(self.reverse_worddict[k])
            results.append(" ".join(sentence))

        return results

