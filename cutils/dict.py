import numpy
import theano
from collections import OrderedDict

from cutils.params.utils import init_tparams
from cutils.numeric import numpy_floatX


class Dict:
    def __init__(self, sentences, n_words, emb_dim):
        self.locked = False
        wordcount = dict()
        for ss in sentences:
            words = ss.strip().split()
            for w in words:
                if w not in wordcount:
                    wordcount[w] = 0
                wordcount[w] += 1
        counts = wordcount.values()
        keys = wordcount.keys()
        self.worddict = dict()
        self.worddict['<UNK>'] = 1
        self.worddict['<PAD>'] = 0
        # Reverse and truncate at max_words
        sorted_idx = numpy.argsort(counts)[::-1][:n_words]
        for idx, ss in enumerate(sorted_idx):
            self.worddict[keys[ss]] = idx + 2

        self.n_words = len(self.worddict)

        self.noise_distribution = None
        self.create_unigram_noise_dist(wordcount, n_words)

        self.locked = True

        print("Total words read by dict = %d" % numpy.sum(counts))
        print("Total unique words read by dict = %d" % len(keys))
        print("Total words retained = %d" % len(self.worddict))

        self.embedding_size = emb_dim
        self.rng = None
        self.Wemb = None
        self.initialize_embedding()

    def create_unigram_noise_dist(self, wordcount, n_words):
        counts = numpy.sort(wordcount.values())[::-1]
        freq = [0, sum(counts[n_words:])] + list(counts[:n_words])
        assert len(freq) == self.n_words
        sum_freq = sum(freq)
        noise_distribution = [float(k) / sum_freq for k in freq]
        self.noise_distribution = init_tparams(
            OrderedDict([('noise_d', numpy_floatX(noise_distribution)
                          .reshape(self.n_words,))])
        )['noise_d']

    def initialize_embedding(self):
        randn = numpy.random.rand(self.n_words, self.embedding_size)
        Wemb = (0.01 * randn).astype(theano.config.floatX)
        self.Wemb = init_tparams(OrderedDict([('Wemb', Wemb)]))['Wemb']

    def read_sentence(self, line):
        line = line.strip().split()
        return [self.worddict[w] if w in self.worddict else 1 for w in line]

    def num_words(self):
        """ + 2 for the UNK symbols """
        return self.n_words

    def get_embedding(self, word):
        if word not in self.words:
            word = "UNK"
        return self.embeddings[self.words[word]]
