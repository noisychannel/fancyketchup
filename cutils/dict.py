import numpy

from numeric import numpy_floatX


class Dict:
    def __init__(self, sentences, n_words):
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
        # Reverse and truncate at max_words
        sorted_idx = numpy.argsort(counts)[::-1][:n_words]
        self.worddict = dict()
        for idx, ss in enumerate(sorted_idx):
            self.worddict[keys[ss]] = idx + 2

        self.locked = True

        print("Total words read by dict = %d" % numpy.sum(counts))
        print("Total unique words read by dict = %d" % len(keys))
        print("Total words retained = %d" % len(self.worddict))

        self.embedding_size = None
        self.rng = None
        self.Wemb = None

    def get_new_embedding(self):
        return numpy_floatX(self.rng.uniform(
            low=-1., high=1.,
            size=(self.embedding_size)
            )
        )

    def read_sentence(self, line):
        line = line.strip().split()
        return [self.worddict[w] if w in self.worddict else 1 for w in line]

    def num_words(self):
        return len(self.worddict)

    def get_embedding(self, word):
        if word not in self.words:
            word = "UNK"
        return self.embeddings[self.words[word]]
