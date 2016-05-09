import numpy

from numeric import numpy_floatX


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
                return self.embeddings[self.words[word]]
        else:
            raise Exception("Trying to add a word to a locked dictionary")

    def lock(self):
        self.lock = True

    def get_new_embedding(self):
        return numpy_floatX(self.rng.uniform(
            low=-1., high=1.,
            size=(self.embedding_size)
            )
        )

    def read_sentence(self, line):
        line = line.strip().split()
        sequence = []
        for word in line:
            if word != "":
                sequence.append(self.add_word(word))
        return numpy.column_stack(sequence)

    def num_words(self):
        return self.next_word_index

    def get_embedding(self, word):
        if word not in self.words:
            word = "UNK"
        return self.embeddings[self.words[word]]
