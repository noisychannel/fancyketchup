import os
import numpy
import theano
import theano.tensor as T

from cutils.data_interface.interface import DataInterface
from cutils.data_interface.utils import create_subset
from cutils.dict import Dict
from cutils.numeric import numpy_floatX


class SetTimes(DataInterface):
    def __init__(self, dataset_path, n_words=100000, emb_dim=100):
        self.dataset_path = dataset_path
        if os.path.isfile(os.getcwd() + "/" + dataset_path):
            self.dataset_path = os.getcwd() + "/" + dataset_path
        if not os.path.isfile(dataset_path):
            raise Exception('Dataset not found and the origin was not specified')
        self.embedding_dimension = emb_dim
        self.dictionary = None
        print('...Building Dictionary')
        self.build_dict(n_words)
        print('...Done building Dictionary')

    def get_dataset_file(self):
        raise NotImplementedError

    def load_data(self, context_size=4, valid_portion=0.1):
        sentences = []
        prefix = ' '.join(['<BOS>'] * context_size) + ' '
        suffix = ' <EOS>'
        with open(self.dataset_path) as f:
            for l in f:
                sentences.append(prefix + l.strip() + suffix)
        train_x = []
        train_y = []
        for ss in sentences:
            ss_d = self.dictionary.read_sentence(ss)
            windows = [(ss_d[i:i+context_size], ss_d[i+context_size])
                       for i in range(len(ss_d) - context_size)]
            for w in windows:
                train_x.append(w[0])
                train_y.append(w[1])
        train_set = (train_x, train_y)

        valid_set = ([], [])
        if valid_portion > 0.:
            train_set, valid_set = create_subset(train_set, valid_portion)

        return train_set, valid_set

    def build_dict(self, n_words):
        sentences = []
        prefix = "$BOS$ $BOS$ $BOS$ $BOS$ "
        suffix = " $EOS$"
        with open(self.dataset_path) as f:
            for l in f:
                sentences.append(prefix + l.strip() + suffix)
        self.dictionary = Dict(sentences, n_words, self.embedding_dimension)
