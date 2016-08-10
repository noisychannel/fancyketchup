"""
Data inteface for the Penn TreeBank LM dataset
"""

from __future__ import print_function

from cutils.data_interface.interface import DataInterface
import cutils.data_interface.utils as du
from cutils.dict import Dict


class PTB(DataInterface):
    def __init__(self, dataset_path, origin=None, n_words=100000, emb_dim=100):
        if dataset_path is None:
            raise Exception('The dataset path was not specified')
        self.dataset_path = dataset_path
        self.origin = origin
        # Create dictionary
        self.embedding_dimension = emb_dim
        self.dictionary = None
        print("... Building dictionary")
        self.build_dict(n_words)
        print("... Done building dictionary")

    def load_data(self, maxlen=None, sort_by_len=True):
        train = self.grab_data(('%s/ptb.train.txt' % self.dataset_path))
        valid = self.grab_data(('%s/ptb.valid.txt' % self.dataset_path))
        test = self.grab_data(('%s/ptb.test.txt' % self.dataset_path))

        if maxlen is not None:
            new_train = []
            for x in train:
                if len(x) < maxlen:
                    new_train.append(x)
            train = new_train

        if sort_by_len:
            sorted_index = du.len_argsort(test)
            test = [test[i] for i in sorted_index]
            sorted_index = du.len_argsort(valid)
            test = [valid[i] for i in sorted_index]
            sorted_index = du.len_argsort(test)
            test = [test[i] for i in sorted_index]

        return train, valid, test

    def grab_data(self, input_file):
        """
        Returns a list of sequences (integerized) corresponding
        to the sentences in a dataset
        """
        sentences = []
        with open(input_file, 'r') as tt:
            for line in tt:
                sentences.append(line.strip())
        seqs = [None] * len(sentences)
        for idx, ss in enumerate(sentences):
            seqs[idx] = self.dictionary.read_sentence(ss)
        return seqs

    def build_dict(self, n_words):
        sentences = []
        train_text = ('%s/ptb.train.txt' % self.dataset_path)
        with open(train_text, 'r') as tt:
            for line in tt:
                sentences.append(line.strip())
        self.dictionary = Dict(sentences, n_words, self.embedding_dimension)

    def get_dataset_file(self):
        """Download file if it does not exist"""
        raise NotImplementedError
