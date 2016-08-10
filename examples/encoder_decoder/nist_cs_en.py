"""
Data interface for the NIST CS-EN dataset
"""

from __future__ import print_function

from cutils.data_interface.interface import DataInterface
import cutils.data_interface.utils as du
from cutils.dict import Dict


class NIST_CS_EN(DataInterface):
    def __init__(self, dataset_path, origin=None,
                 src_n_words=100000, tgt_n_words=100000,
                 src_emb_dim=100, tgt_emb_dim=100):
        if dataset_path is None:
            raise Exception('The dataset path was not specified')
        self.dataset_path = dataset_path
        self.origin = origin
        # Create dictionary
        self.src_embedding_dimension = src_emb_dim
        self.tgt_embedding_dimension = tgt_emb_dim
        self.src_dictionary = None
        self.tgt_dictionary = None
        print("... Building dictionary")
        self.build_dict(src_n_words, tgt_n_words)
        print("... Done building dictionary")

    def load_data(self, maxlen=None, sort_by_len=True):
        train_x = self.grab_data(('%s/dev2.cs' % self.dataset_path))
        valid_x = self.grab_data(('%s/dev1.cs' % self.dataset_path))
        test_x = self.grab_data(('%s/test.cs' % self.dataset_path))
        train_y = self.grab_data(('%s/dev2.en' % self.dataset_path))
        valid_y = self.grab_data(('%s/dev1.en' % self.dataset_path))
        test_y = self.grab_data(('%s/test.en' % self.dataset_path))
        assert len(train_x) == len(train_y)
        assert len(valid_x) == len(valid_y)
        assert len(test_x) == len(test_y)

        if maxlen is not None:
            new_train_x = []
            new_train_y = []
            for x, y in zip(train_x, train_y):
                if len(x) < maxlen:
                    new_train_x.append(x)
                    new_train_y.append(y)
            train_x = new_train_x
            train_y = new_train_y

        if sort_by_len:
            sorted_index = du.len_argsort(train_x)
            train_x = [train_x[i] for i in sorted_index]
            train_y = [train_y[i] for i in sorted_index]
            sorted_index = du.len_argsort(valid_x)
            valid_x = [valid_x[i] for i in sorted_index]
            valid_y = [valid_y[i] for i in sorted_index]
            sorted_index = du.len_argsort(test_x)
            test_x = [test_x[i] for i in sorted_index]
            test_y = [test_y[i] for i in sorted_index]

        train = (train_x, train_y)
        valid = (valid_x, valid_y)
        test = (test_x, test_y)

        return train, valid, test

    def grab_data(self, input_file, dictionary):
        """
        Returns a list of sequences (integerized) corresponding
        to the sentences in a dataset
        """
        sentences = []
        with open(input_file, 'r') as tt:
            for line in tt:
                sentences.append(line.split('|||')[1].strip())
        seqs = [None] * len(sentences)
        for idx, ss in enumerate(sentences):
            seqs[idx] = dictionary.read_sentence(ss)
        return seqs

    def build_dict(self, src_n_words, tgt_n_words):
        source_sentences = []
        train_text = ('%s/dev2.cs' % self.dataset_path)
        with open(train_text, 'r') as tt:
            for line in tt:
                source_sentences.append(line.split('|||')[1].strip())
        self.src_dictionary = Dict(source_sentences, src_n_words, self.src_embedding_dimension)

        target_sentences = []
        train_text = ('%s/dev2.en' % self.dataset_path)
        with open(train_text, 'r') as tt:
            for line in tt:
                target_sentences.append(line.split('|||')[1].strip())
        self.tgt_dictionary = Dict(target_sentences, tgt_n_words, self.tgt_embedding_dimension)

    def get_dataset_file(self):
        """Download file if it does not exist"""
        raise NotImplementedError
