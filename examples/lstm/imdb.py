from __future__ import print_function
import os
import glob

from cutils.data_interface.interface import DataInterface
import cutils.data_interface.utils as du
from cutils.dict import Dict


class IMDB(DataInterface):
    def __init__(self, dataset_path, origin=None, n_words=100000, emb_dim=100):
        self.dataset_path = dataset_path
        self.origin = origin
        # Download the dataset if it does not exist
        if os.path.isdir(os.getcwd() + "/" + dataset_path):
            self.dataset_path = os.getcwd() + "/" + dataset_path
        if not os.path.isdir(self.dataset_path):
            if origin is not None:
                du.download_dataset(self.dataset_path, origin)
            else:
                raise Exception('Dataset not found and the origin \
                    was not specified')
        # Create dictionary
        self.embedding_dimension = emb_dim
        self.dictionary = None
        print("... Building dictionary")
        self.build_dict(n_words)
        print("... Done building dictionary")
        self.train = None
        self.valid = None
        self.test = None

    def load_data(self, valid_portion=0.1,
                  maxlen=None, sort_by_len=True):
        train_x_pos = self.grab_data(self.dataset_path + "/train/pos")
        train_x_neg = self.grab_data(self.dataset_path + "/train/neg")
        train_x = train_x_pos + train_x_neg
        train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)
        train_set = (train_x, train_y)

        test_x_pos = self.grab_data(self.dataset_path + "/test/pos")
        test_x_neg = self.grab_data(self.dataset_path + "/test/neg")
        test_x = test_x_pos + test_x_neg
        test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)
        test_set = (test_x, test_y)

        if maxlen:
            new_train_set_x = []
            new_train_set_y = []
            for x, y in zip(train_set[0], train_set[1]):
                if len(x) < maxlen:
                    new_train_set_x.append(x)
                    new_train_set_y.append(y)
            train_set = (new_train_set_x, new_train_set_y)

        valid_set = ([], [])
        if valid_portion > 0.:
            train_set, valid_set = du.create_subset(train_set, valid_portion)

        train_set_x, train_set_y = train_set
        valid_set_x, valid_set_y = valid_set
        test_set_x, test_set_y = test_set
        if sort_by_len:
            sorted_index = du.len_argsort(test_set_x)
            test_set_x = [test_set_x[i] for i in sorted_index]
            test_set_y = [test_set_y[i] for i in sorted_index]

            sorted_index = du.len_argsort(valid_set_x)
            valid_set_x = [valid_set_x[i] for i in sorted_index]
            valid_set_y = [valid_set_y[i] for i in sorted_index]

            sorted_index = du.len_argsort(train_set_x)
            train_set_x = [train_set[0][i] for i in sorted_index]
            train_set_y = [train_set[1][i] for i in sorted_index]

        train = (train_set_x, train_set_y)
        valid = (valid_set_x, valid_set_y)
        test = (test_set_x, test_set_y)

        self.train = train
        self.valid = valid
        self.test = test

        return train, valid, test

    def grab_data(self, dirpath):
        sentences = []
        currdir = os.getcwd()
        os.chdir(dirpath)
        for ff in glob.glob("*.txt"):
            with open(ff, 'r') as f:
                sentences.append(f.readline().strip())
        os.chdir(currdir)
        sentences = du.tokenize(sentences)
        sentences = du.lowercase(sentences)
        seqs = [None] * len(sentences)
        for idx, ss in enumerate(sentences):
                seqs[idx] = self.dictionary.read_sentence(ss)
        return seqs

    def build_dict(self, n_words):
        sentences = []
        currdir = os.getcwd()
        os.chdir('%s/train/pos' % self.dataset_path)
        for ff in glob.glob("*.txt"):
            with open(ff, 'r') as f:
                sentences.append(f.readline().strip())
        os.chdir('%s/train/neg' % self.dataset_path)
        for ff in glob.glob("*.txt"):
            with open(ff, 'r') as f:
                sentences.append(f.readline().strip())
        os.chdir(currdir)
        sentences = du.tokenize(sentences)
        sentences = du.lowercase(sentences)
        self.dictionary = Dict(sentences, n_words, self.embedding_dimension)

    def get_dataset_file(self):
        """Download file if it does not exist"""
        raise NotImplementedError
