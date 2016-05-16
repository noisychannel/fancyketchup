import abc


class DataInterface(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_dataset_file(self):
        """Download the dataset file if it does not exist"""

    @abc.abstractmethod
    def load_data(self):
        """Implement the processing of the dataset"""

    @abc.abstractmethod
    def build_dict(self):
        """Build and return a dictionary for this dataset"""
