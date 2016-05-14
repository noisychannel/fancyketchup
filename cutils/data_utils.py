import abc
import os
import numpy
import theano
from subprocess import Popen, PIPE
from six.moves import urllib


class DataInterface(object):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def bucket_and_pad(x, y, buckets, consolidate=False):
        """
        Assumes x to be in a list of
        emb_size X len_sent arrays

        If consolidate is set to True, this will also convert the
        list of samples in a bucket to a stacked array

        """
        for sample, label in zip(x, y):
            length_sample = sample.shape[1]
            b = length_sample - length_sample % 10 + 10
            # Create mask
            mask = numpy.ones(length_sample)
            # Pad sample and mask to bucket length
            padded_sample = numpy.pad(sample, ((0, 0), (0, b - length_sample)),
                                      'constant', constant_values=(0))
            padded_mask = numpy.pad(mask, ((0, b - length_sample)),
                                    'constant', constant_values=(0))
            if b in buckets:
                buckets[b][0].append(padded_sample)
                # TODO: Check whether a floatX is necessary here
                buckets[b][1].append(numpy.asarray(label))
                buckets[b][2].append(padded_mask)

        if consolidate:
            for b in buckets.keys():
                buckets[b][0] = numpy.stack(buckets[b][0],
                                            axis=(len(buckets[b][0][0].shape)))
                buckets[b][1] = numpy.stack(buckets[b][1],
                                            axis=(len(buckets[b][1][0].shape)))
                buckets[b][2] = numpy.stack(buckets[b][2],
                                            axis=(len(buckets[b][2][0].shape)))

    @staticmethod
    def pad_and_mask(seqs, labels, maxlen=None):
        """Create the matrices from the datasets.

        This pad each sequence to the same lenght: the lenght of the
        longuest sequence or maxlen.

        if maxlen is set, we will cut all sequence to this maximum
        lenght.

        This swap the axis!
        """
        # x: a list of sentences
        lengths = [len(s) for s in seqs]

        if maxlen is not None:
            new_seqs = []
            new_labels = []
            new_lengths = []
            for l, s, y in zip(lengths, seqs, labels):
                if l < maxlen:
                    new_seqs.append(s)
                    new_labels.append(y)
                    new_lengths.append(l)
            lengths = new_lengths
            labels = new_labels
            seqs = new_seqs

            assert len(lengths) > 0

        n_samples = len(seqs)
        maxlen = numpy.max(lengths)

        x = numpy.zeros((maxlen, n_samples)).astype('int64')
        x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
        for idx, s in enumerate(seqs):
            x[:lengths[idx], idx] = s
            x_mask[:lengths[idx], idx] = 1.

        return x, x_mask, labels

    @staticmethod
    def scale_to_unit_interval(ndar, eps=1e-8):
        """ Scales all values in the ndarray ndar to be between 0 and 1 """
        ndar = ndar.copy()
        ndar -= ndar.min()
        ndar *= 1.0 / (ndar.max() + eps)
        return ndar

    @staticmethod
    def mask_input(theano_rng, input, p):
        """This function keeps ``1-p`` entries of the inputs the
        same and zero-out randomly selected subset of size ``p``
        For use with dropout and denoising autoencoders
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this won't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return theano_rng.binomial(size=input.shape, n=1,
                                   p=1 - p,
                                   dtype=theano.config.floatX) * input

    @staticmethod
    def tokenize(sentences, lang="en"):
        # Download the tokenizer if it does not exist
        cdir = os.path.dirname(__file__)
        tok_origin = "https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl"
        tok_dest = os.path.join(cdir, "../scripts/tokenizer")
        prefix_origin = "https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes"
        prefix_dest = os.path.join(cdir, "../share/nonbreaking_prefixes")
        if not os.path.isfile(tok_dest):
            print('Downloading the tokenizer script')
            urllib.request.urlretrieve(tok_origin, tok_dest)
            os.chmod(tok_dest, 0775)
        # Check if the non-breaking prefix exists for this lang
        if not os.path.isfile(prefix_dest + '/nonbreaking_prefix.' + lang):
            # attempt to download it
            try:
                os.makedirs(prefix_dest)
            except:
                # Directory exists
                pass
            try:
                print('Downloading the non-breaking prefix file')
                urllib.request.urlretrieve(prefix_origin + "/nonbreaking_prefix." + lang,
                                           prefix_dest + "/nonbreaking_prefix." + lang)
            except urllib.error.URLError, e:
                #TODO: This exception does not get triggered
                # An empty file is created instead which contains the phrase "Not Found"
                if e.code == 404:
                    print('The prefix file for the lang %s could not be downloaded' % lang)
                else:
                    print('An unknown error occured while trying to download the prefix \
                           file for the lang %s' % lang)


        tokenizer_cmd = [tok_dest, '-l', 'en', '-q', '-']
        print('Tokenizing...')
        text = '\n'.join(sentences)
        tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
        tok_text, _ = tokenizer.communicate(text)
        toks = tok_text.split('\n')[:-1]
        print('Done tokenizing')

        return toks

    @staticmethod
    def lowercase(sentences):
        return [s.lower() for s in sentences]

    @staticmethod
    def remove_unk(x, n_words):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    @staticmethod
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    @staticmethod
    def download_dataset(dataset_path, origin):
        from six.moves import urllib
        print('Downloading data from %s' % origin)
        try:
            urllib.request.urlretrieve(origin, dataset_path)
        except:
            raise Exception("Could not download the dataset from %s" % origin)

    @staticmethod
    def create_subset(whole, small_portion, shuffle=True):
        whole_x, whole_y = whole
        n_samples = len(whole_x)
        if shuffle:
            # TODO: seed required here
            sidx = numpy.random.permutation(n_samples)
        else:
            sidx = numpy.arange(n_samples)
        n_large = int(numpy.round(n_samples * (1 - small_portion)))
        small_x = [whole_x[s] for s in sidx[n_large:]]
        small_y = [whole_y[s] for s in sidx[n_large:]]
        large_x = [whole_x[s] for s in sidx[:n_large]]
        large_y = [whole_y[s] for s in sidx[:n_large]]
        return (large_x, large_y), (small_x, small_y)

    @abc.abstractmethod
    def get_dataset_file(self):
        """Download the dataset file if it does not exist"""

    @abc.abstractmethod
    def load_data(self):
        """Implement the processing of the dataset"""

    @abc.abstractmethod
    def build_dict(self):
        """Build and return a dictionary for this dataset"""
