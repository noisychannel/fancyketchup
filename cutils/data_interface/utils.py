import os
import numpy
import theano
from subprocess import Popen, PIPE
from six.moves import urllib


def pad_and_mask(seqs, labels, maxlen=None):
    """create the matrices from the datasets.

    this pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    this swap the axis!
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
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatx)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


def scale_to_unit_interval(ndar, eps=1e-8):
    """ scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tokenize(sentences, lang="en"):
    # download the tokenizer if it does not exist
    cdir = os.path.dirname(__file__)
    tok_origin = "https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl"
    tok_dest = os.path.join(cdir, "../scripts/tokenizer")
    prefix_origin = "https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes"
    prefix_dest = os.path.join(cdir, "../share/nonbreaking_prefixes")
    if not os.path.isfile(tok_dest):
        print('downloading the tokenizer script')
        urllib.request.urlretrieve(tok_origin, tok_dest)
        os.chmod(tok_dest, 0775)
    # check if the non-breaking prefix exists for this lang
    if not os.path.isfile(prefix_dest + '/nonbreaking_prefix.' + lang):
        # attempt to download it
        try:
            os.makedirs(prefix_dest)
        except:
            # directory exists
            pass
        try:
            print('downloading the non-breaking prefix file')
            urllib.request.urlretrieve(prefix_origin + "/nonbreaking_prefix." + lang,
                                       prefix_dest + "/nonbreaking_prefix." + lang)
        except urllib.error.urlerror, e:
            #todo: this exception does not get triggered
            # an empty file is created instead which contains the phrase "not found"
            if e.code == 404:
                print('the prefix file for the lang %s could not be downloaded' % lang)
            else:
                print('an unknown error occured while trying to download the prefix \
                       file for the lang %s' % lang)


    tokenizer_cmd = [tok_dest, '-l', 'en', '-q', '-']
    print('tokenizing...')
    text = '\n'.join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    tok_text, _ = tokenizer.communicate(text)
    toks = tok_text.split('\n')[:-1]
    print('done tokenizing')

    return toks

def lowercase(sentences):
    return [s.lower() for s in sentences]

def remove_unk(x, n_words):
    return [[1 if w >= n_words else w for w in sen] for sen in x]

def len_argsort(seq):
    return sorted(range(len(seq)), key=lambda x: len(seq[x]))

def download_dataset(dataset_path, origin):
    from six.moves import urllib
    print('downloading data from %s' % origin)
    try:
        urllib.request.urlretrieve(origin, dataset_path)
    except:
        raise Exception("could not download the dataset from %s" % origin)

def create_subset(whole, small_portion, shuffle=True):
    whole_x, whole_y = whole
    n_samples = len(whole_x)
    if shuffle:
        # todo: seed required here
        sidx = numpy.random.permutation(n_samples)
    else:
        sidx = numpy.arange(n_samples)
    n_large = int(numpy.round(n_samples * (1 - small_portion)))
    small_x = [whole_x[s] for s in sidx[n_large:]]
    small_y = [whole_y[s] for s in sidx[n_large:]]
    large_x = [whole_x[s] for s in sidx[:n_large]]
    large_y = [whole_y[s] for s in sidx[:n_large]]
    return (large_x, large_y), (small_x, small_y)
