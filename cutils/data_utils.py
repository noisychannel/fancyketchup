import os
import numpy
import theano
from subprocess import Popen, PIPE
from six.moves import urllib


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


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


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
            try:
                os.makedirs(prefix_dest)
            except:
                # Directory exists
                pass
            print('Downloading the non-breaking prefix file')
            urllib.request.urlretrieve(prefix_origin + "/nonbreaking_prefix." + lang,
                                       prefix_dest + "/nonbreaking_prefix." + lang)
        except urllib.error.URLError, e:
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
