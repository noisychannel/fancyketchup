import numpy
import theano


def bucket_and_pad(x, y, buckets):
    """
    Assumes x to be in a list of
    emb_size X len_sent arrays

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
            buckets[b][1].append(label)
            buckets[b][2].append(padded_mask)


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
