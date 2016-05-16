import theano.tensor as T


def dropout_layer(state_before, use_noise, trng, p=0.5):
    """ Drop p entries, in expectation"""
    proj = T.switch(use_noise,
                    (state_before *
                     mask_input(trng, state_before, p)),
                    state_before * 0.5)
    return proj


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
                               dtype=input.dtype) * input
