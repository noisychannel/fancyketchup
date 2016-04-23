"""
Implements various regularization techniques
"""

import theano.tensor as T


def L1(params):
    """
    L1 regularization

    Parameters:
        params: The parameters of the model (Symbolic variables)

    Returns:
        A symbolic expression for the regularizer term (should be
        added to the loss expression)
    """
    return T.sum(abs(params))


def L2(params):
    """
    L2 regularization

    Parameters:
        params: The parameters of the model (Symbolic variables)

    Returns:
        A symbolic expression for the regularizer term (should be
        added to the loss expression)
    """
    return T.sum(params ** 2)
