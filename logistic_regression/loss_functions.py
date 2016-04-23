"""
A collection of loss functions for use with Theano
"""

import theano.tensor as T


def zero_one_loss(y_pred, y):
    """
    Implements the Zero-one loss (non-differentiable)
    Returns the average zero-one loss over a batch

    Parameters:
        y_pred : The predicted output
        y : The indices of the true labels
            Expected format : Vector
    Returns : The zero one loss for the dataset with true label y
    """
    return T.mean(T.neq(y_pred, y))


def negative_log_likelihood(p_y_given_x, y):
    """
    Implements the Negative log likelihood loss function
    Differentiable surrogate for the 0-1 loss
    Works for a batch and returns the mean of the NLL over the batch

    Parameters:
        :type p_y_given_x: theano.tensor.TensorType
        :param p_y_given_x: :math p(y|x, \theta)

        :type y: theano.tensor.TensorType
        :param y: A vector of the indices of the true labels
    """
    # The arange, y combination will return M[0,y_1], ... , M[n, y_n]
    # where M is the matrix being indexed
    return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
