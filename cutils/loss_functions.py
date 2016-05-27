"""
A collection of loss functions for use with Theano
"""

import theano
import theano.tensor as T

from cutils.regularization import L1, L2
from cutils.numeric import safe_log


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
    return T.mean(T.neq(y_pred, y), dtype=theano.config.floatX)


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
    return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y],
                   dtype=theano.config.floatX)


def binary_cross_entropy_loss(true_value, p_true_value):
    """
    Implementes the binary cross entropy loss function

    Parameters:
        :type true_value: theano.tensor.TensorType
        :param true_value : The actual value of the label
            This can be a matrix of values, one row for one example

        :type p_true_value: theano.tensor.TensorType
        :param p_true_value : The probability of the true label according to
            the model. One row for one example
    """
    return -T.sum(true_value * T.log(p_true_value)
                  + (1 - true_value) * T.log(1 - p_true_value), axis=1)


def nce_binary_conditional_likelihood(p_unnormalized, y, y_flat,
                                      noise_samples, noise_dist, k):
    """
    \sum_{w \in batch} [
        \frac{u(w)}{u(w) + k * q(w)}
        + \sum{\hat{w} \from q} [
            \frac{k * q(\hat{w})}{u(\hat{w}) + k * q(\hat{w})}
        ]
    ]
    q() is the noise distribution
    u() is the unnormalized probability
    \hat{w} ~ q()
    k is the number of noise samples
    """
    p_unnormalized_exp = T.exp(p_unnormalized)
    p_unnormalized_flat = p_unnormalized_exp.flatten()
    unnorm_y = p_unnormalized_flat[y_flat]
    noise_y = noise_dist[y]
    p_class1 = safe_log(unnorm_y / (unnorm_y + k * noise_y))

    # Shape is bs x V
    E_p_class_0 = safe_log((k * noise_dist)
                           / (p_unnormalized_exp + k * noise_dist))
    E_p_class_0 = T.sum(E_p_class_0 * noise_samples, axis=1)
    return -T.mean(p_class1 + E_p_class_0, dtype=theano.config.floatX)
