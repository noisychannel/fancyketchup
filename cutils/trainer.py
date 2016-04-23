"""
Various training algorithms
"""

import theano.tensor as T
import scipy.optimize


def sgd(cost, params, learning_rate):
    """
    Implements stochastic gradient descent

    :type cost : Theano symbolic expression
    :param cost : The cost function to be minimized

    :type params : List of theano.tensor.TensorType
    :param params : The parameters to be updated

    :type learning_rate : int
    :param learning_rate : The learning rate for SGD

    Returns : A list of tuples representing param updates
              (param, update)
    """
    updates = []
    for p in params:
        grad_p = T.grad(cost=cost, wrt=p)
        updates.append((p, p - learning_rate * grad_p))
    return updates


def conjugate_gradient_descent(train_fn, train_fn_grad,
                               callback, x0, n_epochs):
    """
    Implements the conjugate gradient solver
    """
    best_params = scipy.optimize.fmin_cg(
        f=train_fn,
        x0=x0,
        fprime=train_fn_grad,
        callback=callback,
        disp=0,
        maxiter=n_epochs
    )
    return best_params
