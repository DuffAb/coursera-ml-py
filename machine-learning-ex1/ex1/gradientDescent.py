import numpy as np
from computeCost import *


def gradient_descent(X, y, theta, alpha, num_iters):
    # gradientDescent Performs gradient descent to learn theta
    #    theta = gradientDescent(X, y, theta, alpha, num_iters)  updates theta by
    #    taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = y.size  # number of training examples
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):
        # ===================== Your Code Here =====================
        # Instructions : Perform a single gradient step on the parameter vector theta
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #       X.shape = (97, 2), y.shape = (97, ), theta.shape = (2, )

        theta = theta - alpha * ((X @ theta - y).T @ X).T / m
        # theta = theta - alpha * (X.T @ (X @ theta - y)) / m

        # ===========================================================
        # Save the cost every iteration
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):
        # ===================== Your Code Here =====================
        # Instructions : Perform a single gradient step on the parameter vector theta
        #


        # ===========================================================
        # Save the cost every iteration
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history