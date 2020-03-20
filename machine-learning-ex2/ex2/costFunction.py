import numpy as np
from sigmoid import *


def cost_function(theta, X, y): # 返回代价值，代价函数对应的斜率（导数）
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #

    hypothesis = sigmoid(X @ theta.T)
    first = np.multiply(-y, np.log(hypothesis))
    second = np.multiply(1 - y, np.log(1-hypothesis))
    cost = np.sum(first - second) / m #代价

    grad = np.dot((hypothesis - y), X) / m    #代价函数对应点的斜率（导数）
    # ===========================================================

    return cost, grad
