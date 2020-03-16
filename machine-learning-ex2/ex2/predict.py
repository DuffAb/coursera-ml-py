import numpy as np
from sigmoid import *


def predict(theta, X):
    m = X.shape[0]

    # Return the following variable correctly
    p = np.zeros(m)

    # ===================== Your Code Here =====================
    # Instructions : Complete the following code to make predictions using
    #                your learned logistic regression parameters.
    #                You should set p to a 1D-array of 0's and 1's
    #
    # 预测出集合 X 每一项数据的靠近 1 的百分比
    p = sigmoid(np.dot(X, theta.T))
    # 大于等于 0.5为正
    pos = np.where(p >= 0.5)
    neg = np.where(p < 0.5)

    p[pos] = 1
    p[neg] = 0
    # ===========================================================

    return p
