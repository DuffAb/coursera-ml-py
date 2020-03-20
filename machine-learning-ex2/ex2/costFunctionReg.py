import numpy as np
from sigmoid import *


def cost_function_reg(theta, X, y, lmd):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #

    reg_theta = theta[1:]
    hypothesis = sigmoid(X @ theta.T)
    cost = np.sum(np.multiply(-y, np.log(hypothesis)) - np.multiply(1 - y, np.log(1 - hypothesis))) / m  + np.sum(reg_theta * reg_theta) * lmd / (2 * m) # 代价

    normal_grad = (np.dot((hypothesis - y), X) / m).flatten() #默认按行的方向降维
    grad[0] = normal_grad[0]
    grad[1:] = normal_grad[1:] + reg_theta * (lmd / m)  # 代价函数对应点的斜率（导数）

    # ===========================================================

    return cost, grad
