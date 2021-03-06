import numpy as np


def compute_cost(X, y, theta):
    # Initialize some useful values
    m = y.size  # 代表训练集中实例的数量
    cost = 0

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta.
    #                You should set the variable "cost" to the correct value.
    # print(X.shape)
    # print(theta.shape)
    # print(y.shape)

    inner = np.power((X @ theta - y), 2)# 正确
    #inner = (X @ theta - y)**2          # 正确
    cost = np.sum(inner) / (2 * len(X))
    # ==========================================================

    return cost
