import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y):
    plt.figure()

    # ===================== Your Code Here =====================
    # Instructions : Plot the positive and negative examples on a
    #                2D plot, using the marker="+" for the positive
    #                examples and marker="o" for the negative examples
    #

    pos = np.where(y == 1)[0]   #返回 y = 1 元素的索引
    neg = np.where(y == 0)[0]   #返回 y = 0 元素的索引

    plt.scatter(X[pos,0], X[pos,1], s=None, c='b', marker="+")  # 画离散点
    plt.scatter(X[neg,0], X[neg,1], s=None, c='y', marker="o")  # 画离散点
