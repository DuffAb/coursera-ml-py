import numpy as np


def feature_normalize(X):
    # You need to set these values correctly
    n = X.shape[1]  # the number of features
    X_norm = X
    mu = np.zeros(n)    # 平均值
    sigma = np.zeros(n) # 标准差

    # ===================== Your Code Here =====================
    #     # Instructions : First, for each feature dimension, compute the mean
    #     #                of the feature and subtract it from the dataset,
    #     #                storing the mean value in mu. Next, compute the
    #     #                standard deviation of each feature and divide
    #     #                each feature by its standard deviation, storing
    #     #                the standard deviation in sigma
    #     #
    #     #                Note that X is a 2D array where each column is a
    #     #                feature and each row is an example. You need
    #     #                to perform the normalization separately for
    #     #                each feature.
    #
    # Hint: You might find the 'np.mean' and 'np.std' functions useful.
    #       To get the same result as Octave 'std', use np.std(X, 0, ddof=1)
    #
    mu = np.mean(X, axis=0)          # axis=0，计算每一列的均值
    sigma = np.std(X, axis=0, ddof=1)# axis=0，计算每一列的方差
    X_norm = (X - mu) / sigma

    # ===========================================================

    return X_norm, mu, sigma
