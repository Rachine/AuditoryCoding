import numpy as np


def matching_pursuit(signal, dict_kernels, threshold=0.01, max_iter=10000):
    """
    Matching pursuit algorithm for encoding
    :param signal: input signal
    :param dict_kernels: dictionary of kernels, each column is a kernel
    :param threshold: stop condition
    :param max_iter: maximum number of iterations
    :return: array of scalar weighting factor (one per kernel)
    """
    # Initialization
    res = signal
    coeff = np.zeros(dict_kernels.shape[1])
    for i in range(max_iter):
        inner_prod = dict_kernels.T.dot(res)
        max_kernel = np.argmax(inner_prod)
        coeff[max_kernel] = inner_prod[max_kernel] / np.linalg.norm(dict_kernels[:, max_kernel])**2
        res = res - coeff[max_kernel] * dict_kernels[:, max_kernel]
        if np.linalg.norm(res) < threshold:
            return coeff
    return coeff
