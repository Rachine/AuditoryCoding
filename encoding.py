import numpy as np
import utils2
import pdb

def matching_pursuit(signal, dict_kernels, threshold=0.01, max_iter=5000):
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
    coeff = np.zeros(dict_kernels.shape[0])
    for i in range(max_iter):
        inner_prod = res.dot(dict_kernels.T)
        max_kernel = np.argmax(inner_prod)
        coeff[max_kernel] = inner_prod[max_kernel] / np.linalg.norm(dict_kernels[max_kernel,: ])**2
        res = res - coeff[max_kernel] * dict_kernels[max_kernel,: ]
        if np.linalg.norm(res) < threshold:
            plt.plot(signal)
            pdb.set_trace()
            return coeff
    return coeff

#if __name__ == '__main__':
from scikits.talkbox import segment_axis
resolution = 5000
step = 6
b = 1.019
n_channels = 8
overlap = 80

# Compute a multiscale dictionary

D_multi = np.r_[tuple(gammatone_matrix(b, fc, resolution, step) for
                      fc in erb_space(150, 8000, n_channels))]


# Load test signal
filename = 'data/fsew/fsew0_001.wav'
f = Sndfile(filename, 'r')
nf = f.nframes
fs = f.samplerate
y = f.read_frames(nf)
y = y / 2.0**15
Y = segment_axis(y, resolution, overlap=overlap, end='pad')
Y = np.hanning(resolution) * Y

l = matching_pursuit(Y[5,:],D_multi)
sum(l != 0)