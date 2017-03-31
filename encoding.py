import numpy as np
from utils2 import *
import pdb

def matching_pursuit(signal, dict_kernels, threshold=0.01, max_iter=1000):
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
            print 'exit_threshold'
            plt.plot(signal)
            return coeff
    print 'exit_iter'
    return coeff

#if __name__ == '__main__':
from scikits.talkbox import segment_axis
resolution = 160
step = 6
b = 1.019
n_channels = 64
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
Y = segment_axis(y, resolution, overlap=overlap, end='pad')
Y = np.hanning(resolution) * Y

#l = matching_pursuit(Y[5,:],D_multi)
#sum(l != 0)
X = np.zeros((Y.shape[0],D_multi.shape[0]))
for idx in range(Y.shape[0]):
    X[idx,:] = matching_pursuit(Y[idx,:],D_multi)
    
out= np.zeros(int((np.ceil(len(y)/resolution)+1)*resolution))
for k in range(0, len(X)):
    idx = range(k*(resolution-overlap),k*(resolution-overlap) + resolution)
    out[idx] += np.dot(X[k], D_multi)
squared_error = np.sum((y - out[0:len(y)]) ** 2)
play(y,fs=16000)
play(out,fs=16000)

arr = np.array(range(nf))/float(fs)
plt.figure(1)
plt.subplot(311)
plt.plot(arr,y, 'bo')

plt.subplot(312)
plt.plot(arr,out[0:len(y)], 'r--')

plt.subplot(313)
plt.plot(arr,(y - out[0:len(y)]) ** 2, 'g--')


plt.show()