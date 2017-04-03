#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Rachid and Kimia
"""


from gammatone_utils import *
from scikits.talkbox import segment_axis
from scikits.audiolab import Sndfile, play
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def matching_pursuit(signal, dict_kernels, threshold=0.1, max_iter=2000):
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
    # Iterative decomposition
    for i in range(max_iter):
        inner_prod = res.dot(dict_kernels.T)
        max_kernel = np.argmax(inner_prod)
        coeff[max_kernel] = inner_prod[max_kernel] / np.linalg.norm(dict_kernels[max_kernel,: ])**2
        res = res - coeff[max_kernel] * dict_kernels[max_kernel,: ]
        if np.linalg.norm(res) < threshold:
            return coeff
    return coeff


# Parametrization
b = 1.019
resolution = 160
step = 8
n_channels = 128
overlap = 50

# Compute gammatone-based dictionary
D_multi = np.r_[tuple(gammatone_matrix(b, fc, resolution, step)[0] for
                      fc in erb_space(150, 8000, n_channels))]
freq_c = np.array([gammatone_matrix(b, fc, resolution, step)[1] for
                      fc in erb_space(150, 8000, n_channels)]).flatten()
centers = np.array([gammatone_matrix(b, fc, resolution, step)[2] + i*resolution  for
                      i, fc in enumerate(erb_space(150, 8000, n_channels))]).flatten()

# Load test signal
filename = 'data/fsew/fsew0_001.wav'
f = Sndfile(filename, 'r')
nf = f.nframes
fs = f.samplerate
length_sound = 20000
y = f.read_frames(5000)
y = f.read_frames(length_sound)
Y = segment_axis(y, resolution, overlap=overlap, end='pad')
Y = np.hanning(resolution) * Y

# Encoding with matching pursuit
X = np.zeros((Y.shape[0],D_multi.shape[0]))
for idx in range(Y.shape[0]):
    X[idx, :] = matching_pursuit(Y[idx, :], D_multi)

# Reconstruction of the signal
out = np.zeros(int((np.ceil(len(y)/resolution)+1)*resolution))
for k in range(0, len(X)):
    idx = range(k*(resolution-overlap), k*(resolution-overlap) + resolution)
    out[idx] += np.dot(X[k], D_multi)
squared_error = np.sum((y - out[0:len(y)]) ** 2)

# Play the original signal and the reconstructed for comparison
play(y, fs=16000)
play(out, fs=16000)

# Plotting results

# 1st plot: original signal/reconstructed signal/residuals
arr = np.array(range(length_sound))/float(fs)
plt.figure(1)
plt.subplot(311)
plt.plot(arr, y, 'b', label="Input Signal")
plt.legend()
plt.subplot(312)
plt.plot(arr, out[0:len(y)], 'r', label="Recontruction")
plt.legend()
plt.subplot(313)
plt.plot(arr, (y - out[0:len(y)])**2, 'g', label="Residual")
plt.legend()
plt.xlabel("Time in s")
plt.show()

# 2nd plot: spike train
plt.figure(2)
spikes_pos = np.array(np.nonzero(X))
temporal_position = centers[spikes_pos[0][:]]
centre_freq = freq_c[spikes_pos[1][:]]
plt.scatter(temporal_position, centre_freq, marker='+', s=1)
plt.show()

# 3rd plot: example of gammatone-based dictionary
fig = plt.figure(3)
fig.suptitle("Gammatone filters", fontsize="x-large")
freqs = [1000, 300, 40]
resolution = 5000
for center in [100, 1500, 3000]:
    plt.subplot(311)
    plt.plot(gammatone_function(resolution, freqs[0], center), linewidth=1.5)
    plt.subplot(312)
    plt.plot(gammatone_function(resolution, freqs[1], center+300), linewidth=1.5)
    plt.ylabel("Kernel values")
    plt.subplot(313)
    plt.plot(gammatone_function(resolution, freqs[2], center+1000), linewidth=1.5)
    plt.xlabel("Time (s)")
plt.show()
