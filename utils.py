#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:08:29 2017

@authors: Rachid Riad and Kimia Nadjahi
"""

from scikits.audiolab import Sndfile, play
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from sklearn.decomposition import SparseCoder

def gammatone_function(resolution, fc, center, fs=16000, l=4,
                       b=1.019):
    t = np.linspace(0, resolution-(center+1), resolution-center)/fs
    g = np.zeros((resolution,))
    g[center:] = t**(l-1) * np.exp(-2*np.pi*b*erb(fc)*t)*np.cos(2*np.pi*fc*t)
    return g

def gammatone_matrix(b, fc, resolution, step, fs=16000, l=4, threshold=5):
    """Dictionary of gammatone functions"""
    centers = np.arange(0, resolution - step, step)
    D = []
    for i, center in enumerate(centers):
        t = np.linspace(0, resolution-(center+1), resolution-center)/fs
        env = t**(l-1) * np.exp(-2*np.pi*b*erb(fc)*t)
        if env[-1]/max(env) < threshold:
            D.append(gammatone_function(resolution, fc, center, b=b, l=l))
    D = np.asarray(D)
    D /= np.sqrt(np.sum(D ** 2, axis=1))[:, np.newaxis]
    return D

def erb(f):
    return 24.7+0.108*f

def erb_space(low_freq, high_freq, num_channels, EarQ = 9.26449, minBW = 24.7, order = 1):
    return -(EarQ*minBW) + np.exp(np.arange(1,num_channels+1)*(-np.log(high_freq + EarQ*minBW) + np.log(low_freq + EarQ*minBW))/num_channels) * (high_freq + EarQ*minBW)





if __name__ == '__main__':

    filename = 'data/fsew/fsew0_001.wav'
    f = Sndfile(filename, 'r')
    nf = f.nframes
    fs = f.samplerate
    data = f.read_frames(5000)
    data = f.read_frames(5000)
    x = np.array(range(5000))/float(nf)
    
    
    plt.figure(1)
    plt.title('Signal Wave...')
    plt.xlabel('Time in s')
    plt.ylabel('Amplitude')
    plt.plot(x,data)
    plt.show()
    
    D = {}
    # Parameters for the spike/kernel dictionnary
    n = 4
    f_max = 5000
    f_min = 100
    phi_min = 0.0
    phi_max = np.max(x)
    b_max = 10
    b_min = 1
    
    idx = 0
    for freq in range(f_min,f_max+100,1000):
        for phi_idx in range(0,101,5):
            for b in range(b_min,b_max,1):
                b = b/10.0
                phi = (phi_max)*phi_idx/100.0
                D[idx] = [n,b,freq,phi]
                idx += 1
    
    num_spikes = len(D)
    M = np.zeros((5000,num_spikes))