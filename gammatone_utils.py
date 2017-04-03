#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Rachid and Kimia
"""


import numpy as np


def gammatone_function(resolution, fc, center, fs=16000, n=4, b=1.019):
    """Define a single gammatone function"""
    t = np.linspace(0, resolution-(center+1), resolution-center)/fs
    g = np.zeros((resolution,))
    g[center:] = t**(n-1) * np.exp(-2*np.pi*b*erb(fc)*t)*np.cos(2*np.pi*fc*t)
    return g


def gammatone_matrix(b, fc, resolution, step, fs=16000, n=4, threshold=5):
    """Dictionary of gammatone functions"""
    centers = np.arange(0, resolution - step, step)
    D = []
    for i, center in enumerate(centers):
        t = np.linspace(0, resolution-(center+1), resolution-center)/fs
        env = t**(n-1) * np.exp(-2*np.pi*b*erb(fc)*t)
        if env[-1]/max(env) < threshold:
            D.append(gammatone_function(resolution, fc, center, b=b, n=n))
    D = np.asarray(D)
    D /= np.sqrt(np.sum(D ** 2, axis=1))[:, np.newaxis]
    freq_c = np.array(fc*np.ones(D.shape[0]))
    return D, freq_c, centers


def erb(f):
    """Equivalent rectangular bandwidth formula"""
    return 24.7+0.108*f


def erb_space(low_freq, high_freq, num_channels=1, EarQ=9.26449, minBW=24.7, order=1):
    """Generates sequence of critical (center) frequencies"""
    return -(EarQ*minBW) + np.exp(np.arange(1, num_channels+1)*(-np.log(high_freq + EarQ*minBW) + np.log(low_freq + EarQ*minBW))/num_channels) * (high_freq + EarQ*minBW)