#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:08:29 2017

@authors: Rachid Riad and Kimia Nadjahi
"""

from scikits.audiolab import Sndfile, play
import numpy as np

filename = 'data/fsew/fsew0_001.wav'
f = Sndfile(filename, 'r')
data = f.read_frames(f.nframes)
play(data,fs=16000) # Just to test the read data
