#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 23:32:18 2022

@author: afasjasd
"""

#%%
from Functions import *
import numpy as np

#%%

phi = np.linspace(0, 4*np.pi, 101)
mode_array = np.array([np.cos(phi/2.*i) for i in range(1, 11)])
fourier_amp_ori = np.array([10., 9., 8., 7., 6., 5., 4., 3., 2., 1.])
fourier_trial = mode_array.T@fourier_amp_ori
plt.plot(phi, fourier_trial)

#%%
t, l = interpolate(fourier_trial, phi)
fourier_amp = fourier_decompose(l, t)
print(fourier_amp)
plt.plot(phi, fourier_trial)
plt.plot(t, l)

#%%
def reconstruct(amp, phi):
    mode_array = np.array([np.cos(phi/2.*i) for i in range(1, 11)])
    return mode_array.T@amp

plt.plot(phi, fourier_trial)
plt.plot(phi, reconstruct(fourier_amp, phi))