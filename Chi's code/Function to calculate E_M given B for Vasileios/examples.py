#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 15:02:18 2021

@author: afasjasd
"""

import majorana as mj
import numpy as np
from matplotlib import pyplot as plt

#%%
print(mj.E_M(0.)) # B=0.

#%%
print(mj.E_M(2e-3, W=10, delta=1e-3)) # tuning microscopic parameters

#%%
B_array = np.linspace(0, 5)
EM_array = []
for B in B_array:
    print(B)
    EM = mj.E_M(B)
    EM_array.append(EM)
    print(EM)
print(EM_array)
#%%
plt.plot(B_array, EM_array0)
plt.xlabel(r'$B$')
plt.ylabel(r'$E/\Delta$')
plt.title(r'$W=0, t=1.5, nu=2, \mu=\mu_{leads}=0, \Delta=1$')

#%%
EM_array0 = []
for B in B_array:
    print(B)
    EM=mj.E_M(B, W=0)
    EM_array0.append(EM)
    print(EM)
print(EM_array0)