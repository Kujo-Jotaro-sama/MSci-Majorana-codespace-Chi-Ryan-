#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 17:27:08 2022

@author: afasjasd
"""
#%%
import numpy as np
import kwant
import kwant.continuum
import peierls
import matplotlib.pyplot as plt

import adaptive
#adaptive.notebook_extension()
from concurrent.futures import ProcessPoolExecutor
from operator import itemgetter

import sympy
from sympy.physics.matrices import msigma, Matrix
from sympy import eye
from sympy.physics.quantum import TensorProduct

from sympy.utilities.exceptions import SymPyDeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=SymPyDeprecationWarning)


ham_3DTI = ("- mu * kron(sigma_z, sigma_0, sigma_0) + "
            "{epsilon} * kron(sigma_z, sigma_0, sigma_0) + "
            "{M} * kron(sigma_z, sigma_0, sigma_z) - "
            "A_perp * k_x * kron(sigma_z, sigma_y, sigma_x) + "
            "A_perp * k_y * kron(sigma_0, sigma_x, sigma_x) + "
            "A_z * k_z * kron(sigma_z, sigma_0, sigma_y)")

epsilon = "(C_0 - C_perp * (k_x**2 + k_y**2) - C_z * k_z**2)"
M = "(M_0 - M_perp * (k_x**2 + k_y**2) - M_z * k_z**2)"
ham_3DTI = ham_3DTI.format(epsilon=epsilon, M=M)
ham_3DTI = ham_3DTI.format(C_0="C_0")

SC_complex = ("- re({Delta}) * kron(sigma_y, sigma_y, sigma_0) -"
              "im({Delta}) * kron(sigma_x, sigma_y, sigma_0)")

SC_L_final = SC_complex.format(Delta="Deltaf_L(y, z)")
SC_R_final = SC_complex.format(Delta="Deltaf_R(y, z)")

ham_3DTI_SC_L = ham_3DTI + SC_L_final # 3D TI BdG Hamiltonian with superconducting pairing potential for left lead
ham_3DTI_SC_R = ham_3DTI + SC_R_final # 3D TI BdG Hamiltonian with superconducting pairing potential for right lead

ham_3DTI_discr, coords = kwant.continuum.discretize_symbolic(ham_3DTI)
ham_3DTI_SC_L_discr, coords = kwant.continuum.discretize_symbolic(ham_3DTI_SC_L)
ham_3DTI_SC_R_discr, coords = kwant.continuum.discretize_symbolic(ham_3DTI_SC_R)

vector_potential='[B_y * (z - T_z), -B_x * (z - T_z), 0]'
signs=[-1, -1, -1, -1, 1, 1, 1, 1] # negative charge for the particles, positive charge for the holes

ham_3DTI_discr = peierls.apply(ham_3DTI_discr, coords, A=vector_potential, signs=signs)
ham_3DTI_SC_L_discr = peierls.apply(ham_3DTI_SC_L_discr, coords, A=vector_potential, signs=signs)
ham_3DTI_SC_R_discr = peierls.apply(ham_3DTI_SC_R_discr, coords, A=vector_potential, signs=signs)

a = 10
temp_syst = kwant.continuum.build_discretized(ham_3DTI_discr, coords, grid=a)
temp_lead_L = kwant.continuum.build_discretized(ham_3DTI_SC_L_discr, coords, grid=a)
temp_lead_R = kwant.continuum.build_discretized(ham_3DTI_SC_R_discr, coords, grid=a)

#%%
# shape of system
def get_shape(L, W, T):
    L_start, W_start, T_start = 0, 0, 0
    L_stop, W_stop, T_stop = L, W, T

    def shape(site):
        (x, y, z) = site.pos
        return (W_start <= y <= W_stop and
                T_start <= z <= T_stop and
                L_start <= x <= L_stop)

    return shape, np.array([L_stop, W_stop, T_stop])

#%%
L_x = 100
W_y = 100
T_z = 100

syst = kwant.Builder()
_ = syst.fill(temp_syst, *get_shape(L_x, W_y, T_z))
lat = kwant.lattice.cubic(a, norbs=8)

#%%
# sigma_0 = np.identity(2)
# sigma_z = np.array([[1, 0], [0, -1]])
# conservation_law = -np.kron(sigma_z, np.kron(sigma_0, sigma_0))

lead_L = kwant.Builder(kwant.TranslationalSymmetry((-a, 0, 0)))
lead_R = kwant.Builder(kwant.TranslationalSymmetry((a, 0, 0)))

lead_L.fill(temp_lead_L, *get_shape(L_x, W_y, T_z))
lead_R.fill(temp_lead_R, *get_shape(L_x, W_y, T_z))

syst.attach_lead(lead_L)
syst.attach_lead(lead_R)

#%%
#kwant.plotter.set_engine("matplotlib")
# kwant.plotter.set_engine("plotly")
fig = kwant.plot(syst, show=False)
# fig.update_layout(scene_aspectmode='data')
# fig.show()
fig

#%%
fsyst = syst.finalized()

#%%
f = dict(
    re=lambda x: x.real,
    im=lambda x: x.imag,
    phi_0=1.0, # units with flux quantum equal to 1
    exp=np.exp
)

params_3DTI = dict(
    A_perp=3.4,
    A_z=0.84,#3 toy model, 0.5 realistic
    M_0=0.22,
    M_perp=48.51,
    M_z=19.64,
    C_0=0.001,
    C_perp=10.78,
    C_z=12.39
)

params_JJ = dict(
    a = 30,
    L_x=L_x,
    W_y=W_y,
    T_z=T_z,
    mu = 0,
    B_x = 0,
    B_y = 0,
    Deltaf_L = lambda y, z: 0,
    Deltaf_R = lambda y, z: 0,
)

#%%
lead_ind = 0

mu = 0.18-0.26#-np.sqrt(2.*10.78**2.+12.39**2.)
flux = 0.5
Delta = 0.001 #Nb ~2.3meV ~1meV
phase_diff = np.pi

params_bands = dict(
    **f,
    **params_3DTI,
    **params_JJ
)
params_bands.update(
    mu = mu,
    B_x = flux/W_y/T_z,
    Deltaf_L = lambda y, z: Delta if z==T_z else 0,
    Deltaf_R = lambda y, z: Delta*np.exp(1j*phase_diff) if z==T_z else 0
)
bands = kwant.physics.Bands(fsyst.leads[lead_ind], params=params_bands)

#kbounds = (-np.pi, np.pi)
kbounds = (-.15,.15)

def bands_wrapper(k):
    return bands(k)

#%%
f = dict(
    re=lambda x: x.real,
    im=lambda x: x.imag,
    phi_0=1.0, # units with flux quantum equal to 1
    exp=np.exp
)

params_3DTI = dict(
    A_perp=3.,
    A_z=0.5,#3 toy model, 0.5 realistic
    M_0=0.3,
    M_perp=15,
    M_z=15,
    C_0=0,
    C_perp=0,
    C_z=0
)

params_JJ = dict(
    a = 10,
    L_x=L_x,
    W_y=W_y,
    T_z=T_z,
    mu = 0,
    B_x = 0,
    B_y = 0,
    Deltaf_L = lambda y, z: 0,
    Deltaf_R = lambda y, z: 0,
)

#%%
lead_ind = 0

mu = 0.#-np.sqrt(2.*10.78**2.+12.39**2.)
flux = 2.
Delta = 0.001 #Nb ~2.3meV ~1meV
phase_diff = np.pi

params_bands = dict(
    **f,
    **params_3DTI,
    **params_JJ
)
params_bands.update(
    mu = mu,
    B_x = flux/W_y/T_z,
    Deltaf_L = lambda y, z: Delta if z==T_z else 0,
    Deltaf_R = lambda y, z: Delta*np.exp(1j*phase_diff) if z==T_z else 0
)
bands = kwant.physics.Bands(fsyst.leads[lead_ind], params=params_bands)

#kbounds = (-np.pi, np.pi)
kbounds = (-.15,.15)

def bands_wrapper(k):
    return bands(k)
#%%
import matplotlib.pyplot as plt
momenta=np.linspace(-0.15,0.15,101)
energies=[bands(k) for k in momenta]

x=np.array(energies)
en=(x*1000.)
plt.ylim(-50,50)
print(en)


plt.plot(momenta, en)

plt.ylabel('Energy (meV)', fontsize=12)
plt.xlabel('k', fontsize=12)
plt.title(r'$A_z=0.5, \Phi/\Phi_0=2$')
plt.show()
plt.savefig('./3dti_realistic/A_z=0p5_2fluxquanta', dpi=300, bbox_inches='tight')

#%%
for i in np.arange(0, 4.1, 0.1):
    #mu = 0.02
    mu = 0.
    flux = i
    Delta = 0. #Nb ~2.3meV ~1meV
    phase_diff = np.pi
    
    params_bands = dict(
        **f,
        **params_3DTI,
        **params_JJ
    )
    params_bands.update(
        mu = mu,
        B_x = flux/W_y/T_z,
        Deltaf_L = lambda y, z: Delta if z==T_z else 0,
        Deltaf_R = lambda y, z: Delta*np.exp(1j*phase_diff) if z==T_z else 0
    )
    bands = kwant.physics.Bands(fsyst.leads[lead_ind], params=params_bands)
    
    #kbounds = (-np.pi, np.pi)
    kbounds = (-.15,.15)
    
    def bands_wrapper(k):
        return bands(k)
    
    momenta=np.linspace(-0.15,0.15,101)
    energies=[bands(k) for k in momenta]
    
    x=np.array(energies)
    en=(x*1000.)
    fig = plt.figure()
    plt.ylim(-50,50)
    
    plt.plot(momenta, en)
    
    plt.ylabel('Energy (meV)', fontsize=12)
    plt.xlabel('k', fontsize=12)
    plt.title(r'$A_z=0.5, \Delta=0, \mu=0, \Phi/\Phi_0=%.1f$' % (i))
    #plt.show()
    plt.savefig('./3dti_realistic/spectrum_Phi_ani_mu=0_delta=0/Phi=%s' % (str(int(i*10))), dpi=300, bbox_inches='tight')
    
#%%
#band_gap

f = dict(
    re=lambda x: x.real,
    im=lambda x: x.imag,
    phi_0=1.0, # units with flux quantum equal to 1
    exp=np.exp
)

params_3DTI = dict(
    A_perp=3.,
    A_z=3.,#3 toy model, 0.5 realistic
    M_0=0.3,
    M_perp=15,
    M_z=15,
    C_0=0,
    C_perp=0,
    C_z=0
)

params_JJ = dict(
    a = 10,
    L_x=L_x,
    W_y=W_y,
    T_z=T_z,
    mu = 0,
    B_x = 0,
    B_y = 0,
    Deltaf_L = lambda y, z: 0,
    Deltaf_R = lambda y, z: 0,
)

zero_k_gap_array = []
for i in np.arange(0, 4.1, 0.01):
    lead_ind = 0

    mu = 0.#-np.sqrt(2.*10.78**2.+12.39**2.)
    flux = i
    Delta = 0.001 #Nb ~2.3meV ~1meV
    phase_diff = 0.
    
    params_bands = dict(
        **f,
        **params_3DTI,
        **params_JJ
    )
    params_bands.update(
        mu = mu,
        B_x = flux/W_y/T_z,
        Deltaf_L = lambda y, z: Delta if z==T_z else 0,
        Deltaf_R = lambda y, z: Delta*np.exp(1j*phase_diff) if z==T_z else 0
    )
    bands = kwant.physics.Bands(fsyst.leads[lead_ind], params=params_bands)
    
    print(np.min(np.abs(bands(0.)))*1000.)
    zero_k_gap_array.append(np.min(np.abs(bands(0.)))*1000.)
    
#%%
plt.plot(np.arange(0, 4.1, 0.01), zero_k_gap_array)
plt.xlabel(r'$\Phi/\Phi_0$')
plt.ylabel(r'$E (meV)$')
plt.title('Original toy params')
plt.ylim(-0.001, 1.)
print(np.min(np.array(zero_k_gap_array)))