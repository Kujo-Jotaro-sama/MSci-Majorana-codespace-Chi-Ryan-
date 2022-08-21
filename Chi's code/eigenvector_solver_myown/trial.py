#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 23:28:25 2022

@author: afasjasd
"""
#%%
import numpy as np
import kwant
from scipy import linalg

import kwant.continuum

hamiltonian_lead = ("(- mu + A*(k_x**2 + k_y**2))*sigma_0 + B*sigma_z - alpha*(k_x*sigma_y - k_y*sigma_x) ")

params = dict(
    mu = 0,
    A = 1,
    B = 0.5,
    alpha = 1.
)

a = 1

ham_lead_discretized_symbolic, ham_lead_coords = kwant.continuum.discretize_symbolic(hamiltonian_lead)
ham_lead_discretized = kwant.continuum.discretize(hamiltonian_lead, grid=a)

lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
    
def lead_shape(site):
    x, y = site.pos
    return (y==0)

lead.fill(ham_lead_discretized, lead_shape, (0, 0))

syst = kwant.Builder()
    
def syst_shape(site):
    x, y = site.pos
    return (x==0) and (y==0)

syst.fill(ham_lead_discretized, syst_shape, (0, 0))

syst.attach_lead(lead)

syst = syst.finalized()

#%%
import matplotlib.pyplot as plt, matplotlib.backends
plot = kwant.plot(syst, show=False)
plot.show()

#%%
bands=kwant.physics.Bands(syst.leads[0], params=params)
momenta=np.linspace(-np.pi,np.pi,101)
energies=[bands(k) for k in momenta]
x=np.array(energies)
en=(x)
plt.plot(momenta, en)
plt.axhline(1.4)

#%%



#%%
import warnings

def modes(infinitesystem, energy=0, params=None):
    import leads_myown
    ham = infinitesystem.cell_hamiltonian(params=params)
    hop = infinitesystem.inter_cell_hopping(params=params)
    symmetries = infinitesystem.discrete_symmetry(params=params)
    # Check whether each symmetry is broken.
    # If a symmetry is broken, it is ignored in the computation.
    broken = set(symmetries.validate(ham) + symmetries.validate(hop))
    attribute_names = {'Conservation law': 'projectors',
                      'Time reversal': 'time_reversal',
                      'Particle-hole': 'particle-hole',
                      'Chiral': 'chiral'}
    for name in broken:
        warnings.warn('Hamiltonian breaks ' + name +
                      ', ignoring the symmetry in the computation.')
        assert name in attribute_names, 'Inconsistent naming of symmetries'
        setattr(symmetries, attribute_names[name], None)

    shape = ham.shape
    assert len(shape) == 2
    assert shape[0] == shape[1]
    # Subtract energy from the diagonal.
    ham.flat[::ham.shape[0] + 1] -= energy

    # Particle-hole and chiral symmetries only apply at zero energy.
    if energy:
        symmetries.particle_hole = symmetries.chiral = None
    return leads_myown.modes(ham, hop, discrete_symmetry=symmetries)
    
#%%
import time
t0 = time.time()
print(syst.leads[0].modes(energy=2.1, params=params)[1].vecs)
t1 = time.time()
print(modes(syst.leads[0], 2.1, params=params)[1].vecs)
t2=time.time()
print(t1-t0)
print(t2-t1)