#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 17:59:29 2021

@author: afasjasd
"""

import boundstate as bs
import kwant
import numpy as np
import scipy as spi


def make_system_mu(W=1, t=1.5, nu=2., mu=0., mu_lead0=0., mu_lead1=0., delta=1., B=2., phi=0.):
    '''

    Parameters
    ----------
    W : integer, optional
        The length of the normal (unproximimisted junction). The default is 1.
    t : float, optional
        Hopping intgeral in the tight binding model. The default is 1.5.
    nu : float, optional
        float chracterising the strength of spin-orbit interaction in the NW. The default is 2..
    mu : float, optional
        Chemical potential in the normal region. The default is 0..
    mu_lead0 : float, optional
        Chemical potential in lead 0. The default is 0..
    mu_lead1 : float, optional
        Chemical potential in lead 1. The default is 0..
    delta : float, optional
        The superconducting gap. The default is 1..
    B : float, optional
        Zeeman energy associated with the external B field. The default is 2..
    phi : float, optional
        The phase difference between the two superconducting islands. The default is 0..

    Returns
    -------
    <kwant.system.FiniteSystem> object
        The 1D tight binding system for superconductor-proximitised 1D rashba III-V nanowires.
        The default parameters are set following Fig. 4 of Keselman et al.
        (doi:10.21468/SciPostPhys.7.4.050). For a realistic system the respective tight-binding
        parameters can be accessed from standard online databases. a=1 needs to be updated
        manually for realistic systems

    '''
    a = 1
    lat = kwant.lattice.chain()
    syst = kwant.Builder()
    
    # Pauli matrices
    tau_x = np.kron(np.array([[0, 1], [1, 0]]), np.eye(2))
    tau_y = np.kron(np.array([[0, -1j], [1j, 0]]), np.eye(2))
    tau_z = np.kron(np.array([[1, 0], [0, -1]]), np.eye(2))
    sigma_x = np.kron(np.eye(2), np.array([[0, 1], [1, 0]]))
    sigma_y = np.kron(np.eye(2), np.array([[0, -1j], [1j, 0]]))
    sigma_z = np.kron(np.eye(2), np.array([[1, 0], [0, -1]]))
    
    #### Define the scattering region. ####
    syst[(lat(0))] = (2.*t-mu_lead0)*tau_z + B*tau_z@sigma_x + delta*tau_y@sigma_y
    syst[(lat(W+1))] = (2.*t-mu_lead1)*tau_z + B*tau_z@sigma_x + delta*tau_y@sigma_y
    
    syst[(lat(x) for x in range(1, W+1))] = (2.*t-mu)*tau_z + B*tau_z@sigma_x
    
    syst[(lat(0), lat(1))] = -spi.linalg.expm(1j*phi*tau_z/2.)@(t*tau_z + 1j*nu*tau_z@sigma_y)
    if W>0:
        syst[((lat(x), lat(x+1)) for x in range(1, W+1))] = -t*tau_z - 1j*nu*tau_z@sigma_y
    
    
    #### Define the leads. ####
    sym_left = kwant.TranslationalSymmetry([-1])
    lead0 = kwant.Builder(sym_left)
    lead0[(lat(-1))] = (2.*t-mu_lead0)*tau_z + B*tau_z@sigma_x + delta*tau_y@sigma_y
    lead0[lat.neighbors()] = -t*tau_z - 1j*nu*tau_z@sigma_y
    sym_right = kwant.TranslationalSymmetry([1])
    lead1 = kwant.Builder(sym_right)
    lead1[(lat(W+2))] =  (2.*t-mu_lead1)*tau_z + B*tau_z@sigma_x + delta*tau_y@sigma_y
    lead1[lat.neighbors()] = -t*tau_z - 1j*nu*tau_z@sigma_y

    #### Attach the leads and return the system. ####
    syst.attach_lead(lead0)
    syst.attach_lead(lead1)

    return syst.finalized()


def E_M(B, mode='quick', W=1, t=1.5, nu=2., mu=0., mu_leads=0., delta=1.):
    '''
    

    Parameters
    ----------
    B : float
        Energy characterising the external B field.
    mode : string, optional
        string that decides in which mode the function is called. The default is 'quick'.
    W : int, optional
        number of sites in the normal region. The default is 1.
    t : float, optional
        hopping integral. The default is 1.5.
    nu : float, optional
        spin orbit coupling. The default is 2..
    mu : float, optional
        chemical potential in the normal region. The default is 0..
    mu_leads : float, optional
        chemical potential in the leads. The default is 0..
    delta : float, optional
        superconducting gap. The default is 1..

    Raises
    ------
    ValueError
        When the value of 'modes' is neither 'quick' nor 'rigorous'.

    Returns
    -------
    float
        E_M, the Majorana energy calculated from the microscopics according to
        eq. 48 of Keselman et al. (doi:10.21468/SciPostPhys.7.4.050)

    '''
    
    if (mode!='quick') and (mode!='rigorous'):
        raise ValueError('the parameter mode can only be "quick" or "rigorous"!')
    else:
        syst=make_system_mu(W=W, t=t, nu=nu, mu=mu, mu_lead0=mu_leads, mu_lead1=mu_leads, delta=delta, B=B, phi=np.pi)
        e, psi = bs.find_boundstates(syst, -1.1*delta, 1.1*delta, rtol=1e-2*delta)
        if np.min(np.abs(e))>1e-3*delta:
            return 0.
        elif mode=='quick':
            syst0 = make_system_mu(W=W, t=t, nu=nu, mu=mu, mu_lead0=mu_leads, mu_lead1=mu_leads, delta=delta, B=B, phi=0.)
            e0, psi0 = bs.find_boundstates(syst0, -1.1*delta, 1.1*delta, rtol=1e-2*delta)
            return np.min(np.abs(e0))
        elif mode=='rigorous':
            energy_arrays=[]
            wavefunc_arrays=[]
            ex_inte=np.linspace(0., 4.*np.pi, 101)
            ey_inte = []
            branch_count=0
            for i in ex_inte:
                syst = make_system_mu(W=W, B=B, t=t, nu=nu, mu=mu, mu_lead0=mu_leads, mu_lead1=mu_leads, delta=delta, phi=i)
                energies, wavefunctions = bs.find_boundstates(syst, -1.1*delta, 1.1*delta, rtol=1e-2*delta)
                energy_arrays.append(energies)
                wavefunc_arrays.append(wavefunctions)
            for ey in energy_arrays:
                if np.min(np.abs(ey))<1e-3*delta:
                    branch_count+=1
                    ey_inte.append(((-1.)**branch_count)*np.min(np.abs(ey)))
                else:
                    ey_inte.append(((-1.)**branch_count)*np.min(np.abs(ey)))
            ey_inte=np.array(ey_inte)
            ey_inte2 = np.cos(ex_inte/2.)*ey_inte
            return spi.integrate.simpson(ey_inte2, ex_inte)/(2.*np.pi)
        
        

#test1 = E_M(0) # expect 0
#print(test1)

#test2 = E_M(2) # expect 0.1319 from previous builds
#print(test2) # correct

#test3 = E_M(2, mode='rigorous')
#print(test3) # not right. missed a bracket for (-1)

#test4 = E_M(2, mode='rigorous') # expect 0.1369 (this is actually worse numerically 
#since with 100 points there's still quite a bit of integration errors)
# but 1000 pts would have even more unrealistic runtime
# would be better to keep to using 'quick'
#print(test4) # got 0.1369

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    