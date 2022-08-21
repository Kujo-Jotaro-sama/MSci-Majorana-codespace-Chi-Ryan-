# -*- coding: utf-8 -*-
"""
Bunch of functions to help in plotting/ conducting parameter sweeps.
"""
import kwant
import BoundState as bs
import numpy as np
import tinyarray
import matplotlib.pyplot as plt
from types import SimpleNamespace as sns



#%%
p0 = tinyarray.array([[1, 0], [0, 1]])
px = tinyarray.array([[0, 1], [1, 0]])
py = tinyarray.array([[0, -1j], [1j, 0]])
pz = tinyarray.array([[1, 0], [0, -1]])

#tau and sigma are operating on different spaces N_t=4.
tau_0=np.kron(p0,p0)
tau_x=np.kron(p0,px)
tau_y=np.kron(p0,py)
tau_z=np.kron(p0,pz)

sigma_0=np.kron(p0,p0)
sigma_x=np.kron(px,p0)
sigma_y=np.kron(py,p0)
sigma_z=np.kron(pz,p0)

def mat_exp(phi):
    '''
    Returns the matrix exponent of exp(i phi tau_z) required in the Hamiltonian)
    '''
    return np.cos(phi)*tau_0+1j*np.sin(phi)*tau_z

def make_system_1(L,p,phi=0,V_n=1.25):
    '''
    Creates a 1D chain josephson junction attached to 2 superconducting leads
    Picture 1, so phase is added to the H terms of N and right S
    
    Inputs
    L - length of N region
    p - SimpleNamespace object containing all the variables
    phi - phase
    V_n - Applied voltage
    
    Outputs
    JJ - Builder() instance in Kwant, a JJ with L normal lattice sites
    '''
    #Picture 1: throw phase into H of right lead
    #N region has delta=0, so unaffected.
    a=p.a
    B=p.B
    mu=p.mu
    alpha=p.alpha
    gap=p.gap
    t=p.t
    
    H_sc_L1=tau_z*(mu-2*t)+sigma_z*B+tau_x*gap #left lead, phase 0
    V1=tau_z*t+1j*tau_z@sigma_x*alpha
    H_sc_R1=tau_z*(mu-2*t)+sigma_z*B+mat_exp(phi)@tau_x*gap
    H_N1=tau_z*(mu-2*t)+sigma_z*B+V_n*tau_z#Hamiltonian for normal region
    
    JJ = kwant.Builder() #creates an empty object, which we need to fill.
    lat = kwant.lattice.chain(a)#lat defines the lattice that we want, 

    JJ[(lat(x) for x in range(L))] = H_N1
    JJ[kwant.HoppingKind((1,), lat)] = V1

    left_lead=kwant.Builder(kwant.TranslationalSymmetry([-a]))
    left_lead[(lat(0))] = H_sc_L1
    left_lead[kwant.HoppingKind((1,), lat)] = V1
    
    right_lead=kwant.Builder(kwant.TranslationalSymmetry([a]))
    right_lead[(lat(0))] = H_sc_R1 #with added phase
    right_lead[kwant.HoppingKind((1,), lat)] = V1

    JJ.attach_lead(left_lead) 
    JJ.attach_lead(right_lead)
    JJ=JJ.finalized()
    
    return JJ

def make_system_2(L,p,phi=0,V_n=1.25):
    '''
    This code is now obsolete with make_system_3.
    This was wrong. Phase is only added to ONE hopping term between SN.
    The rest of the hopping terms do not change.
    
    Future work: can repurpose this to manually attach lead by defining the
    hopping matrices for each attachment site
    '''
    
    a=p.a
    B=p.B
    mu=p.mu
    alpha=p.alpha
    gap=p.gap
    t=p.t
    
    #Picture 2
    H_sc2=tau_z*(mu-2*t)+sigma_z*B+tau_x*gap 
    V_N2=tau_z*t+1j*tau_z@sigma_x*alpha
    V_SN2=mat_exp(phi)@V_N2
    H_N2=(tau_z*(mu-2*t)+sigma_z*B+V_n*tau_z)
    
    JJ = kwant.Builder() #middle region
    lat = kwant.lattice.chain(a)
    
    JJ[(lat(x) for x in range(L))] = H_N2
    JJ[kwant.HoppingKind((1,), lat)] = V_N2
    
    left_lead=kwant.Builder(kwant.TranslationalSymmetry([-a]))
    left_lead[(lat(0))] = H_sc2
    #left_lead[kwant.HoppingKind((1,), lat)] = V_SN2
    left_lead[lat.neighbors()] = V_SN2
    
    right_lead=kwant.Builder(kwant.TranslationalSymmetry([a]))
    right_lead[(lat(0))] = H_sc2 #with added phase
    #right_lead[kwant.HoppingKind((1,), lat)] = V_N2
    right_lead[lat.neighbors()]= V_N2
    
    #reversed_lead=lead.reversed() #right lead, lattice vector a
    JJ.attach_lead(left_lead) 
    JJ.attach_lead(right_lead)
    JJ=JJ.finalized()
    return JJ

def make_system_3(L,p,phi=0,V_n=1.25):
    '''
    Picture 2: phase added to leftside S-N interface hopping term
    via addition of 1 extra N site.
    '''
    
    a=p.a
    B=p.B
    mu=p.mu
    alpha=p.alpha
    gap=p.gap
    t=p.t
    
    H_sc3=tau_z*(mu-2*t)+sigma_z*B+tau_x*gap 
    V_N3=tau_z*t+1j*tau_z@sigma_x*alpha
    V_SN3=mat_exp(phi)@V_N3
    H_N3=(tau_z*(mu-2*t)+sigma_z*B+V_n*tau_z)
    
    JJ = kwant.Builder() #middle region
    lat = kwant.lattice.chain(a)
    
    JJ[(lat(x) for x in range(1,L+1))] = H_N3
    JJ[(lat(0))]=H_sc3
    #JJ[kwant.HoppingKind((1,), lat)] = V_N3
    JJ[lat.neighbors()] = V_N3
    del JJ[(lat(0),lat(1))] #redundant, but just in case it doesnt override
    JJ[(lat(0),lat(1))]=V_SN3
    
    left_lead=kwant.Builder(kwant.TranslationalSymmetry([-a]))
    left_lead[(lat(0))] = H_sc3
    #left_lead[kwant.HoppingKind((1,), lat)] = V_N3
    left_lead[lat.neighbors()] = V_N3
    
    right_lead=kwant.Builder(kwant.TranslationalSymmetry([a]))
    right_lead[(lat(0))] = H_sc3
    #right_lead[kwant.HoppingKind((1,), lat)] = V_N3
    right_lead[lat.neighbors()]= V_N3
    
    #reversed_lead=lead.reversed() #right lead, lattice vector a
    JJ.attach_lead(left_lead) 
    JJ.attach_lead(right_lead)
    JJ=JJ.finalized()
    return JJ
    

def plot_dispersion(JJ,krange=np.pi,save=False,xlim=2,ylim=4,lead=1):
    '''
    Plots the bands of the superconducting leads. Assumes 2 leads, will plot
    the rightmost one. lead=0 for left lead
    '''
    bands=kwant.physics.dispersion.Bands(JJ.leads[lead])
    momenta = np.linspace(-np.pi, np.pi, 500)

    fig=plt.figure()
    energies = [bands(k) for k in momenta]
    plt.plot(momenta, energies)
    plt.xlabel(r'$k$')
    plt.ylabel(r'$E_m(k)$')
    plt.xlim(-xlim,xlim)
    plt.ylim(-ylim,ylim)
    plt.tight_layout()
    plt.show()
    if save==True:
        savename=input('please enter a filename: ')
        fig.savefig(r'%s'%(savename))
        
    return fig
        
def phase_sweep(p,datapoints,max_E,V_n_arr=[1.25],L=1,rtol=1e-2,system=3):
    '''
    Uses the boundstate algorithm to return the energy eigenvalues for different
    phases.
    
    Inputs
    p-SimpleNamespace object
    Datapoints - no of points to divide phase 2pi
    JJ - SNS system
    max_E - is the region of energies
    V_n_arr - array of onsite potentials to plot.
    
    Outputs
    V_energy_arr - Array containing energy eigenvalues for each V_n
    phi_arr - Phase array
    '''
    
    phi_arr=np.linspace(0,2*np.pi,num=datapoints) #our x values
    print('Plotting a total of %s points from 0 to 2pi...'%(datapoints))
    V_energy_arr=[]
    for V_n in V_n_arr:
        energy_arr=[]
        for phi in phi_arr:
            if system==3: #hopping implementation
                JJ=make_system_3(L,p,phi=phi,V_n=V_n)
            if system==1: #phase implementation
                JJ=make_system_1(L,p,phi=phi,V_n=V_n)
            energy,_=bs.find_boundstates(JJ,-max_E,max_E,rtol=rtol)
            #energy is an array of boundstate energies at this point
            energy_arr.append(energy)

        energy_arr=np.array(energy_arr) #rows of energies against phi
        V_energy_arr.append(energy_arr)
    return V_energy_arr,phi_arr #in case this is wrong

def t_sweep(p,t_arr,max_E=1,V_n=1.25,L=1,phi=0,rtol=1e-3, system=3):
    '''
    Plots the energy of V=1.25 at 0 phase difference for different t values
    '''
    a=p.a
    B=p.B
    mu=p.mu
    alpha=p.alpha
    gap=p.gap
    energy_arr=[]
    for t in t_arr:
        p=sns(a=a,B=B,alpha=alpha,mu=mu,gap=gap,t=t)
        if system==3: #hopping implementation
            JJ=make_system_3(L,p,phi=phi,V_n=V_n)
        if system==1: #phase implementation
            JJ=make_system_1(L,p,phi=phi,V_n=V_n)
        energy,_=bs.find_boundstates(JJ,-max_E,max_E,rtol=rtol)
        energy_arr.append(energy)
    return energy_arr

def t_sweep_plot(t_arr,energy_arr,save=False):
    '''
    Takes energy_arr output from t_sweep, and t values, as inputs. Outputs
    
    '''
    fig=plt.figure()
    x=[]
    y=[]
    for i in range(len(t_arr)):
        for j in range(len(energy_arr[i])):
            x.append(t_arr[i])
            y.append(energy_arr[i][j])
    plt.scatter(x,y)
    plt.xlabel('t',fontsize=14)
    plt.ylabel('E',fontsize=14)
    plt.show()
    if save==True:
        savename=input('please enter a filename: ')
        fig.savefig(r'%s'%(savename))
        
    return x,y

def phase_sweep_plot(V_energy_arr,phi_arr,V_n_arr,save=False):
    '''
    Takes the V_energy_arr output from phase_sweep function and sorts out the
    results according to V_n_arr. Plots results.
    '''
    #first, arrange everything
    total_V=len(V_energy_arr) #total number of plotted Vs
    fig=plt.figure()
    for i in range(total_V):
        
        current_V=[]
        current_phi=[]
        for j in range(len(V_energy_arr[i])): #no of phi points
            for k in range(len(V_energy_arr[i][j])): #datapts per phi point
                current_V.append(V_energy_arr[i][j][k])
                current_phi.append(phi_arr[j])
        plt.scatter(current_phi,current_V,label=r'$V_n$=%s'%(V_n_arr[i]),
                    s=8)
    plt.legend()
    plt.xlabel(r'$\phi$',fontsize=14)
    plt.ylabel('E',fontsize=14)
    plt.ylim(-0.35,0.35)
    plt.xlim(0,np.pi*2)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    #plt.show()
    if save==True:
        filename=input('Enter file name:' ,)
        fig.savefig(filename)
    return fig

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    