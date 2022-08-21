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
import scipy as sp
import pickle
from kwant.digest import uniform,gauss

from qutip import *
from scipy.interpolate import interp1d
import scipy.constants as const

p0 = tinyarray.array([[1, 0], [0, 1]])
px = tinyarray.array([[0, 1], [1, 0]])
py = tinyarray.array([[0, -1j], [1j, 0]])
pz = tinyarray.array([[1, 0], [0, -1]])

#Constants and matrix definitions
tau_0=np.kron(p0,p0)
tau_x=np.kron(p0,px)
tau_y=np.kron(p0,py)
tau_z=np.kron(p0,pz)

sigma_0=np.kron(p0,p0)

t_x = np.kron(np.kron(np.array([[0, 1], [1, 0]]), np.eye(2)), np.eye(2)) # e-h subspace
t_y = np.kron(np.kron(np.array([[0, -1j], [1j, 0]]), np.eye(2)), np.eye(2))
t_z = np.kron(np.kron(np.array([[1, 0], [0, -1]]), np.eye(2)), np.eye(2))
sigma_x = np.kron(np.kron(np.eye(2), np.array([[0, 1], [1, 0]])), np.eye(2)) # subband subspace
sigma_y = np.kron(np.kron(np.eye(2), np.array([[0, -1j], [1j, 0]])), np.eye(2))
sigma_z = np.kron(np.kron(np.eye(2), np.array([[1, 0], [0, -1]])), np.eye(2))
s_x = np.kron(np.kron(np.eye(2), np.eye(2)), np.array([[0, 1], [1, 0]])) # spin subspace
s_y = np.kron(np.kron(np.eye(2), np.eye(2)), np.array([[0, -1j], [1j, 0]]))
s_z = np.kron(np.kron(np.eye(2), np.eye(2)), np.array([[1, 0], [0, -1]]))

hbar = const.hbar
m_e = const.m_e
eV = const.e
mu_B = const.physical_constants['Bohr magneton'][0]
#%%
#%% Archived functions

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
    lat = kwant.lattice.chain(a,norbs=4)#lat defines the lattice that we want, 

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
    Chi's implementation
    '''
    
    a=p.a
    B=p.B
    mu=p.mu
    alpha=p.alpha
    gap=p.gap
    t=p.t
    
    #Picture 2
    H_sc2=tau_z*(mu-2.*t)+sigma_z*B+tau_x*gap 
    V_N2=tau_z*t+1j*tau_z@sigma_x*alpha
    V_SN2=mat_exp(phi)@V_N2
    H_N2=(tau_z*(mu-2.*t)+sigma_z*B+V_n*tau_z)
    
    JJ = kwant.Builder() #middle region
    lat = kwant.lattice.chain(a,norbs=4)
    
    JJ[(lat(x) for x in range(1,L+1))] = H_N2
    JJ[(lat(0))]=H_sc2
    JJ[(lat(L+1))]=H_sc2

    JJ[lat.neighbors()] = np.conj(V_N2)
    JJ[(lat(0),lat(1))]=V_SN2
    JJ[(lat(L),lat(L+1))]==V_N2 #just checks neighbors is implemented correctly
    
    left_lead=kwant.Builder(kwant.TranslationalSymmetry([-a]))
    left_lead[(lat(0))] = H_sc2
    left_lead[lat.neighbors()] = V_N2
    
    right_lead=kwant.Builder(kwant.TranslationalSymmetry([a]))
    right_lead[(lat(0))] = H_sc2
    right_lead[kwant.HoppingKind((1,), lat)] = V_N2

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
    
    H_sc3=tau_z*(mu-2.*t)+sigma_z*B+tau_x*gap 
    V_N3=tau_z*t+1j*np.matmul(tau_z,sigma_x)*alpha
    V_SN3=mat_exp(phi)@V_N3
    H_N3=(tau_z*(mu-2.*t)+sigma_z*B+V_n*tau_z)      
    
    JJ = kwant.Builder() #middle region
    lat = kwant.lattice.chain(a,norbs=4)
    
    JJ[(lat(x) for x in range(1,L+1))] = H_N3
    JJ[(lat(0))]=H_sc3
    #JJ[kwant.HoppingKind((1,), lat)] = V_N3
    JJ[lat.neighbors()] = np.conjugate(V_N3)
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

def make_system_5(L,p,phi=0,V_n=1.25):
    '''
    Picture 2, with the hopping at the interface defined in the
    opposite direction. Considering neighbors sets stuff from
    right to left.
    
    Would this be the 'right' system setup??
    '''
    a=p.a
    B=p.B
    mu=p.mu
    alpha=p.alpha
    gap=p.gap
    t=p.t
    
    H_sc3=tau_z*(mu-2.*t)+sigma_z*B+tau_x*gap 
    V_N3=tau_z*t+1j*np.matmul(tau_z,sigma_x)*alpha
    V_SN3=mat_exp(phi)@V_N3
    H_N3=(tau_z*(mu-2.*t)+sigma_z*B+V_n*tau_z)      
    
    JJ = kwant.Builder() #middle region
    lat = kwant.lattice.chain(a,norbs=4)
    
    JJ[(lat(x) for x in range(1,L+1))] = H_N3
    JJ[(lat(0))]=H_sc3
    #JJ[kwant.HoppingKind((1,), lat)] = V_N3
    #JJ[lat.neighbors()] = V_N3
    JJ[(lat(1),lat(0))]=V_SN3
    
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
            if system==2: #hopping implementation
                JJ=make_system_2(L,p,phi=phi,V_n=V_n)
            if system==1: #phase implementation
                JJ=make_system_1(L,p,phi=phi,V_n=V_n)
            if system==5: #phase implementation
                JJ=make_system_5(L,p,phi=phi,V_n=V_n)
            if system==4:
                JJ=make_system_ex(p,V_N=V_n, phi=phi, L=3)
            if system==6:
                JJ=make_system_ex2(p,V_N=V_n, phi=phi, L=3)
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
        if system==2: #hopping implementation
            JJ=make_system_2(L,p,phi=phi,V_n=V_n)
        if system==1: #phase implementation
            JJ=make_system_1(L,p,phi=phi,V_n=V_n)
        if system==5: #phase implementation
                JJ=make_system_5(L,p,phi=phi,V_n=V_n)
        if system==4:
            JJ=make_system_ex(p,V_N=V_n, phi=phi, L=3)
        if system==6:
            JJ=make_system_ex2(p,V_N=V_n, phi=phi, L=3)
        energy,_=bs.find_boundstates(JJ,-max_E,max_E,rtol=rtol,)
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

def B_sweep(p,B_arr,max_E=1,V_n=1.25,L=1,phi=0,rtol=1e-3, system=3):
    '''
    Plots the energy of V=1.25 at 0 phase difference for different B values
    '''
    a=p.a
    mu=p.mu
    alpha=p.alpha
    gap=p.gap
    t=p.t
    energy_arr=[]
    for B in B_arr:
        p=sns(a=a,B=B,alpha=alpha,mu=mu,gap=gap,t=t)
        if system==3: #hopping implementation
            JJ=make_system_3(L,p,phi=phi,V_n=V_n)
        if system==2: #hopping chi implementation
            JJ=make_system_2(L,p,phi=phi,V_n=V_n)
        if system==1: #phase implementation
            JJ=make_system_1(L,p,phi=phi,V_n=V_n)
        if system==5: #phase implementation
                JJ=make_system_5(L,p,phi=phi,V_n=V_n)
        if system==4:
            JJ=make_system_ex(p,V_N=V_n, phi=phi, L=3)
        if system==6:
            JJ=make_system_ex2(p,V_N=V_n, phi=phi, L=3)
        energy,_=bs.find_boundstates(JJ,-max_E,max_E,rtol=rtol)
        energy_arr.append(energy)
    return energy_arr

def B_sweep_plot(B_arr,energy_arr,save=False):
    '''
    Takes energy_arr output from B_sweep, and B values, as inputs. Outputs
    
    '''
    fig=plt.figure()
    x=[]
    y=[]
    for i in range(len(B_arr)):
        for j in range(len(energy_arr[i])):
            x.append(B_arr[i])
            y.append(energy_arr[i][j])
    plt.scatter(x,y)
    plt.xlabel('B',fontsize=14)
    plt.ylabel('E',fontsize=14)
    plt.show()
    if save==True:
        savename=input('please enter a filename: ')
        fig.savefig(r'%s'%(savename))
        
    return x,y,fig
    
def make_system_ex(p,V_N=1.25, phi=0, L=3):
    
    a=p.a
    B=p.B
    mu=p.mu
    alpha_so=p.alpha
    delta=p.gap
    t=p.t
    lat = kwant.lattice.chain(norbs=4)
    syst = kwant.Builder()

    syst[(lat(0))] = (mu-2.*t)*tau_z + B*sigma_z + delta*tau_x
    syst[(lat(x) for x in range(1, L-1))] = (mu-2.*t+V_N)*tau_z + B*sigma_z
    syst[(lat(L-1))] = (mu-2.*t)*tau_z + B*sigma_z + delta*tau_x

    syst[(lat(0), lat(1))] = np.matmul(sp.linalg.expm(phi*1j*tau_z), t*tau_z + alpha_so*1j*np.matmul(tau_z, sigma_x))
    syst[(lat(1), lat(2))] = t*tau_z + alpha_so*1j*np.matmul(tau_z, sigma_x)


    sym_left = kwant.TranslationalSymmetry([-a])
    lead0 = kwant.Builder(sym_left)
    lead0[(lat(-1))] = (mu-2.*t)*tau_z + B*sigma_z + delta*tau_x
    lead0[lat.neighbors()] = t*tau_z + alpha_so*1j*np.matmul(tau_z, sigma_x)
    sym_right = kwant.TranslationalSymmetry([a])
    lead1 = kwant.Builder(sym_right)
    lead1[(lat(L))] =  (mu-2.*t)*tau_z + B*sigma_z + delta*tau_x
    lead1[lat.neighbors()] = t*tau_z + alpha_so*1j*np.matmul(tau_z, sigma_x)
    syst.attach_lead(lead0)
    syst.attach_lead(lead1)
    syst=syst.finalized()
    return syst
    
    
def make_system_ex2(p,V_N=1.25, phi=0, L=3):
    '''
    Reversed hopping at SN interface
    '''
    a=p.a
    B=p.B
    mu=p.mu
    alpha_so=p.alpha
    delta=p.gap
    t=p.t
    lat = kwant.lattice.chain(norbs=4)
    syst = kwant.Builder()

    syst[(lat(0))] = (mu-2.*t)*tau_z + B*sigma_z + delta*tau_x
    syst[(lat(x) for x in range(1, L-1))] = (mu-2.*t+V_N)*tau_z + B*sigma_z
    syst[(lat(L-1))] = (mu-2.*t)*tau_z + B*sigma_z + delta*tau_x

    syst[(lat(1), lat(0))] = np.matmul(sp.linalg.expm(phi*1j*tau_z), t*tau_z + alpha_so*1j*np.matmul(tau_z, sigma_x))
    syst[(lat(1), lat(2))] = t*tau_z + alpha_so*1j*np.matmul(tau_z, sigma_x)


    sym_left = kwant.TranslationalSymmetry([-a])
    lead0 = kwant.Builder(sym_left)
    lead0[(lat(-1))] = (mu-2.*t)*tau_z + B*sigma_z + delta*tau_x
    lead0[lat.neighbors()] = t*tau_z + alpha_so*1j*np.matmul(tau_z, sigma_x)
    sym_right = kwant.TranslationalSymmetry([a])
    lead1 = kwant.Builder(sym_right)
    lead1[(lat(L))] =  (mu-2.*t)*tau_z + B*sigma_z + delta*tau_x
    lead1[lat.neighbors()] = t*tau_z + alpha_so*1j*np.matmul(tau_z, sigma_x)
    syst.attach_lead(lead0)
    syst.attach_lead(lead1)
    syst=syst.finalized()
    return syst

#%% Active functions

def gradient_addition(start_ind,data,phi_arr,printing=False,single=False,rtol=1/10):
    '''
    Function that returns the branch of solutions given the initial specified index.
    Note: for this to work, the data MUST be pre-processed. No missing points.
    Must do interpolation+reflectiobn on dataset.
    
    Returns
    branch_data - list of datapoints
    new_phi - truncated phi_array in case some were removed during tracking
    (new and old phi should be same if data processing was done)
    '''
    if single:
        start_val=data[start_ind]
    else:
        start_val=data[0][start_ind]
    delphi=phi_arr[1]-phi_arr[0]
    new_phi=[phi_arr[0]]
    branch_data=[start_val]
    
    current_i=1
    grad=[(j-branch_data[0])/(delphi) for j in data[1]]
    idx=np.abs(grad).argmin()
    current_grad=grad[idx]
    #print(grad,current_grad)
    if single:
        branch_data.append(data[idx])
    else:
        branch_data.append(data[1][idx])
    new_phi.append(phi_arr[1])
    
    for i in range(2,len(phi_arr)):
        total_i=i-current_i
        #print(i,len(branch_data))
        #print(data[i],branch_data[-1])
        if data[i].any():
            projected=branch_data[-1]+current_grad*total_i*delphi
            distances=[j-projected for j in data[i]]
            idx=np.abs(distances).argmin()
            new_point=data[i][idx]
            #print('New',new_grad,'Current',current_grad,i)
            if printing:
                print('i:',i,'prev',branch_data[-1],'new',
                          new_point,'project',projected)
                print('i:',i,'new dist',np.abs(distances[np.abs(distances).argmin()]),
                      'project dist',np.abs(current_grad*total_i*delphi))
    
            if np.abs(new_point)<=np.abs(start_val*(1+rtol)):
                if printing:
                    print('i:',i,'Success')
                current_grad=(new_point-branch_data[-1])/(total_i*delphi)
                current_i=i
                branch_data.append(data[i][idx])
                new_phi.append(phi_arr[i])

    return branch_data,new_phi

def mixed_system_vary_eta(L_A=3700., W=1400., a=6.0583, m=0.023,
                          alpha=350., mu=0., U_0=0., delta=0.182, B_Tesla=0.85,
                          phi=0., g=14.7, salt='chizhang',eta=0.35355):
    '''
    System Hamiltonian creation with sub-band mixing.
    '''
    L = int(np.round(L_A/a))
    t = (const.hbar**2.)/(2.*m*const.m_e*(a*1e-10)**2.)/const.e*1000.
    E_plus = (6.*const.hbar**2.)/(m*const.m_e*(W*1e-10)**2.)/const.e*1000.
    E_minus = (-2.*const.hbar**2.)/(m*const.m_e*(W*1e-10)**2.)/const.e*1000.
    eta = eta
    nu = alpha/(2.*a)
    B = g*mu_B*B_Tesla/2./const.e*1000.
    lat = kwant.lattice.chain()
    syst = kwant.Builder()

    #### Define the scattering region. ####
    ham = (2.*t+E_plus-mu)*t_z + E_minus*sigma_z@t_z + eta*s_x@sigma_y@t_z + B*s_x
    hop = -t*t_z + 1j*nu*s_y@t_z
    ham_scl = delta*(np.cos(-phi/2.)*t_x - np.sin(-phi/2.)*t_y)
    ham_scr = delta*(np.cos(phi/2.)*t_x - np.sin(phi/2.)*t_y)

    syst[(lat(x) for x in range(0, L))] = ham
    #syst[(lat(int(np.round(L/2.))))] = ham-50.*t_z
    syst[((lat(x), lat(x+1)) for x in range(0, L-1))] = hop
    
    #### Define the leads. ####
    sym_left = kwant.TranslationalSymmetry([-1])
    lead0 = kwant.Builder(sym_left)
    lead0[(lat(-1))] = ham+ham_scl
    lead0[lat.neighbors()] = hop
    sym_right = kwant.TranslationalSymmetry([1])
    lead1 = kwant.Builder(sym_right)
    lead1[(lat(L))] = ham+ham_scr
    lead1[lat.neighbors()] = hop
    
    #### Attach the leads and return the system. ####
    syst.attach_lead(lead0)
    syst.attach_lead(lead1)
    
    return syst.finalized()

def qubit_energies(Ec,Ej,charge_arr,N,ng,eigvals=4):
    """
    Plot energy levels as a function of bias parameter B_vec.
    """
    energies =general_hamiltonian(Ec, Ej, N, ng,charge_arr).eigenenergies(eigvals=eigvals)
    energies=np.array(np.real(energies))
    return energies

def general_hamiltonian(Ec,Ej,N,ng,charge_arr):
    '''
    Ec,Ej,ng as per usual
    N is no. of levels to consider
    charge_arr is list of Fourier coefficients for cos(i*phi/2) WITHOUT OFFSET
    '''
    if N<len(charge_arr):
        raise Exception('Considering %s charge transfer but only %s levels'%(
                        len(charge_arr),N))
    m = np.diag(Ec * (np.arange(-N,N+1)-ng)**2)
    m+= 0.5 * Ej * (np.diag(-np.ones(2*N-1), 2) +  np.diag(-np.ones(2*N-1), -2))
    for i in range(len(charge_arr)):
        amp=charge_arr[i]
        term=np.diag(np.ones(2*N-i), i+1)
        hc=np.diag(np.ones(2*N-i), -(i+1))      
        m+=0.5*amp*(term + hc)
    
    return Qobj(m)

def arrange_E(E_arr,phi_arr):
    '''
    Takes the output from boundstate algorithm and the scanned phases
    and arranges them into a single list for plotting.
    '''
    new_y=[]
    new_x=[]
    for i in range(len(phi_arr)):
        for j in range(len(E_arr[i])):
            new_x.append(phi_arr[i])
            new_y.append(E_arr[i][j])
    
    return new_x,new_y

def get_minima(system,lead=0,resolution=300):
    '''
    Input a system
    Outputs a list of minima in the lead's dispersion plot'
    'lead' is whether its left (0) or right(1)
    
    FOR SOME REASON THIS DOESNT WORK
    '''
    bands=kwant.physics.dispersion.Bands(system.leads[lead])
    momenta = np.linspace(-.1, .1, resolution)
    band_no=len(bands(0))
    energies = [bands(k) for k in momenta]
    #adding endpoints
    minima=[]
    k=[]
    for i in range(band_no):
        arr=[]
        k_arr=[]
        if round(abs(energies[0][i]),5)<round(abs(energies[1][i]),5):
            arr.append(round(energies[0][i],5))
            k_arr.append(round(momenta[0],3))
        if round(abs(energies[-1][i]),5)<round(abs(energies[-2][i]),5):
            arr.append(round(energies[-1][i],5))
            k_arr.append(round(momenta[-1],3))
        for E in range(1,len(energies)-1):
            prev=round(abs(energies[E-1][i]),5)
            current=round(abs(energies[E][i]),5)
            nex=round(abs(energies[E+1][i]),5)
            if prev>=current:
                if nex>=current:
                    arr.append(round(energies[E][i],5))
                    k_arr.append(round(momenta[E],3))

        minima.append(arr)
        k.append(k_arr)
    return minima,k

def make_system_keselman(L,p,disorder=False,profile='Gaussian'):
    '''
    Trying to implement the Hamiltonian in the Avila paper
    To be consistent with the Istas paper, we will have 1 extra site from the
    left/right lead in the system to implement hopping etc.
    
    For L=0, S-S hopping takes value of kL
    '''
    
    
    delmu_n=p.delmu_n 
    delmu_l=p.delmu_l
    delmu_r=p.delmu_r
    a=p.a
    B=p.B
    mu=p.mu
    alpha=p.alpha
    gap=p.delta
    t=p.t
    nu=alpha/a
    phi=p.phi

    kappa_L=p.kappa_L
    kappa_R=p.kappa_R
    
    #Defining the Hamiltonians and hoppings.
    h_sc_l=-(mu+delmu_l-2*t)*tau_z+B*tau_z@sigma_x+gap*tau_y@sigma_y
    v_sc=-(t*tau_z+1j*nu*tau_z@sigma_y)
    SN_hopping=-kappa_L*(sp.linalg.expm(1j*phi/2*tau_z)@(t*tau_z+1j*nu*tau_z@sigma_y))
    NS_hopping=-kappa_R*(t*tau_z+1j*nu*tau_z@sigma_y)
    h_n=-((mu+delmu_n)-2*t)*tau_z+B*tau_z@sigma_x
    h_v=-(t*tau_z+1j*nu*tau_z@sigma_y)
    h_sc_r=-(mu+delmu_r-2*t)*tau_z+B*tau_z@sigma_x+gap*tau_y@sigma_y

    #build system
    JJ=kwant.Builder()
    lat = kwant.lattice.chain(a)
    
    if disorder==True:
        if profile=='Gaussian':
            func=gauss
        elif profile=='Uniform':
            func=uniform
        else:
            raise Exception('Please specify Uniform or Gaussian disorder profile')
        U0=p.U0
        salt=p.salt
        def scatter_onsite(site):
            disorder=U0 * (func(repr(site), repr(salt)) - 0.5)*tau_z
            return disorder+h_n
    
    #on-site potentials
    if disorder==True:
        JJ[(lat(x) for x in range(1,L+1))] = scatter_onsite
    else:
        JJ[(lat(x) for x in range(1,L+1))]=h_n
    JJ[(lat(0))]=h_sc_l #1 lead site in scattering region
    JJ[(lat(L+1))]=h_sc_r

    #hoppings
    JJ[lat.neighbors()] = np.conj(h_v).T
    JJ[(lat(L),lat(L+1))]=NS_hopping
    JJ[(lat(0),lat(1))]=SN_hopping
    
    #lead hoppings and on-site potentials
    left_lead=kwant.Builder(kwant.TranslationalSymmetry([-a]))
    left_lead[(lat(0))] = h_sc_l
    left_lead[lat.neighbors()] = v_sc
    
    right_lead=kwant.Builder(kwant.TranslationalSymmetry([a]))
    right_lead[(lat(0))] = h_sc_r
    right_lead[lat.neighbors()] = v_sc

    JJ.attach_lead(left_lead) 
    JJ.attach_lead(right_lead)
    JJ=JJ.finalized()
    return JJ

def make_system_mixed(L_A=3700., W=1400., a=6.0583, m=0.023, alpha=350., mu=0., U_0=0., delta=0.182, B_Tesla=0.85, phi=0., g=14.7, salt='chizhang'):
    L = int(np.round(L_A/a))
    t = (const.hbar**2.)/(2.*m*const.m_e*(a*1e-10)**2.)/const.e*1000.
    E_plus = (6.*const.hbar**2.)/(m*const.m_e*(W*1e-10)**2.)/const.e*1000.
    E_minus = (-2.*const.hbar**2.)/(m*const.m_e*(W*1e-10)**2.)/const.e*1000.
    eta = np.sqrt(2)*alpha/W
    nu = alpha/(2.*a)
    B = g*mu_B*B_Tesla/2./const.e*1000.
    lat = kwant.lattice.chain()
    syst = kwant.Builder()
    
    #### Define the scattering region. ####
    ham = (2.*t+E_plus-mu)*t_z + E_minus*sigma_z@t_z + eta*s_x@sigma_y@t_z + B*s_x
    hop = -t*t_z + 1j*nu*s_y@t_z
    ham_scl = delta*(np.cos(-phi/2.)*t_x - np.sin(-phi/2.)*t_y)
    ham_scr = delta*(np.cos(phi/2.)*t_x - np.sin(phi/2.)*t_y)

    syst[(lat(x) for x in range(0, L))] = ham
    #syst[(lat(int(np.round(L/2.))))] = ham-50.*t_z
    syst[((lat(x), lat(x+1)) for x in range(0, L-1))] = hop
    
    #### Define the leads. ####
    sym_left = kwant.TranslationalSymmetry([-1])
    lead0 = kwant.Builder(sym_left)
    lead0[(lat(-1))] = ham+ham_scl
    lead0[lat.neighbors()] = hop
    sym_right = kwant.TranslationalSymmetry([1])
    lead1 = kwant.Builder(sym_right)
    lead1[(lat(L))] = ham+ham_scr
    lead1[lat.neighbors()] = hop
    
    #### Attach the leads and return the system. ####
    syst.attach_lead(lead0)
    syst.attach_lead(lead1)
    
    return syst.finalized()
    
def fourier_decompose(y,phi_arr, modes=10,offset=False,odd=False):
    '''
    It could be because there are gaps in the data, so we need to do a
    non-uniform discrete fourier transform
    '''
    N=len(y)
    phi_arr=np.array(phi_arr)
    y=np.array(y)
    if offset:
        amplitudes=np.zeros(modes+1)
        amp_list=range(0,modes+1) #assume no constant offset
    else:
        amplitudes=np.zeros(modes)
        amp_list=range(1,modes+1) #assume no constant offset
    for i in range(len(y)):
        amplitudes+=2/N*(y[i])*np.cos(amp_list*phi_arr[i]/2)
    if offset:
        amplitudes[0]/=2
    if odd:
        odd_amps=range(1,modes+1)
        odd_amplitudes=np.zeros(modes)
        for i in range(len(y)):
            odd_amplitudes+=2/N*(y[i])*np.sin(odd_amps*phi_arr[i]/2)
        return amplitudes, odd_amplitudes
    return amplitudes

def process_data(arr,phi_arr,fill=True,extend=True,single=False):
    '''
    Takes the data, fills missing points, extend to 4pi
    Interpolation is not included since that one requires the tracking of
    specific branch solutions.
    If single, we are analysing only one solution/phi.
    '''
    if single:
        data=arr.copy()
        phic=np.array(phi_arr.copy())
        if fill:
            for j in range(len(data)):
                testdataj=[round(k,5) for k in data[j]]
                data[j]=np.append(data[j],[-k for k in data[j] if round(-k,5) not in testdataj])
                
                #zeros=[k for k in data[j] if np.abs(k)<1e-10]
                #data[j]=np.append(data[j],[-k for k in data[j] if round(-k,10) not in zeros])
        if extend: #extends the solution from 2pi to 4pi
            list1=list(reversed(data))
            data=data+list1[1:]
            list2=4*np.pi-np.array(list(reversed(phic))[1:])
            phic=np.append(phic,list2)
    
        return data,phic         
        
    l=len(arr)
    arrc=arr.copy() #making changes to new array rather than touching old one.
    phic=np.array(phi_arr.copy())
    for i in range(l):
        data=arrc[i]
        if fill: #fills all missing solutions. assuming they were lost due to rtol
            for j in range(len(data)):
                testdataj=[round(k,5) for k in data[j]]
                data[j]=np.append(data[j],[-k for k in data[j] if round(-k,5) not in testdataj])
                
        if extend: #extends the solution from 2pi to 4pi
            list1=list(reversed(data))
            arrc[i]=data+list1[1:]

    if extend: #extends the phi array as a final step
        list2=4*np.pi-np.array(list(reversed(phic))[1:])
        phic=np.append(phic,list2)
    
    return arrc,phic

def interpolate(branch,phi,resolution=1001):
    interf=interp1d(phi,branch,kind='cubic')
    interx=np.linspace(0,4*np.pi,resolution)
    intery=interf(interx)
    return interx, intery

def mixed_system_vary_eta_disorder(L_A=3700., W=1400., a=6.0583, m=0.023,
                          alpha=350., mu=0., U_0=0., delta=0.182, B_Tesla=0.85,
                          phi=0., g=14.7, salt='chizhang',eta=0.35355):
    '''
    System Hamiltonian creation with sub-band mixing.
    '''
    L = int(np.round(L_A/a))
    t = (const.hbar**2.)/(2.*m*const.m_e*(a*1e-10)**2.)/const.e*1000.
    E_plus = (6.*const.hbar**2.)/(m*const.m_e*(W*1e-10)**2.)/const.e*1000.
    E_minus = (-2.*const.hbar**2.)/(m*const.m_e*(W*1e-10)**2.)/const.e*1000.
    eta = eta
    nu = alpha/(2.*a)
    B = g*mu_B*B_Tesla/2./const.e*1000.
    lat = kwant.lattice.chain()
    syst = kwant.Builder()

    #### Define the scattering region. ####
    ham = (2.*t+E_plus-mu)*t_z + E_minus*sigma_z@t_z + eta*s_x@sigma_y@t_z + B*s_x
    hop = -t*t_z + 1j*nu*s_y@t_z
    ham_scl = delta*(np.cos(-phi/2.)*t_x - np.sin(-phi/2.)*t_y)
    ham_scr = delta*(np.cos(phi/2.)*t_x - np.sin(phi/2.)*t_y)
    def scatter_onsite(site):
        disorder=U_0 * (gauss(repr(site), repr(salt)) - 0.5)*t_z
        return disorder+ham

    syst[(lat(x) for x in range(0, L))] = scatter_onsite
    #syst[(lat(int(np.round(L/2.))))] = ham-50.*t_z
    syst[((lat(x), lat(x+1)) for x in range(0, L-1))] = hop
    
    #### Define the leads. ####
    sym_left = kwant.TranslationalSymmetry([-1])
    lead0 = kwant.Builder(sym_left)
    lead0[(lat(-1))] = ham+ham_scl
    lead0[lat.neighbors()] = hop
    sym_right = kwant.TranslationalSymmetry([1])
    lead1 = kwant.Builder(sym_right)
    lead1[(lat(L))] = ham+ham_scr
    lead1[lat.neighbors()] = hop
    
    #### Attach the leads and return the system. ####
    syst.attach_lead(lead0)
    syst.attach_lead(lead1)
    
    return syst.finalized()

def new_gradient_addition(data,phi_arr):
    '''
    Function that tracks the lowest-energy solution. Records gradient to
    account for sign flips
    
    Returns
    branch_data - list of datapoints
    new_phi - truncated phi_array in case some were removed during tracking
    (new and old phi should be same if data processing was done)
    '''
    start_ind=int(len(data[0])/2)
    start_val=np.sort(data[0])[start_ind]
    delphi=phi_arr[1]-phi_arr[0]
    new_phi=[phi_arr[0]]
    branch_data=[start_val]
    
    current_i=1
    current_grad=(np.sort(data[1])[int(len(data[1])/2)]-start_val)/delphi
    branch_data.append(np.sort(data[1])[int(len(data[1])/2)])
    new_phi.append(phi_arr[1])
    sign=0
    for i in range(2,len(phi_arr)):
        total_i=i-current_i
        if data[i].any():
            projected=branch_data[-1]+current_grad*total_i*delphi
            if projected*branch_data[-1]<0.: #flip sign, flip tracking
                sign+=1
                sign=sign%2

            if sign==1: #negative
                new_point=np.sort(data[i])[int(len(data[i])/2-1)]
            else:
                new_point=np.sort(data[i])[int(len(data[i])/2)]

            current_grad=(new_point-branch_data[-1])/(total_i*delphi)
            current_i=i
            branch_data.append(new_point)
            
            new_phi.append(phi_arr[i])
    return branch_data,new_phi

#%%
def chi_gradient_addition(start_ind,data,phi_arr,printing=False,single=False,rtol=1/10, second_ind_correct=None):
    '''
    Function that returns the branch of solutions given the initial specified index.
    Note: for this to work, the data MUST be pre-processed. No missing points.
    Must do interpolation+reflectiobn on dataset.
    
    Returns
    branch_data - list of datapoints
    new_phi - truncated phi_array in case some were removed during tracking
    (new and old phi should be same if data processing was done)
    '''
    if single:
        start_val=data[start_ind]
    else:
        start_val=data[0][start_ind]
    delphi=phi_arr[1]-phi_arr[0]
    new_phi=[phi_arr[0]]
    branch_data=[start_val]
    
    current_i=1
    grad=[(j-branch_data[0])/(delphi) for j in data[1]]
    idx=np.abs(grad).argmin()
    current_grad=grad[idx]     
    if second_ind_correct is not None:
        idx=second_ind_correct
        current_grad = grad[idx]
    #print(grad,current_grad)
    if single:
        branch_data.append(data[idx])
    else:
        branch_data.append(data[1][idx])
    new_phi.append(phi_arr[1])
    
    for i in range(2,len(phi_arr)):
        total_i=i-current_i
        #print(i,len(branch_data))
        #print(data[i],branch_data[-1])
        if data[i].any():
            projected=branch_data[-1]+current_grad*total_i*delphi
            distances=[j-projected for j in data[i]]
            idx=np.abs(distances).argmin()
            new_point=data[i][idx]
            #print('New',new_grad,'Current',current_grad,i)
            if printing:
                print('i:',i,'prev',branch_data[-1],'new',
                          new_point,'project',projected)
                print('i:',i,'new dist',np.abs(distances[np.abs(distances).argmin()]),
                      'project dist',np.abs(current_grad*total_i*delphi))
    
            if np.abs(new_point)<=np.abs(start_val*(1+rtol)):
                if printing:
                    print('i:',i,'Success')
                current_grad=(new_point-branch_data[-1])/(total_i*delphi)
                current_i=i
                branch_data.append(data[i][idx])
                new_phi.append(phi_arr[i])

    return branch_data,new_phi

#%% Testing
'''
filename='Flat_band_highres'
import pickle
import matplotlib.pyplot as plt
arr,waves,etas=pickle.load(open(filename,'rb'))
phi_arr=np.linspace(0,2*np.pi,31)
new_data,new_phi=process_data(arr,phi_arr,fill=True,extend=True)
print('Done!')

for i in range(7,len(arr)-1):
    x,y=arrange_E(new_data[i],new_phi)
    x2,y2=arrange_E(arr[i],phi_arr)
    index=int(len(arr[i][0])/2) #to track the middle positive (Majorana) soln
    if index==4:
        print(arr[i][0])
    plt.figure()
    plt.scatter(x,y,label='Processed',s=1)
    plt.scatter(x2,y2,label='Original',marker='x',s=1)
    branch,phi=gradient_addition(index,new_data[i],new_phi,printing=False)
    interx,intery=interpolate(branch,phi)
    plt.plot(interx,intery)
    plt.legend()
    
#successful data processing and solution tracking.
'''