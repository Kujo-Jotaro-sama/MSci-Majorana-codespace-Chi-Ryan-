#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 20:52:29 2022

@author: afasjasd
"""
#%%
import pickle
e_phi_B_scan_without_eta = []
for i in range(1, 101):
    e_temp = pickle.load(open('../debugged_probes/data/e_phi_B_rerun_without_eta/e_B_index_%i' % (i), 'rb'))
    e_phi_B_scan_without_eta.append(e_temp)
    
#%%
e_phi_B_scan = []
for i in range(1, 101):
    e_temp = pickle.load(open('../debugged_probes/data/e_phi_B_rerun/e_B_index_%i' % (i), 'rb'))
    e_phi_B_scan.append(e_temp)
    
#%%
B_gap_no_eta_0_final = pickle.load(open('../debugged_probes/data/8-orb no eta gaps/subband 0 gap', 'rb'))
B_gap_no_eta_1_final = pickle.load(open('../debugged_probes/data/8-orb no eta gaps/subband 1 gap', 'rb'))

#%%
phi_eta_tp_trialc = pickle.load(open('./phi_eta_tp_trialc', 'rb'))
phi_eta_tp_triald = pickle.load(open('./phi_eta_tp_triald', 'rb'))

#%%
e_no_barrier = pickle.load(open('./Data/last_figure/e_B_index_73', 'rb'))
e_barrier1 = pickle.load(open('./Data/last_figure/e_barrier_m1', 'rb'))
e_barrier5 = pickle.load(open('./e_barrier_m5', 'rb'))

#%%
import scipy.constants as const
mu_B = const.physical_constants['Bohr magneton'][0]
B_array = np.linspace(0., 14.7*mu_B*2./2./const.e*1000., 101)

#%%
print(B_array[72])

#%%
import Functions as fn
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 24})

import kwant
#%%
def phi_plot(energy_array, delta=0.182, title=r'$L=1000, B=0.0182, \mu=0, \Delta=0.182$'):
    ex=[]
    ey=[]
    
    phi = np.linspace(0, 4*np.pi, 41)
    for i in range(len(energy_array)):
        for j in range(len(energy_array[i])):
            ex.append(phi[i])
            ey.append(energy_array[i][j])
        
    ex = np.array(ex)
    ey = np.array(ey)/delta
    #plt.axhline(gap/delta, color='r')
    #plt.axhline(-gap/delta, color='r')
    plt.scatter(ex, ey, s=5)
    #plt.xlim(B_range[0], B_range[1])
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$E/\Delta$')
    #plt.ylim(-1, 1)
    #plt.title(title, fontsize=14)
    #plt.savefig('./Real params/'+figure_title, dpi=300)
    
#%%
## realistic parameters InAs with Al shell
# Energy unit: meV, length unit: A
#g = 14.7 #https://doi.org/10.1016/0375-9601(67)90541-5

def make_system_mixed(L_A=3700., W=1400., a=6.0583*10., m=0.023, alpha=350., mu=0., delta=0.182, B=0., phi=0., eta=None):
    
    #a = 6.0583 #https://en.wikipedia.org/wiki/Indium_arsenide
    L = int(np.round(L_A/a))
    t = (const.hbar**2.)/(2.*m*const.m_e*(a*1e-10)**2.)/const.e*1000.
    E_plus = (6.*const.hbar**2.)/(m*const.m_e*(W*1e-10)**2.)/const.e*1000.
    E_minus = (-2.*const.hbar**2.)/(m*const.m_e*(W*1e-10)**2.)/const.e*1000.
    if eta is None:
        eta = np.sqrt(2)*alpha/W
    else:
        eta = eta
    nu = alpha/(2.*a)
    #B = g*mu_B*B_Tesla/2./const.e*1000.
    print('Parameters: L, t, E_plus, E_minus, eta, nu, B')
    print([L, t, E_plus, E_minus, eta, nu, B])
    lat = kwant.lattice.chain()
    syst = kwant.Builder()
    
    # Pauli matrices
    t_x = np.kron(np.kron(np.array([[0, 1], [1, 0]]), np.eye(2)), np.eye(2)) # e-h subspace
    t_y = np.kron(np.kron(np.array([[0, -1j], [1j, 0]]), np.eye(2)), np.eye(2))
    t_z = np.kron(np.kron(np.array([[1, 0], [0, -1]]), np.eye(2)), np.eye(2))
    sigma_x = np.kron(np.kron(np.eye(2), np.array([[0, 1], [1, 0]])), np.eye(2)) # subband subspace
    sigma_y = np.kron(np.kron(np.eye(2), np.array([[0, -1j], [1j, 0]])), np.eye(2))
    sigma_z = np.kron(np.kron(np.eye(2), np.array([[1, 0], [0, -1]])), np.eye(2))
    s_x = np.kron(np.kron(np.eye(2), np.eye(2)), np.array([[0, 1], [1, 0]])) # spin subspace
    s_y = np.kron(np.kron(np.eye(2), np.eye(2)), np.array([[0, -1j], [1j, 0]]))
    s_z = np.kron(np.kron(np.eye(2), np.eye(2)), np.array([[1, 0], [0, -1]]))
    
    #### Define the scattering region. ####
    ham = (2.*t+E_plus-mu)*t_z + E_minus*sigma_z@t_z + eta*s_x@sigma_y@t_z + B*s_x
    hop = -t*t_z + 1j*nu*s_y@t_z
    ham_scl = delta*(np.cos(-phi/2.)*t_x - np.sin(-phi/2.)*t_y)
    ham_scr = delta*(np.cos(phi/2.)*t_x - np.sin(phi/2.)*t_y)
    
    syst[(lat(x) for x in range(0, L))] = ham
    syst[((lat(x), lat(x+1)) for x in range(0, L-1))] = hop
        
    
    #### Define the leads. ####
    sym_left = kwant.TranslationalSymmetry([-1])
    lead0 = kwant.Builder(sym_left)
    lead0[(lat(-1))] = ham+ham_scl
    lead0[lat.neighbors()] = hop.T.conj()
    sym_right = kwant.TranslationalSymmetry([1])
    lead1 = kwant.Builder(sym_right)
    lead1[(lat(L))] =  ham+ham_scr
    lead1[lat.neighbors()] = hop.T.conj()

    #### Attach the leads and return the system. ####
    syst.attach_lead(lead0)
    syst.attach_lead(lead1)

    return syst.finalized()

#%%
def B_gap_scan(L_A=3700., W=1400., a=6.0583*10, m=0.023, alpha=350., mu=0.887, delta=0.182, B_array=[0., 14.7*mu_B*2./2./const.e*1000.], phi=0., eta=None):
    band_gap_array=[]
    higher_band_gap_array=[]
    for i in np.linspace(B_array[0], B_array[1], 101):
        print(i)
        syst = make_system_mixed(L_A=L_A, W=W, a=a, m=m, alpha=alpha, mu=mu, delta=delta, B=i, phi=phi, eta=eta)
        bands=kwant.physics.Bands(syst.leads[1])
        momenta=np.linspace(-np.pi,np.pi,100001)
        energies=[bands(k) for k in momenta]
        x=np.array(energies)
        en=(x)
        band_gap_array.append(np.min(en[:,4]))
        higher_band_gap_array.append(np.min(en[:,5]))
    return np.array(band_gap_array), np.array(higher_band_gap_array)

#%%
def B_gap_calc(L_A=3700., W=1400., a=6.0583*10, m=0.023, alpha=350., mu=0.887, delta=0.182, B=0., phi=0., eta=None):
    syst = make_system_mixed(L_A=L_A, W=W, a=a, m=m, alpha=alpha, mu=mu, delta=delta, B=B, phi=phi, eta=eta)
    bands=kwant.physics.Bands(syst.leads[1])
    momenta=np.linspace(-np.pi,np.pi,100001)
    energies=[bands(k) for k in momenta]
    x=np.array(energies)
    en=(x)
    return np.min(en[:,4])


#%%
#barrier_gap
barrier_gap = B_gap_calc(B=14.7*mu_B*2./2./const.e*1000./100.*72., eta=None)

#%%
#no barrier
plt.figure(figsize=(6., 4.))
n_en,n_phi=fn.process_data(e_no_barrier, np.linspace(0, 2*np.pi, 41),fill=True,extend=False,single=True)
n_en[0]=np.sort(n_en[0])
print(n_en[0])
index=int(len(n_en[0])/2)
branch,phi=fn.chi_gradient_addition(-2, n_en,n_phi)
#plt.plot(phi, branch)
phi=n_phi*2.
branch = np.r_[branch[:21], np.flip(branch[:20])]
#plt.plot(phi, branch)
x,y=fn.arrange_E(n_en,n_phi*2.)
#x,y = fn.arrange_E(e_no_barrier, n_phi*2.)
#x, y = fn.arrange_E(e_phi_B_scan[72], n_phi*2.)
plt.scatter(x,np.array(y)/0.182,s=15)
plt.scatter(phi,branch/0.182,s=15, c='#ff7f0e')
#plt.scatter(phi,-branch/0.182,s=15, c='#ff7f0e')
plt.axhline(barrier_gap/0.182, c='r', ls='-.')
plt.axhline(-barrier_gap/0.182, c='r', ls='-.')
plt.xlabel(r'$\phi$')
plt.xlim(-0.1, 4*np.pi+0.1)
#plt.ylim(-ratio*barrier_gap/0.182, ratio*barrier_gap/0.182)
plt.ylabel(r'$\epsilon/\Delta$')
plt.title(r'$U_0=0$', fontsize=24)
plt.xticks(np.linspace(0, 4*np.pi, 5), [r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'])
#plt.scatter([np.pi/4.], [0.2*0.9], s=150, marker='*', c='#b0f246', zorder=100)
#plt.savefig('./thesis_result_figures/last_fig/U_0=0', dpi=300, bbox_inches='tight')
#%%

interx,intery=fn.interpolate(branch,n_phi*2.)
intery=intery/1000*1.6*10**-19/(6.63*10**-34)
coeffs=fn.fourier_decompose(intery,interx,modes=10,offset=False)
print(coeffs)

Ec=512*10**6
Ej=4.7*10**9
ng=np.linspace(-2,2,9+8*60)

energies=np.array([fn.qubit_energies(Ec,Ej,coeffs,50,g)/10**6 for g in ng])
split=min(energies[:,3]-energies[:,0]-(energies[:,2]-energies[:,1]))
energy = fn.qubit_energies(Ec,Ej,coeffs,50,0.5)/10**6
split_indi = energy[3]-energy[0]-(energy[2]-energy[1])

plt.plot(ng, energies)
print(split_indi)

#%%
#reconstruction
x = np.linspace(0, 4*np.pi, 101)
y = np.zeros(101)
for i in range(len(coeffs)):
    y += coeffs[i]*np.cos((i+1.)*x/2.)

plt.plot(x, y/0.182)


#%%
plt.figure(figsize=(6., 4.))
n_en,n_phi=fn.process_data(e_barrier1, np.linspace(0, 2*np.pi, 41),fill=True,extend=False,single=True)
n_en[0]=np.sort(n_en[0])
print(n_en[0])
index=int(len(n_en[0])/2)
branch,phi=fn.chi_gradient_addition(-1, n_en,n_phi)
#plt.plot(phi, branch)
phi=n_phi*2.
branch = np.r_[branch[:21], np.flip(branch[:20])]
#plt.plot(phi, branch)
x,y=fn.arrange_E(n_en,n_phi*2.)
#x,y = fn.arrange_E(e_no_barrier, n_phi*2.)
#x, y = fn.arrange_E(e_phi_B_scan[72], n_phi*2.)
plt.scatter(x,np.array(y)/0.182,s=15)
plt.scatter(phi,branch/0.182,s=15, c='#ff7f0e')
#plt.scatter(phi,-branch/0.182,s=15, c='#ff7f0e')
plt.axhline(barrier_gap/0.182, c='r', ls='-.')
plt.axhline(-barrier_gap/0.182, c='r', ls='-.')
plt.xlabel(r'$\phi$')
plt.xlim(-0.1, 4*np.pi+0.1)
#plt.ylim(-0.2, 0.2)
#plt.ylabel(r'$\epsilon/\Delta$')
plt.yticks(np.linspace(-0.1, 0.1, 3), labels=['']*3)
plt.title(r'$U_0=1$')
plt.xticks(np.linspace(0, 4*np.pi, 5), [r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'])
#plt.savefig('./thesis_result_figures/last_fig/U_0=1', dpi=300, bbox_inches='tight')

#%%
interx,intery=fn.interpolate(branch,n_phi*2.)
intery=intery/1000*1.6*10**-19/(6.63*10**-34)
coeffs=fn.fourier_decompose(intery,interx,modes=10,offset=False)
print(coeffs)

Ec=512*10**6
Ej=4.7*10**9
ng=np.linspace(-2,2,9+8*60)

energies=np.array([fn.qubit_energies(Ec,Ej,coeffs,50,g)/10**6 for g in ng])
split=min(energies[:,3]-energies[:,0]-(energies[:,2]-energies[:,1]))
energy = fn.qubit_energies(Ec,Ej,coeffs,50,0.5)/10**6
split_indi = energy[3]-energy[0]-(energy[2]-energy[1])

plt.plot(ng, energies)
print(split_indi)

#%%
plt.figure(figsize=(6., 4.))
n_en,n_phi=fn.process_data(e_barrier5, np.linspace(0, 2*np.pi, 41),fill=True,extend=False,single=True)
n_en[0]=np.sort(n_en[0])
print(n_en[0])
index=int(len(n_en[0])/2)
branch,phi=fn.new_gradient_addition(n_en,n_phi)
#plt.plot(phi, branch)
phi=n_phi*2.
branch1 = np.r_[branch[:11], -np.flip(branch[:10])]
branch = np.r_[branch1, np.flip(branch1[:20])]
#plt.plot(phi, branch)
x,y=fn.arrange_E(n_en,n_phi*2.)
#x,y = fn.arrange_E(e_no_barrier, n_phi*2.)
#x, y = fn.arrange_E(e_phi_B_scan[72], n_phi*2.)
plt.scatter(x,np.array(y)/0.182,s=15)
plt.scatter(phi,branch/0.182,s=15, c='#ff7f0e')
#plt.scatter(phi,-branch/0.182,s=15, c='#ff7f0e')
plt.axhline(barrier_gap/0.182, c='r', ls='-.')
plt.axhline(-barrier_gap/0.182, c='r', ls='-.')
plt.xlabel(r'$\phi$')
plt.xlim(-0.1, 4*np.pi+0.1)
#plt.ylim(-0.2, 0.2)
#plt.ylabel(r'$\epsilon/\Delta$')
plt.yticks(np.linspace(-0.1, 0.1, 3), labels=['']*3)
plt.title(r'$U_0=5$')
plt.xticks(np.linspace(0, 4*np.pi, 5), [r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'])
#plt.savefig('./thesis_result_figures/last_fig/U_0=5', dpi=300, bbox_inches='tight')

#%%
interx,intery=fn.interpolate(branch,n_phi*2.)
intery=intery/1000*1.6*10**-19/(6.63*10**-34)
coeffs=fn.fourier_decompose(intery,interx,modes=10,offset=False)
print(coeffs)

Ec=512*10**6
Ej=4.7*10**9
ng=np.linspace(-2,2,9+8*60)

energies=np.array([fn.qubit_energies(Ec,Ej,coeffs,50,g)/10**6 for g in ng])
split=min(energies[:,3]-energies[:,0]-(energies[:,2]-energies[:,1]))
energy = fn.qubit_energies(Ec,Ej,coeffs,50,0.5)/10**6
split_indi = energy[3]-energy[0]-(energy[2]-energy[1])

plt.plot(ng, energies)
print(split_indi)

#%%
energies=np.array([fn.qubit_energies(1,0,[10., 0., 20],50,g) for g in ng])
split=min(energies[:,3]-energies[:,0]-(energies[:,2]-energies[:,1]))
plt.plot(ng, energies)

#%%

for i in np.linspace(0, 20, 21):
    plt.figure()
    energies=np.array([fn.qubit_energies(1,10,[i],50,g) for g in ng])
    split=min(energies[:,3]-energies[:,0]-(energies[:,2]-energies[:,1]))
    plt.plot(ng, energies)

#%%
Ec=512*10**6
Ej=4.7*10**9
ng=np.linspace(-2,2,9+8*60)

splitting = []
for i in np.linspace(-30e9, 30e9, 1001):
    energy = fn.qubit_energies(Ec, Ej, [1e10, 0, i],50,0.5)
    split_indi = energy[3]-energy[0]-(energy[2]-energy[1])
    splitting.append(split_indi)
    
energy_EM = fn.qubit_energies(Ec, Ej, [1e10],50,0.5)
splitting_EM = energy_EM[3]-energy_EM[0]-(energy_EM[2]-energy_EM[1])

#%%
plt.figure(figsize=(6, 3.5))

plt.plot(np.linspace(-30e9, 30e9, 1001)/1e9, np.array(splitting)/1e9, lw=2)
plt.axvline(-1, c='r', ls=':', lw=2)
plt.axvline(1, c='r', ls=':', lw=2)
plt.axhline(0, c='k', lw=0.7)
plt.xlabel(r'$E_{3e}$ (GHz)')
plt.ylabel(r'$f_q$ (GHz)')
plt.xlim(-30, 30)
plt.ylim(0)
plt.text(7, 1, "$E_M = 10$ GHz\n$E_J = 4.7$ GHz\n$E_C = 512$ MHz", size=18,
         multialignment="right")
plt.title('Effect of third harmonic', fontsize=20)
#plt.savefig('./thesis_result_figures/e3evsem', dpi=300, bbox_inches='tight', transparent=True)
#plt.text(0.95, 0.9, r'$E_M = 10$ GHz\ $E_J = 4.7$ GHz\ $E_C = 512$ MHz', size=18,
#         va="baseline", ha="right", multialignment="left",
#         bbox=dict(fc="none"))
#plt.axhline(splitting_EM/1e9, c='r', ls='-.', lw=2)


#%%
print(B_gap_no_eta_1_final[55]/0.182)



#%%
n_en,n_phi=fn.process_data(e_phi_B_scan_without_eta[55], np.linspace(0, 2*np.pi, 41),fill=False,extend=False,single=False)
n_en[0]=np.sort(n_en[0])
print(n_en[0])
index=int(len(n_en[0])/2)
branch,phi=fn.chi_gradient_addition(-3, n_en,n_phi)
branch_MBS, phi = fn.chi_gradient_addition(-2, n_en,n_phi, -2)
phi=n_phi*2.
branch = np.r_[branch[:21], np.flip(branch[:20])]
#plt.plot(phi, branch)
x,y=fn.arrange_E(n_en,n_phi*2.)
plt.scatter(x,np.array(y)/0.182,s=15)
plt.scatter(phi,branch/0.182,s=15, c='#ff7f0e')
plt.scatter(phi,-branch/0.182,s=15, c='#ff7f0e')
plt.axhline(B_gap_no_eta_1_final[55]/0.182, c='r', ls='-.', label=r'$\Delta_{min, 1}$')
plt.axhline(-B_gap_no_eta_1_final[55]/0.182, c='r', ls='-.')
plt.xlabel(r'$\phi$')
plt.xlim(-0.1, 4*np.pi+0.1)
plt.ylim(-0.4, 0.4)
plt.xticks(np.linspace(0, 4*np.pi, 5), [r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'])
plt.ylabel(r'$\epsilon/\Delta$')
plt.legend(fontsize=18, loc=1)
plt.scatter([np.pi/4.], [0.4*0.9], s=80, marker='v', c='#f2de58', zorder=100)
#plt.savefig('./thesis_result_figures/compare_figs/eta=0', dpi=300, bbox_inches='tight')
#plt.plot(phi,branch_MBS)
#plt.plot(phi,branch)

#%%
#CCCCCCCCCC

gap_C = B_gap_calc(B=0.521, eta=0.2)
#%%

n_en,n_phi=fn.process_data(phi_eta_tp_trialc, np.linspace(0, 2*np.pi, 41),fill=False,extend=False,single=False)
n_en[0]=np.sort(n_en[0])
print(n_en[0])
index=int(len(n_en[0])/2)
branch,phi=fn.chi_gradient_addition(-2, n_en,n_phi)
#plt.plot(phi, branch)
phi=n_phi*2.
branch = np.r_[branch[:21], np.flip(branch[:20])]
#plt.plot(phi, branch)
x,y=fn.arrange_E(n_en,n_phi*2.)
plt.scatter(x,np.array(y)/0.182,s=15)
#plt.scatter(phi,branch/0.182,s=15, c='#ff7f0e')
#plt.scatter(phi,-branch/0.182,s=15, c='#ff7f0e')
plt.axhline(gap_C/0.182, c='r', ls='-.', label=r'$\Delta_{min, 1}$')
plt.axhline(-gap_C/0.182, c='r', ls='-.')
plt.xlabel(r'$\phi$')
plt.xlim(-0.1, 4*np.pi+0.1)
plt.ylim(-0.2, 0.2)
plt.ylabel(r'$\epsilon/\Delta$')
plt.xticks(np.linspace(0, 4*np.pi, 5), [r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'])
plt.scatter([np.pi/4.], [0.2*0.9], s=150, marker='*', c='#b0f246', zorder=100)
#plt.ylabel(r'$\epsilon/\Delta$')
#plt.legend(fontsize=18, loc=1)
#plt.savefig('./thesis_result_figures/compare_figs/eta=0p2', dpi=300, bbox_inches='tight')

#%%
#CDDDDDDDDDDD

gap_D = B_gap_calc(B=0.579, eta=0.3)
#%%

n_en,n_phi=fn.process_data(phi_eta_tp_triald, np.linspace(0, 2*np.pi, 41),fill=True,extend=False,single=True)
n_en[0]=np.sort(n_en[0])
print(n_en[0])
index=int(len(n_en[0])/2)
branch,phi=fn.chi_gradient_addition(-2, n_en,n_phi)
#plt.plot(phi, branch)
phi=n_phi*2.
branch = np.r_[branch[:21], np.flip(branch[:20])]
#plt.plot(phi, branch)
x,y=fn.arrange_E(n_en,n_phi*2.)
plt.scatter(x,np.array(y)/0.182,s=15)
#plt.scatter(phi,branch/0.182,s=15, c='#ff7f0e')
#plt.scatter(phi,-branch/0.182,s=15, c='#ff7f0e')
plt.axhline(gap_D/0.182, c='r', ls='-.', label=r'$\Delta_{min, 1}$')
plt.axhline(-gap_D/0.182, c='r', ls='-.')
plt.xlabel(r'$\phi$')
plt.xlim(-0.1, 4*np.pi+0.1)
plt.ylim(-0.2, 0.2)
plt.xticks(np.linspace(0, 4*np.pi, 5), [r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'])
#plt.ylabel(r'$\epsilon/\Delta$')
#plt.legend(fontsize=18, loc=1)
plt.savefig('./thesis_result_figures/compare_figs/eta=0p3', dpi=300, bbox_inches='tight')

#%%
#fffff

gap_f = B_gap_calc(B=14.7*mu_B*2./2./const.e*1000./100.*72., eta=None)

#%%
ratio = 0.2/(gap_C/0.182)
print(ratio)
#%%

n_en,n_phi=fn.process_data(e_phi_B_scan[72], np.linspace(0, 2*np.pi, 41),fill=True,extend=False,single=True)
n_en[0]=np.sort(n_en[0])
print(n_en[0])
index=int(len(n_en[0])/2)
branch,phi=fn.chi_gradient_addition(-2, n_en,n_phi)
#plt.plot(phi, branch)
phi=n_phi*2.
branch = np.r_[branch[:21], np.flip(branch[:20])]
#plt.plot(phi, branch)
x,y=fn.arrange_E(n_en,n_phi*2.)
plt.scatter(x,np.array(y)/0.182,s=15)
#plt.scatter(phi,branch/0.182,s=15, c='#ff7f0e')
#plt.scatter(phi,-branch/0.182,s=15, c='#ff7f0e')
plt.axhline(gap_f/0.182, c='r', ls='-.', label=r'$\Delta_{min, 1}$')
plt.axhline(-gap_f/0.182, c='r', ls='-.')
plt.xlabel(r'$\phi$')
plt.xlim(-0.1, 4*np.pi+0.1)
plt.ylim(-ratio*gap_f/0.182, ratio*gap_f/0.182)
plt.xticks(np.linspace(0, 4*np.pi, 5), [r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'])
plt.ylabel(r'$\epsilon/\Delta$')
plt.scatter([np.pi/4.], [ratio*gap_f/0.182*0.9], s=70, marker='D', c='#ff5a8e', zorder=100)
#plt.legend(fontsize=18, loc=1)
#plt.savefig('./thesis_result_figures/compare_figs/eta=0p353', dpi=300, bbox_inches='tight')



#%%
#interf=interp1d(phi,branch,kind='cubic')
#print(phi)
#print(branch)
#interx = np.linspace(0,4*np.pi,1001)
#intery=interf(interx)
#print(intery)
#plt.plot(interx, intery)
interx,intery=fn.interpolate(branch,phi)
intery=intery/1000*1.6*10**-19/(6.63*10**-34)
coeffs=fn.fourier_decompose(intery,interx,modes=10,offset=False)
print(coeffs)
plt.plot(interx, intery)

#%%

#%%
n_en,n_phi=fn.process_data(e_phi_B_scan_without_eta[49], np.linspace(0, 4*np.pi, 41),fill=False,extend=False,single=False)
n_en[0]=np.sort(n_en[0])
print(n_en[0])
index=int(len(n_en[0])/2)
branch,phi=fn.chi_gradient_addition(-1, n_en,n_phi, second_ind_correct=-2)
plt.plot(phi, branch)
x,y=fn.arrange_E(n_en,n_phi)
plt.scatter(x,y,s=5)
plt.plot(n_phi, np.r_[branch[:21], np.flip(branch[:20])])
print(len(phi))

#%%
n_en,n_phi=fn.process_data(e_phi_B_scan_without_eta[47], np.linspace(0, 4*np.pi, 41),fill=False,extend=False,single=False)
n_en[0]=np.sort(n_en[0])
print(n_en[0])
index=int(len(n_en[0])/2)
branch,phi=fn.chi_gradient_addition(-1, n_en,n_phi, second_ind_correct=-1)
x,y=fn.arrange_E(n_en,n_phi)
plt.scatter(x,y,s=5)
plt.plot(n_phi, np.r_[branch[:21], np.flip(branch[:20])])
#%%
n_en,n_phi=fn.process_data(e_phi_B_scan_without_eta[56], np.linspace(0, 4*np.pi, 41),fill=False,extend=False,single=False)
n_en[0]=np.sort(n_en[0])
print(n_en[0])
index=int(len(n_en[0])/2)
branch,phi=fn.chi_gradient_addition(-1, n_en,n_phi, second_ind_correct=-2)
x,y=fn.arrange_E(n_en,n_phi)
plt.scatter(x,y,s=5)
plt.plot(n_phi, np.r_[branch[:21], np.flip(branch[:20])])

#%%
n_en,n_phi=fn.process_data(e_phi_B_scan_without_eta[57], np.linspace(0, 4*np.pi, 41),fill=False,extend=False,single=False)
n_en[0]=np.sort(n_en[0])
print(n_en[0])
index=int(len(n_en[0])/2)
branch,phi=fn.chi_gradient_addition(-1, n_en,n_phi, second_ind_correct=-3)
x,y=fn.arrange_E(n_en,n_phi)
plt.scatter(x,y,s=5)
plt.plot(n_phi, np.r_[branch[:21], np.flip(branch[:20])])

#%%
n_en,n_phi=fn.process_data(e_phi_B_scan_without_eta[47:60], np.linspace(0, 4*np.pi, 41),fill=True,extend=False,single=False)
#%%
n_en = 
#%%
coeff_arr = np.zeros((60-47,10))
second_coeff_array = np.r_[[-1]*2, [-2]*(57-49), -3, [-2]*2].astype(int)
for i in range(60-47):
    plt.figure()
    n_en[i][0] = np.sort(n_en[i][0])
    branch,phi = fn.chi_gradient_addition(-1, n_en[i], n_phi, second_ind_correct=second_coeff_array[i])
    print(second_coeff_array[i])
    print(branch)
    branch = np.r_[branch[:21], np.flip(branch[:20])]
    x,y=fn.arrange_E(n_en[i],n_phi)
    plt.scatter(x,y,s=5)
    if len(branch)!=41:
        print(i)
        print(branch)
        print(len(branch))
    plt.plot(n_phi, branch)
    
    
    interx,intery=fn.interpolate(branch,n_phi)
    plt.plot(interx, intery)
    plt.title('i=%i' % (47+i))
    intery=intery/1000*1.6*10**-19/(6.63*10**-34)
    coeffs=fn.fourier_decompose(intery,interx,modes=10,offset=False)
    coeff_arr[i,:]=coeffs

coeff_arr[8,:]=np.zeros(10)
coeff_arr[9,:]=np.zeros(10)
coeff_arr[12,:]=np.zeros(10)

#%%
coeff_arr = coeff_arr[~np.all(coeff_arr == 0, axis=1)]

#%%
print(B_array[np.r_[np.arange(47, 55), 57, 58]])
#%%
for i in range(10):
    plt.scatter(B_array[np.r_[np.arange(47, 55), 57, 58]], coeff_arr[:,i], label='mode %i' % (i), marker='x')
#plt.legend()

#%%
print(len(B_array[np.r_[np.arange(47, 55), 57, 58]]))

#%%
print(coeff_arr.shape)

#%%
import Functions as fn
energy_array=np.zeros((10))
E_M_reconstruction = np.zeros((10))
E_up_to_3_reconstruction = np.zeros((10))
Ec=512*10**6
Ej=4.7*10**9
ng=np.linspace(-2,2,9+8*60)

for B_ind in range(10):
    #energies=np.array([fn.qubit_energies(Ec,Ej,coeff_arr[eta_ind,B_ind,:],50,g)/10**6 for g in ng])
    #split=min(energies[:,3]-energies[:,0]-(energies[:,2]-energies[:,1]))
    energy = fn.qubit_energies(Ec,Ej,coeff_arr[B_ind,:],50,0.5)/10**6
    split = energy[3]-energy[0]-(energy[2]-energy[1])
    energy_array[B_ind]=split
    
    E_M = fn.qubit_energies(Ec,Ej,[coeff_arr[B_ind,0]],50,0.5)/10**6
    E_M_split = E_M[3]-E_M[0]-(E_M[2]-E_M[1])
    E_M_reconstruction[B_ind] = E_M_split
    
    E_3e = fn.qubit_energies(Ec,Ej,coeff_arr[B_ind,0:3],50,0.5)/10**6
    E_3e_split = E_3e[3]-E_3e[0]-(E_3e[2]-E_3e[1])
    E_up_to_3_reconstruction[B_ind] = E_3e_split

#%%
from scipy import stats
print(energy_array)
print(E_M_reconstruction)
print(np.average(energy_array))
print(stats.sem(energy_array)/1000.)
print(np.average(E_M_reconstruction))
print(stats.sem(E_M_reconstruction)/1000.)
print(np.average(energy_array-E_M_reconstruction))

    
    
#%%
pickle.dump(energy_array, open('./qubit_no_eta', 'wb'))
pickle.dump(E_M_reconstruction, open('./E_M_no_eta', 'wb'))
pickle.dump(coeff_arr, open('./fourier_no_eta', 'wb'))

#%%
plt.semilogy(B_array[np.r_[np.arange(47, 55), 57, 58]], energy_array/1000., label='full')
plt.semilogy(B_array[np.r_[np.arange(47, 55), 57, 58]], E_M_reconstruction/1000., label='E_M')
#plt.plot(B_array[np.r_[np.arange(47, 55), 57, 58]], E_up_to_3_reconstruction, label='E_3e')
for i in range(0,3,2):
    plt.semilogy(B_array[np.r_[np.arange(47, 55), 57, 58]], coeff_arr[:,i]/1e9, label='mode %i' % (i+1), marker='x')
plt.semilogy(B_array[np.r_[np.arange(47, 55), 57, 58]], [4.7]*len(B_array[np.r_[np.arange(47, 55), 57, 58]]))
plt.legend(fontsize=14)

#%%
plt.scatter(B_array[np.r_[np.arange(47, 55), 57, 58]], coeff_arr[:,0], label='mode %i' % (1), marker='x')

#%%
print(B_array[47])
print(B_array[59])
print(B_array[53])

#%%
Ec=512*10**6
Ej=4.7*10**9
ng=np.linspace(-1.6,1.6,9+8*60)
coeff_trial = np.array([Ec/5., 0., 0.])
energies=np.array([fn.qubit_energies(Ec,Ec/5.,coeff_trial,50,g)/10**9 for g in ng])
energies_no_gap = np.array([fn.qubit_energies(Ec,0.,[0.],50,g)/10**9 for g in ng])
plt.plot(ng, energies, c='k')
#plt.plot(ng, energies_no_gap, c='k', ls='-.')
plt.plot(ng, Ec*ng**2./1e9, c='b', ls='-.')
plt.plot(ng, Ec*(ng-1.)**2./1e9, c='orange', ls='-.')
plt.plot(ng, Ec*(ng+1.)**2./1e9, c='orange', ls='-.')
plt.plot(ng, Ec*(ng-2.)**2./1e9, c='b', ls='-.')
plt.plot(ng, Ec*(ng+2.)**2./1e9, c='b', ls='-.')
plt.plot(ng, Ec*(ng-3.)**2./1e9, c='orange', ls='-.')
plt.plot(ng, Ec*(ng+3.)**2./1e9, c='orange', ls='-.')
plt.xlabel(r'$n_g$ (e)')
plt.ylabel(r'$E$ (arb. unit)')
plt.ylim(-0.5, 2)

#%%
Ec=512*10**6
Ej=4.7*10**9
ng=np.linspace(-1.6,1.6,641)
coeff_trial = np.array([Ec/5., 0., 0.])
energies=np.array([fn.qubit_energies(Ec,Ec/2.,coeff_trial,50,g)/10**9 for g in ng])
energies_no_gap = np.array([fn.qubit_energies(Ec,0.,[0.],50,g)/10**9 for g in ng])

color_array1 = ['b', 'orange', 'orange', 'b']
color_array2 = ['orange', 'b', 'b', 'orange']

for i in range(4):
    plt.plot(ng[:20], energies[:20, i], c=color_array1[i])

for i in range(4):
    plt.plot(ng[20:220], energies[20:220, i], c=color_array2[i])
    
for i in range(4):
    plt.plot(ng[220:420], energies[220:420, i], c=color_array1[i])
    
for i in range(4):
    plt.plot(ng[420:620], energies[420:620, i], c=color_array2[i])

for i in range(4):
    plt.plot(ng[620:], energies[620:, i], c=color_array1[i])

plt.plot(ng, energies_no_gap, c='k', ls=':', alpha=0.5)
plt.xlabel(r'$n_g$ (e)')
plt.ylabel(r'$E$ (arb. unit)')
plt.savefig('./charge_dispersion_thesis', dpi=300, bbox_inches='tight')


#%%
print(ng[220])


#%%
np.r_[np.linspace(-1.6,-1.5, 21), np.linspace(-0.5, 0.5, 101)]
#%%
Ec=512*10**6
Ej=4.7*10**9
coeff_trial = np.array([Ec/5., 0., 0.])

ng1=np.r_[np.linspace(-1.6,-1.5, 21), np.linspace(-0.5, 0.5, 101), np.linspace(1.5, 1.6, 21)]
ng2 = np.r_[np.linspace(-1.5, -0.5, 101), np.linspace(0.5, 1.5, 101)]

energies1=np.array([fn.qubit_energies(Ec,Ec/5.,coeff_trial,50,g)/10**9 for g in ng1])
energies2=np.array([fn.qubit_energies(Ec,Ec/5.,coeff_trial,50,g)/10**9 for g in ng2])
energies_no_gap = np.array([fn.qubit_energies(Ec,0.,[0.],50,g)/10**9 for g in ng])

color_array1 = ['b', 'orange', 'orange', 'b']
color_array2 = ['orange', 'b', 'b', 'orange']

for i in range(len(energies1[0])):
    plt.plot(ng1, energies1[:,i], c=color_array1[i])

plt.xlabel(r'$n_g$ (e)')
plt.ylabel(r'$E$ (arb. unit)')

#%%
plt.figure(figsize=(3.6, 1.8))
Ec=512*10**6
Ej=4.7*10**9
ng=np.linspace(-1.5,1.5,9+8*60)
energies = np.array([fn.qubit_energies(Ec,Ej,[0.],50,g)/10**6 for g in ng])/1000.
#plt.plot(ng, energies, c='k')
plt.plot(ng, energies[:,3]-energies[:,0], label=r'$E^-$')
plt.plot(ng, energies[:,2]-energies[:,1], label=r'$E^+$')
plt.xlabel(r'$n_g$ (e)')
plt.ylabel(r'$f_q$ (GHz)')
plt.legend(fontsize=14, loc=1)
plt.savefig('./qubit_validation', dpi=300, bbox_inches='tight')

#%%
phi_plot(e_phi_B_scan_without_eta[47])
#%%
phi_plot(e_phi_B_scan_without_eta[53])
plt.ylim(-0.3, 0.3)
plt.savefig('./53 zoomed', dpi=300, bbox_inches='tight')
