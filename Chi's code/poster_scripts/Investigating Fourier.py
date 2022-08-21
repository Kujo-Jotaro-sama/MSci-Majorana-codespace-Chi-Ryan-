# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 22:58:35 2022

@author: hht18
"""
import Functions as f
import numpy as np
import matplotlib.pyplot as plt
import pickle
#%% Try to recreate Metzger paper
#The Hamiltonian function should be working now!!
#Last revised: 12/1/2022

Ec=512*10**6
Ej=4.7*10**9
#Em=np.array([0])
#Em=np.array([2.4*10**6])
#Em=np.array([2.4*10**10]) actual gap magnitude scale, 0.1mev
Em=[5.12*10**7] #0.1 Ec


levels=50
ng=np.linspace(-2,2,9+8*60)
energies=np.array([f.qubit_energies(Ec,Ej,Em,levels,g)/10**6 for g in ng])

plt.figure()
plt.xlabel(r'$n_g$')
plt.ylabel(r'$f(MHz)$')
plt.plot(ng,energies[:,3]-energies[:,0],label='$E_{+}$')
plt.plot(ng,energies[:,2]-energies[:,1],label='$E_{-}$')

plt.xlim(-1.5,1.5)
plt.legend()

plt.title(r"Qubit splittings, $E_{C}=%s MHz, E_{J}=%s GHz,E_{M}=%s MHz$"%(Ec/10**6,Ej/10**9,round(Em[0]/10**6,3)))
print('Ratio Ec/Ej',round(Ec/Ej,3),'Ratio Em/Ej',round(Em[0]/Ej,3))
plt.show()
plt.figure()
plt.xlabel(r'$n_g$')
plt.ylabel(r'$f(MHz)$')
for i in range(4):
    plt.plot(ng,energies[:,i])

plt.xlim(-1.5,1.5)
plt.title(r"Eigenenergies, $E_{C}=%s MHz, E_{J}=%s GHz,E_{M}=%s MHz$"%(Ec/10**6,Ej/10**9,round(Em[0]/10**6,3)))

#%% Fourier on past data
# Hopefully the final time we're doing this...
#Last modified 12/1/2022
filename='Flat_band_highres'
arr,waves,etas=pickle.load(open(filename,'rb'))

ind=7 #graph 7 onwards is where the majorana solution appears
phi_arr=np.linspace(0,2*np.pi,31)
ng=np.linspace(-2,2,9+8*60)
levels=50
new_data,new_phi=f.process_data(arr,phi_arr)

#%%
arr = pickle.load(open('../subband/Data/eta=0 no barrier','rb'))
#print(arr)
phi_arr=np.linspace(0,4*np.pi,41)
ng=np.linspace(-2,2,9+8*60)
levels=50
#new_data,new_phi=f.process_data(arr,phi_arr)
#%%
coeff_arr=[]
Ec=512*10**6
Ej=4.7*10**9
Em=[5.12*10**7] #0.1 Ec
index=int(len(arr[0])/2)
branch,phi=f.gradient_addition(index,arr,phi_arr,printing=False)
interx,intery=f.interpolate(branch,phi)
plt.plot(interx, intery)
#%%
intery=intery/1000*1.6*10**-19/(6.63*10**-34)
coeffs=f.fourier_decompose(intery,interx,modes=10,offset=False)
coeff_arr.append(coeffs)
energies=np.array([f.qubit_energies(Ec,Ej,coeffs,levels,g)/10**6 for g in ng])
print(np.min(energies[:,3]-energies[:,0])-np.max(energies[:,2]-energies[:,1]))
#%%
plt.figure(figsize=(6.4, 2.0))
plt.xticks(ticks=[-1, 0, 1], labels=[])
#plt.xlabel(r'$n_g$')
plt.ylabel(r'$f(MHz)$')
plt.plot(ng,energies[:,3]-energies[:,0],label=r'$E^{-}$')
plt.plot(ng,energies[:,2]-energies[:,1],label=r'$E^{+}$')

plt.xlim(-1.5,1.5)
plt.ylim(0,17000)
plt.legend(fontsize=18, loc=1)

plt.title(r"$\eta=0$")
#plt.savefig('../poster_replots/fig61', dpi=300, bbox_inches='tight')

#%% Plotting qubit spectra
coeff_arr=[]
Ec=512*10**6
Ej=4.7*10**9
Em=[5.12*10**7] #0.1 Ec
for i in range(12,13):
    print('i=',i)
    index=int(len(arr[i][0])/2)
    branch,phi=f.gradient_addition(index,new_data[i],new_phi,printing=False)
    interx,intery=f.interpolate(branch,phi)
    intery=intery/1000*1.6*10**-19/(6.63*10**-34)
    coeffs=f.fourier_decompose(intery,interx,modes=10,offset=False)
    coeff_arr.append(coeffs)
    energies=np.array([f.qubit_energies(Ec,Ej,coeffs,levels,g)/10**6 for g in ng])
    print(np.min(energies[:,3]-energies[:,0])-np.max(energies[:,2]-energies[:,1]))
#%%
    plt.figure(figsize=(6.4, 2.0))
    plt.xlabel(r'$n_g$')
    plt.ylabel(r'$f(MHz)$')
    plt.plot(ng,energies[:,3]-energies[:,0],label=r'$E^{-}$')
    plt.plot(ng,energies[:,2]-energies[:,1],label=r'$E^{+}$')

    plt.xlim(-1.5,1.5)
    plt.ylim(0,13500)
    plt.legend(fontsize=18, loc=1)

    plt.title(r"$\eta=0.36$")
    #plt.savefig('../poster_replots/fig62', dpi=300, bbox_inches='tight')
    
    #print('Ratio Ec/Ej',round(Ec/Ej,3),'Ratio Em/Ej',round(Em[0]/Ej,3))
    #plt.show()
    '''plt.figure()
    plt.xlabel(r'$n_g$')
    plt.ylabel(r'$f(MHz)$')
    plt.title(r"Eigenenergies, $E_{C}=%s MHz, E_{J}=%s GHz,\eta=%s$"%(Ec/10**6,Ej/10**9,round(etas[i],3)))
    for i in range(4):
        plt.plot(ng,energies[:,i])
    plt.ylim(-5000,9000)
    plt.xlim(-1.5,1.5)'''
    #print(len(coeff_arr))
    
#%%
#effect of 3e transfer
coeffs = np.array([1e9, 0., -1e9, 0., 0., 0., 0., 0., 0., 0.])
energies=np.array([f.qubit_energies(Ec,0.,coeffs,levels,g, eigvals=10)/10**6 for g in ng])
print(energies)
#%%
print(energies.shape)
#%%
plt.plot(ng, energies)
#%%
#plt.figure(figsize=(6.4, 2.0))
plt.xlabel(r'$n_g$')
plt.ylabel(r'$f(MHz)$')
plt.plot(ng,energies[:,3]-energies[:,0],label=r'$E^{-}$')
plt.plot(ng,energies[:,2]-energies[:,1],label=r'$E^{+}$')
#%%explicitly plotting Fourier coefficients
eta_use=etas[ind:len(etas)-1]
print(len(eta_use))
print(len(coeff_arr))
#%% 14 Jan. Lump with 12 Jan stuff. End of Fourier.
plt.figure()
coeff_arr=np.array(coeff_arr)
#%%
plt.rcParams.update({'font.size': 22})
#%%
label_list = [r'$4\pi$ mode', r'$2\pi$ mode', r'$4\pi/3$ mode']
for i in range(3): #all fourier modes plotted
    plt.plot(eta_use,coeff_arr[:,i]/1e9,'x',label=label_list[i], ms=7, mew=2)
    plt.xlabel(r'$\eta$ (meV)')
    plt.ylabel('Amplitude (GHz)')
plt.legend(fontsize=18)
plt.xlim(eta_use[0], eta_use[-1])
#plt.title('Subband mixing, all modes')
plt.savefig('../poster_replots/fig5', dpi=300, bbox_inches='tight')
#%%
plt.figure()

for i in range(1,10,2): #only odd modes
    plt.scatter(eta_use,coeff_arr[:,i],'x',label=label_list[i])
    plt.xlabel(r'$\eta$')
    plt.ylabel('Coefficients')
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.title('Subband mixing, even modes')
plt.show()

plt.figure()
for i in range(0,10,2): #only even modes
    plt.plot(eta_use,coeff_arr[:,i],'x',label=r'Mode %s'%(i+1))
    plt.xlabel(r'$\eta$')
    plt.ylabel('Coefficients')
plt.legend()
plt.title('Subband mixing, odd modes')
#plt.show()
#%% Reconstruction of graphs using only odd modes
for i in range(ind,len(etas)-1):
    print('i=',i)
    index=int(len(arr[i][0])/2)
    branch,phi=f.gradient_addition(index,new_data[i],new_phi,printing=False)
    interx,intery=f.interpolate(branch,phi)
    intery=intery/1000*1.6*10**-19/(6.63*10**-34)
    odd_modes=f.fourier_decompose(intery,interx,modes=10,offset=False)[::2]

    plt.figure()# using only odd modes
    recon=[sum([odd_modes[j]*np.cos(x/2*(2*j+1)) for j in range(len(odd_modes))]) for x in interx]
    recon2=[sum([odd_modes[j]*np.cos(x/2*(2*j+1)) for j in range(2)]) for x in interx]
    plt.plot(interx,intery,'.',label='Original',markersize=3)
    plt.plot(interx,recon,'--',label='Odd reconstruction, 5 modes',markersize=1)
    plt.plot(interx,recon2,'--',label='Odd reconstruction, 2 modes',markersize=1)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.title(r'$\eta=%s$ Majorana solution E-$\phi$'%(round(etas[i],3)))
    plt.xlabel('$\phi$')
    plt.ylabel('E/$\Delta$')
    plt.xticks(np.linspace(0,4*np.pi,9,endpoint=True), ['0',r'$\frac{\pi}{2}$',r'$\pi$',
                                                     r'$\frac{3\pi}{2}$',r'$2\pi$',
                                                     r'$\frac{5\pi}{2}$',r'$3\pi$',
                                                     r'$\frac{7\pi}{2}$',r'$4\pi$'])
    plt.ylim(-4e9,4e9)
    
