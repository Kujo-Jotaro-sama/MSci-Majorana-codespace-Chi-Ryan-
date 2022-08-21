# fourier analysis
import Functions as fn
import numpy as np
import pickle
from matplotlib import pyplot as plt
#filename='./rerun_data/Data/2D_plot_vary_eta_B_no_wfn'

#phi_arr=np.linspace(0,2*np.pi,31)
#data=pickle.load(open(filename+'%s'%(0),'rb'))

Ec=512*10**6
Ej=4.7*10**9
ng=np.linspace(-2,2,9+8*60)

energy_array=np.zeros((21,49))
coeff_arr = pickle.load(open('./Data/Fourier_coefficients', 'rb'))
#coeff_arr=np.zeros((len(sim_eta_arr),49,10)) #eta, B, modes

'''for i in range(24):
    data=pickle.load(open(filename+'%s'%(i),'rb'))
    for j in range(len(data)):
        eta_ind=data[j][0]
        B_ind=data[j][1]
        en=data[j][2]

        n_en,n_phi=fn.process_data(en,phi_arr,fill=True,extend=True,single=True)
        n_en[0]=np.sort(n_en[0])
        index=int(len(n_en[0])/2)
        branch,phi=fn.new_gradient_addition(n_en,n_phi)
        
        interx,intery=fn.interpolate(branch,phi)
        intery=intery/1000*1.6*10**-19/(6.63*10**-34)
        coeffs=fn.fourier_decompose(intery,interx,modes=10,offset=False)
        coeff_arr[eta_ind,B_ind,:]=coeffs'''
        
for eta_ind in range(21):
    print(eta_ind)
    for B_ind in range(49):
        #energies=np.array([fn.qubit_energies(Ec,Ej,coeff_arr[eta_ind,B_ind,:],50,g)/10**6 for g in ng])
        #split=min(energies[:,3]-energies[:,0]-(energies[:,2]-energies[:,1]))
        energy = fn.qubit_energies(Ec,Ej,coeff_arr[eta_ind,B_ind,:],50,0.5)/10**6
        split = energy[3]-energy[0]-(energy[2]-energy[1])
        energy_array[eta_ind,B_ind]=split
        
#%%
energies=np.array([fn.qubit_energies(Ec,Ej,coeff_arr[10,10,0:3:2],50,g)/10**6 for g in ng])
plt.plot(ng, energies)
#plt.savefig('./first_third_harm', dpi=300, bbox_inches='tight')
#split=min(energies[:,3]-energies[:,0]-(energies[:,2]-energies[:,1]))

#%%
print(coeff_arr[10,10,:])
#%%
pickle.dump(energy_array, open('./Data/E_M_reconstruction', 'wb'))


#%%

plt.imshow(energy_array/1000., aspect = 'auto', extent=[0.,0.24,0.3, 0.4], origin='lower', vmin=0, vmax=11)
plt.xlabel(r'$B-B_{C}$ (meV)')
plt.ylabel(r'$\eta$ (meV)')
cbar = plt.colorbar()
cbar.ax.set_ylabel(r'$f_q$ (GHz)')
plt.axhline(np.sqrt(2.)*350/1400, c='#00ffff', lw=2)
#plt.savefig('./E_M_to_all_reconstruction', dpi=300, bbox_inches='tight')

#%%
print(np.round(E_M_flatten))
#%%
E_M_flatten = energy_array.flatten()
E_M_clean = E_M_flatten[np.round(E_M_flatten)!=0.]
E_M_avg = np.average(E_M_clean)
E_M_max = np.max(energy_array)
print(E_M_avg)
print(E_M_max)

#%%
E_qubit_flatten = energy_array.flatten()
E_qubit_clean = E_M_flatten[np.round(E_M_flatten)!=0.]
E_qubit_avg = np.average(E_M_clean)
E_qubit_max = np.max(energy_array)
print(E_qubit_avg)
print(E_qubit_max)

#%%
print(E_M_max-E_qubit_max)

#%%
print(np.max(energy_array))
print(np.arange(0, 11)[0:11:2])
#%%
print(np.max(coeff_arr[:,:,0])/1e9)
print(np.max(coeff_arr[:,:,1])/1e9)
print(np.max(coeff_arr[:,:,2])/1e9)
print(np.max(coeff_arr[:,:,3])/1e9)
print(np.max(coeff_arr[:,:,4])/1e9)



#%%
print(coeff_arr[:,:,1])
                    

