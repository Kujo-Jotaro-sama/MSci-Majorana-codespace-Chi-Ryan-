# -*- coding: utf-8 -*-
"""
A script to test out the boundstate algorithm, and figure out where there are
inconsistencies
"""
import numpy as np
from types import SimpleNamespace as sns
import Functions as f
import pickle
import matplotlib.pyplot as plt
import BoundState as bs
#%% first, conduct B scan to check the bound state energies
# and where the algorithm produces discontinuities
B_arr=np.linspace(0,1,50)
p=sns(a=1,B=1.,alpha=0.5,mu=0.5,gap=0.5,t=-1)
E_arr=f.B_sweep(p,B_arr,max_E=1,V_n=1.25,L=1,phi=0,rtol=1e-2, system=3)
data=[]

data.append(B_arr)
data.append(E_arr)
data.append(p)
filename='B_sweep'
pickle.dump(data,open(filename,'wb'))
_,_,fig=f.B_sweep_plot(B_arr,E_arr)
#very different from what chi gets. will try with his system
#%%
p=sns(a=1,B=1.,alpha=0.5,mu=0.5,gap=0.5,t=-1)
B_arr=np.linspace(0,1,50)
p=sns(a=1,B=1.,alpha=0.5,mu=0.5,gap=0.5,t=-1)
E_arr=f.B_sweep(p,B_arr,max_E=1,V_n=1.25,L=1,phi=0,rtol=1e-2, system=2)
data=[]

data.append(B_arr)
data.append(E_arr)
data.append(p)
filename='B_sweep_chi_system'
pickle.dump(data,open(filename,'wb'))
_,_,fig=f.B_sweep_plot(B_arr,E_arr)
print(E_arr)
#dont get chi's results...?
#%% #check t=0.7 using Chi's system. Do i get his results for t?
t=-1
trivial=sns(a=1,B=0.4,alpha=0.5,mu=0.5,gap=0.5,t=t)
topcond=sns(a=1,B=1.,alpha=0.5,mu=0.5,gap=0.5,t=t)
V=[1.25,2.5,5]
V_n_array,phi_array=f.phase_sweep(topcond,datapoints=15,max_E=1,
                                  V_n_arr=V,L=1,system=2)

#V_n_array2,phi_array2=f.phase_sweep(trivial,datapoints=2,max_E=1,
#                                  V_n_arr=V,L=1,system=2)
f.phase_sweep_plot(V_n_array,phi_array,V,save=False)
#f.phase_sweep_plot(V_n_array2,phi_array2,V,save=False)
# gives same thing as my own code...
#%% Using chi's explicit code
t=0.7
trivial=sns(a=1,B=0.4,alpha=0.5,mu=0.5,gap=0.5,t=t)
topcond=sns(a=1,B=1.,alpha=0.5,mu=0.5,gap=0.5,t=t)
V=[1.25,2.5,5]
V_n_array,phi_array=f.phase_sweep(topcond,datapoints=15,max_E=1,
                                  V_n_arr=V,L=1,system=4)
V_n_array2,phi_array2=f.phase_sweep(trivial,datapoints=15,max_E=1,
                                  V_n_arr=V,L=1,system=4)
f.phase_sweep_plot(V_n_array,phi_array,V,save=False)
f.phase_sweep_plot(V_n_array2,phi_array2,V,save=False)
#gives chi's results
#%% Checking difference between lat neighbour and explicit
topcond=sns(a=1,B=1.,alpha=0.5,mu=0.5,gap=0.5,t=-1)
JJ1=f.make_system_3(1,topcond)
JJ2=f.make_system_ex(topcond)
print('--------')
print('left after finalised')
print(JJ1.leads[0].hamiltonian(0,1)==JJ2.leads[0].hamiltonian(0,1))
print('right after finalised')
print(JJ1.leads[1].hamiltonian(0,1)==JJ2.leads[1].hamiltonian(0,1))
print('--------')
print('SN interface')
print(JJ1.hamiltonian(0,1)==JJ2.hamiltonian(0,1))
print(JJ2.hamiltonian(1,2)==JJ1.leads[1].hamiltonian(0,1))
#%%

print('left, finalised, explicit')
print(JJ1.hamiltonian(0,1))
#%%
print('Checking On-site Hamiltonians...')
for i in range(3): #check onsite hamiltonians
    if np.all(JJ1.hamiltonian(i,i)!=JJ2.hamiltonian(i,i)):
        print('False! At site',i)
for i in range(2): #check hopping matrices
    print(JJ1.hamiltonian(i,i+1)==JJ2.hamiltonian(i,i+1))
#%% neighbors sets hopping from right to left, so Chi and
#my systems weren't the same. 
#try another sweep, check for discontinuity

B_arr=np.linspace(0,1,50)
p=sns(a=1,B=1.,alpha=0.5,mu=0.5,gap=0.5,t=-1)
E_arr=f.B_sweep(p,B_arr,max_E=1,V_n=1.25,L=1,phi=0,rtol=1e-2,
                system=5)
data=[]

data.append(B_arr)
data.append(E_arr)
data.append(p)
filename='B_sweep_reversed_hopping'
pickle.dump(data,open(filename,'wb'))
_,_,fig=f.B_sweep_plot(B_arr,E_arr)
#still dont get any bound states below B=0.7
#%% B sweep using Chi's system, made using my code
#explicitly defining site2-3 as V_N
B_arr=np.linspace(0,1,50)
p=sns(a=1,B=1.,alpha=0.5,mu=0.5,gap=0.5,t=-1)
E_arr=f.B_sweep(p,B_arr,max_E=1,V_n=1.25,L=1,phi=0,rtol=1e-2,
                system=2)
data=[]

data.append(B_arr)
data.append(E_arr)
data.append(p)
filename='B_sweep_chi_mycode'
pickle.dump(data,open(filename,'wb'))
_,_,fig=f.B_sweep_plot(B_arr,E_arr)
#managed to recreate chi's plot. but is it right??


#%% t sweep with the hopping done properly. 2 site system
trivial=sns(a=1,B=0,alpha=0.5,mu=0.5,gap=0.5,t=-1)
topcond=sns(a=1,B=1.,alpha=0.5,mu=0.5,gap=0.5,t=-1)
t_arr=np.linspace(-1,1,30)
t_sweep_E_topo=f.t_sweep(topcond,t_arr,max_E=1,V_n=1.25,L=1,phi=0,rtol=1e-2,
                         system=5)
t_sweep_E_triv=f.t_sweep(trivial,t_arr,max_E=1,V_n=1.25,L=1,phi=0,rtol=1e-2,
                         system=5)
abc=[topcond,t_arr,t_sweep_E_topo]
efg=[trivial,t_arr,t_sweep_E_triv]
filename=r't_sweep_data_trivial_reversed_hopping'
filename2=r't_sweep_data_topo_reversed_hopping'
pickle.dump(abc,open(filename2,'wb'))
pickle.dump(efg,open(filename,'wb'))
#%%
triv_namespace,t_arr_triv,t_sweep_E_triv=pickle.load(open(
        't_sweep_data_trivial_reversed_hopping','rb'))
topo_namespace,t_arr_topo,t_sweep_E_topo=pickle.load(open(
        't_sweep_data_topo_reversed_hopping','rb'))

x,y=f.t_sweep_plot(t_arr_topo,t_sweep_E_topo,save=False)
fig=plt.figure() #want to vary xlim,ylim myself
plt.scatter(x,y)
plt.xlabel('t',fontsize=14)
plt.ylabel('E',fontsize=14)
plt.show()
fig.savefig('t_sweep_plot_V125_topo_full_reversed_hopping')

x,y=f.t_sweep_plot(t_arr_triv,t_sweep_E_triv,save=False)
fig=plt.figure() #want to vary xlim,ylim myself
plt.scatter(x,y)
plt.xlabel('t',fontsize=14)
plt.ylabel('E',fontsize=14)
plt.show()
fig.savefig('t_sweep_plot_V125_triv_full_reversed_hopping')

#%% t sweep with the hopping done properly. 3 site system (Chi's implementation)
trivial=sns(a=1,B=0,alpha=0.5,mu=0.5,gap=0.5,t=-1)
topcond=sns(a=1,B=1.,alpha=0.5,mu=0.5,gap=0.5,t=-1)
t_arr=np.linspace(-1,1,30)
t_sweep_E_topo=f.t_sweep(topcond,t_arr,max_E=1,V_n=1.25,L=1,phi=0,rtol=1e-2,
                         system=6)
t_sweep_E_triv=f.t_sweep(trivial,t_arr,max_E=1,V_n=1.25,L=1,phi=0,rtol=1e-2,
                         system=6)
abc=[topcond,t_arr,t_sweep_E_topo]
efg=[trivial,t_arr,t_sweep_E_triv]
filename=r't_sweep_data_trivial_reversed_hopping_chi'
filename2=r't_sweep_data_topo_reversed_hopping_chi'
pickle.dump(abc,open(filename2,'wb'))
pickle.dump(efg,open(filename,'wb'))
#%%
triv_namespace,t_arr_triv,t_sweep_E_triv=pickle.load(open(
        't_sweep_data_trivial_reversed_hopping_chi','rb'))
topo_namespace,t_arr_topo,t_sweep_E_topo=pickle.load(open(
        't_sweep_data_topo_reversed_hopping_chi','rb'))

x,y=f.t_sweep_plot(t_arr_topo,t_sweep_E_topo,save=False)
fig=plt.figure() #want to vary xlim,ylim myself
plt.scatter(x,y)
plt.xlabel('t',fontsize=14)
plt.ylabel('E',fontsize=14)
plt.show()
fig.savefig('t_sweep_plot_V125_topo_full_reversed_hopping_chi')

x,y=f.t_sweep_plot(t_arr_triv,t_sweep_E_triv,save=False)
fig=plt.figure() #want to vary xlim,ylim myself
plt.scatter(x,y)
plt.xlabel('t',fontsize=14)
plt.ylabel('E',fontsize=14)
plt.show()
fig.savefig('t_sweep_plot_V125_triv_full_reversed_hopping_chi')
#%%
def plot_eigenvalues(p,E_min=0,E_max=0.5,points=100,system=2):
    '''
    Plots the explicit function used in the boundstate algorithm
    that is being solved to find the roots of.
    '''
    B=p.B
    mu=p.mu
    alpha=p.alpha
    gap=p.gap
    t=p.t
    if system==3: #hopping implementation
        JJ=f.make_system_3(1,p)
    if system==2: #hopping implementation
        JJ=f.make_system_2(1,p)
    if system==1: #phase implementation
        JJ=f.make_system_1(1,p)
    if system==5: #phase implementation
        JJ=f.make_system_5(1,p)
    if system==4:
        JJ=f.make_system_ex(p,L=3)
    if system==6:
        JJ=f.make_system_ex2(p,L=3)
    energy=np.linspace(-0.5,0.5,100)
    solns=[]
    for candidate in energy:
        solns.append(bs.min_eigenvalue(candidate, JJ))
    plt.figure()
    plt.plot(energy,solns)
    plt.ylabel('f(E)')
    plt.xlabel('E')
    plt.title(r'B=%s,$\alpha_{so}$=%s,$\mu$=%s,$\Delta$=%s,t=%s'%(
        B,alpha,mu,gap,t))
    plt.axhline(linestyle='--')
#%% Make a slider of the crossings at different B
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
import matplotlib
#p=sns(a=1,B=0.35,alpha=0.5,mu=0.5,gap=0.5,t=-1)
#plot_eigenvalues(p,-0.5,0.5)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.5)
p=sns(a=1,B=0.4,alpha=0.5,mu=0.5,gap=0.5,t=0.7)
B=p.B
mu=p.mu
alpha=p.alpha
gap=p.gap
t=p.t
JJ=f.make_system_3(1,p, phi=0.5*np.pi)
E_arr=np.linspace(-0.5,0.5,200)
f_E = [bs.min_eigenvalue(candidate, JJ) for candidate in E_arr]
colorarr=[x for x in matplotlib.colors.ColorConverter.colors.keys() if len(x)==1]
print(colorarr)
#i=0
#ax.plot(E_arr[0], f_E[0], '.',markersize=4,lw=2, color=colorarr[i%2])
curr_prop_fns=JJ.leads[0].modes(0)[1].nmodes
for E in range(0,len(E_arr)):
    new_prop_fns=JJ.leads[0].modes(E_arr[E])[1].nmodes
    #if np.any(new_prop_fns!=curr_prop_fns):
        #i+=1
    #    curr_prop_fns=new_prop_fns
    ax.plot(E_arr[E], f_E[E], '.',markersize=4,lw=2, color=colorarr[new_prop_fns], label='nmodes = %i' % (new_prop_fns))
ax.axhline(linestyle='--')
ax.title.set_text(r'B=%s,$\alpha_{so}$=%s,$\mu$=%s,$\Delta$=%s,t=%s'%(
        B,alpha,mu,gap,t))

axcolor = 'lightgoldenrodyellow'
axB = plt.axes([0.25, 0.35, 0.65, 0.03], facecolor=axcolor)
axt = plt.axes([0.25, 0.3, 0.65, 0.03], facecolor=axcolor)
axalpha = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
axmu = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
axgap = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
axphi=plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axbutton=plt.axes([0.4,0.02,0.35,0.07])
plot_bs=Button(axbutton,'Identify Boundstates')

sB=Slider(axB,'B',0,1,valinit=0.4)
st = Slider(axt, 't', -1, 1, valinit=0.7)
salpha= Slider(axalpha, r'$\alpha$', 0, 1, valinit=0.5)
smu= Slider(axmu, '$\mu$', 0, 1, valinit=0.5)
sgap= Slider(axgap, '$\Delta$', 0, 1, valinit=0.5)
sphi= Slider(axphi, '$\phi$', 0, 2*np.pi, valinit=0.5*np.pi)

def update(val):
    B = sB.val    # call a transform on the slider value
    t=st.val
    alpha=salpha.val
    gap=sgap.val
    mu=smu.val
    phi=sphi.val
    name=sns(a=1,B=B,alpha=alpha,mu=mu,gap=gap,t=t)
    JJ=f.make_system_3(1,name,phi=phi,V_n=1.25)
    E_arr=np.linspace(-0.5,0.5,200)
    ax.cla()
    #plt.clf()
    f_E = [bs.min_eigenvalue(candidate, JJ) for candidate in E_arr]
    colorarr=[x for x in matplotlib.colors.ColorConverter.colors.keys() if len(x)==1]
    curr_prop=JJ.leads[0].modes(0)[1].nmodes
    #i=0
    #ax.plot(E_arr[0], f_E[0], '.',markersize=4,lw=2, color=colorarr[i%2])
    for E in range(0,len(E_arr)):
        new_prop=JJ.leads[0].modes(E_arr[E])[1].nmodes
        print(new_prop)
        #if np.any(new_prop!=curr_prop):
         #   i+=1
           # curr_prop=new_prop
        ax.plot(E_arr[E], f_E[E], '.',markersize=4,lw=2, color=colorarr[new_prop])
    ax.axhline(linestyle='--')
    ax.title.set_text(r'B=%s,$\alpha_{so}$=%s,$\mu$=%s,$\Delta$=%s,t=%s'%(
        B,alpha,mu,gap,t))
    #ax.axis([-3, 3, -6, 6])
    fig.canvas.draw_idle()

def add_real_solns(val):
    B = sB.val    # call a transform on the slider value
    t=st.val
    alpha=salpha.val
    gap=sgap.val
    mu=smu.val
    phi=sphi.val
    name=sns(a=1,B=B,alpha=alpha,mu=mu,gap=gap,t=t)
    JJ=f.make_system_3(1,name,phi=phi,V_n=1.25)
    realE,_=bs.find_boundstates(JJ, -0.5, 0.5)
    zeros=np.zeros(realE.shape)
    ax.plot(realE,zeros,'x')
sphi.on_changed(update)
sB.on_changed(update)
st.on_changed(update)
salpha.on_changed(update)
smu.on_changed(update)
sgap.on_changed(update)
plot_bs.on_clicked(add_real_solns)

plt.show()
#%% Trying to figure out why  some are OK and some are not
#Discontinuities as a result of opening of lead channels
for E in np.linspace(0,1,50):
    p=sns(a=1,B=1,alpha=0.5,mu=0.5,gap=0.5,t=-1)
    JJ=f.make_system_2(1, p)
    prop,stab=JJ.leads[0].modes(E) #propagating and stabilised modes
#stab is the basis of evanescent and propagating modes
#prop are the propagating solutions
    print(prop.momenta)
#%%
p=sns(a=1,B=B,alpha=0.5,mu=0.5,gap=0.5,t=-1)
JJ=f.make_system_2(3, p)
bs.make_linsys_check(JJ,1)
#%%
a=[[1,2],[3,4]]
print(np.all(a!=a))