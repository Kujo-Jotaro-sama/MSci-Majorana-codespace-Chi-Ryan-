#only testing half flux because HPC takes fucking long
import peierls
import numpy as np
import kwant
import kwant.continuum
import pickle
import sympy
import BoundState as bs
from sympy.physics.matrices import msigma, Matrix
from sympy import eye
from sympy.physics.quantum import TensorProduct
import os
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

def get_shape(L, W, T):
    L_start, W_start, T_start = 0, 0, 0
    L_stop, W_stop, T_stop = L, W, T

    def shape(site):
        (x, y, z) = site.pos
        return (W_start <= y <= W_stop and
                T_start <= z <= T_stop and
                L_start <= x <= L_stop)

    return shape, np.array([L_stop, W_stop, T_stop])

L_x = 100
W_y = 100
T_z = 100

syst = kwant.Builder()
_ = syst.fill(temp_syst, *get_shape(L_x, W_y, T_z))
lat = kwant.lattice.cubic(a, norbs=8)

lead_L = kwant.Builder(kwant.TranslationalSymmetry((-a, 0, 0)))
lead_R = kwant.Builder(kwant.TranslationalSymmetry((a, 0, 0)))

lead_L.fill(temp_lead_L, *get_shape(L_x, W_y, T_z))
lead_R.fill(temp_lead_R, *get_shape(L_x, W_y, T_z))
syst.attach_lead(lead_L)
syst.attach_lead(lead_R)

fsyst = syst.finalized()

f = dict(
    re=lambda x: x.real,
    im=lambda x: x.imag,
    phi_0=1.0, # units with flux quantum equal to 1
    exp=np.exp
)
params_3DTI = dict(
    A_perp=3,
    A_z=3,
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
params_bands = dict(
    **f,
    **params_3DTI,
    **params_JJ
)

mu = 0.02
Delta = 0.001

flux=0.5
phi_arr=np.linspace(0,2*np.pi,31)
arr_ind=int(os.environ['PBS_ARRAY_INDEX'])
jobs=16
perjob=2
filename='3dti_energy_phase_230122_halfflux'
fluxarr=[]
fluxwaves=[]
counter=0
for i in range(len(phi_arr)):
    counter+=1
    if (counter <= (arr_ind+1)*perjob) and (counter>(arr_ind*perjob)):
        params_bands.update(
            mu = mu,
            B_x = flux/W_y/T_z,
            Deltaf_L = lambda y, z: Delta if z==T_z else 0,
            Deltaf_R = lambda y, z: Delta*np.exp(1j*phi_arr[i]) if z==T_z else 0)
        energy,waves=bs.find_boundstates(fsyst,-Delta,Delta,rtol=Delta/30,params=params_bands)
        fluxarr.append(energy)
        fluxwaves.append(waves)
        combine=[i,fluxarr,fluxwaves]
        pickle.dump(combine,open(filename+'%s'%(i),'wb'))
    
