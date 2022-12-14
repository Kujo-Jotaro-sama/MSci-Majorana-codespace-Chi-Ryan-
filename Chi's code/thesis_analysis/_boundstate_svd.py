import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar
import scipy.sparse as sp
from _boundstate_common import *

def BS_finder(H_s, H_leads, V_leads, transf, e_range, sparse=False,
              eps=1e-6, sigma=None, tol=1e-8, sing_values=4, uv=False):
    """
    Finds the minimum of the singular values of L in e_range.
    If the singular value is smaller than eps, then it returns 
    the energy of the bound state.
    Parameters:
    e_range: tuple countaining the minimum and the maximum of the 
             energy range
    
    returns:
    E: energy of the bound state if minimal singular value is smaller
        than eps
    None otherwise
    eps: float, if the minimal singular value found by minimize_scalar is
        smalle than eps, then it consider you found a bound state
    tol: float, function minimize_scalar stops only if the minimum of the 
        previous step is closer than tol (or something like that). Should
        be smaller than eps
    
    Comment about the parameters tol and eps:
    At which scale do we consider a minimum as a bound state and not 
    as a singular value close to zero?
    It would be good to compare the zero found in the minimizer with 
    quantities coming from the system itself (the size of the gap? but 
    not all systems are gapped, the width of the band structure? the 
    biggest singular value?).
    """

    def dumb(e):
        s = min(singular_values(H_s, H_leads, V_leads, transf, e, 
                                sparse=sparse, uv=False, sigma=sigma,
                                sing_values=sing_values))
        return s

    minimum = minimize_scalar(dumb, method='Bounded', bounds=e_range, options={'xatol': tol})
    if minimum.fun < eps:
        # if the last singular value is smaller than eps,
        # then there is a bound state.
        E = minimum.x
        if not uv:
            return E
        else:
            S, Wh, L_out, X_out = singular_values(H_s, H_leads, V_leads,
                                                  transf, E, sparse=sparse,
                                                  uv=True, sigma=sigma,
                                                  sing_values=sing_values)
            return E, S, Wh, L_out, X_out
    else:
        return None

def BS_solver(H_s, H_leads, V_leads, transf, E, eps=1e-4, sparse=False, sigma=None):
    """
    Returns the elements to compute the wavefunction of a bound state
    Raise an error if there is no bound state (singular value > eps)
    Parameters:
    transf: numpy array or sparse csr matrix. V.dot(transf) gives the 
    hopping matrix between the scattering region and the leads.
    """

    S, Wh, L_leads, X_out_leads = singular_values(H_s, H_leads, V_leads, transf, E,
                                      sparse=sparse, uv=True, sigma=sigma)
    assert(min(S) < eps), 'The energy does not match the bound state energy, {0}'.format(min(S)) 
    N = H_s.shape[0]
    W = Wh.conj().T
    return compute_wf(S, W, L_leads, X_out_leads, N, eps=eps)

def hopping_svd(V_leads, eps=1e-4):
    """
    Check if one or several hopping matrices in the lead are not 
    invertible and replace them by their SVD in the former case.
    """
    DBh_leads = []
    for V in V_leads:
        _, D, Bh = la.svd(V)
        # keep only the vectors assiocated with non zero singular value
        non_zeros = sum(D > eps)
        D = np.diag(D[:non_zeros])
        Bh = Bh[:non_zeros, :]
        DBh = np.dot(D, Bh)
        DBh_leads.append(DBh)

    return DBh_leads

def singular_values(H_s, H_leads, V_leads, transf, E, sparse=False,
                    uv=True, sing_values=4, eps=1e-4, sigma=None,
                    need_orth=True):
    """
    Solver that returns the singular values of the SVD of 
    the left hand side of the bound state equation (L in the notes)
    
    Parameters:
    H_s: Hamiltonian of the central system
    H_leads: tuple, countains the onsite matrices of the unit cells of the leads
    V_leads: tuple, countains the hopping matrices between the unit 
        cells of every lead
    uv: Whether to compute the the full SVD of the matrix L 
        or only the singular values
    returns:
    if uv is false: the singular values of the matrix L
    else: 
    S: The singular values of L 
    Wh: The right matrix of the svd decomposition
    L: array that countains the evanescent outgoing lambdas
    X_out: Columns are the corresponding evanescent wavefunctions
    """
    L_out_leads, X_out_leads = leads_modes(H_leads, V_leads, E,
                                           need_orth=need_orth)

    lhs = setup_lhs(H_s, V_leads, transf, L_out_leads, X_out_leads, E, 
                    sparse=sparse)

    if sparse:
        B = sp.coo_matrix.dot(lhs.conj().T, lhs)
        if sigma is None:
            # First compute the biggest eigenvalue
            max_eval = sp.linalg.eigsh(B, k=1, which='LM', return_eigenvectors=False)[0]
            # shift the matrix so that the lowest eigenvalue become the biggest in magnitude
            B = -B + 2 * max_eval * sp.eye(B.shape[0])
        if not uv:
            try: # if the singular value is exactly zero
                evals = sp.linalg.eigsh(B, return_eigenvectors=False, which='LM',
                                        k=sing_values, sigma=sigma)
            except RuntimeError:
                evals = sp.linalg.eigsh(B + eps*sp.eye(B.shape[0]), 
                                        which='LM', 
                                        return_eigenvectors=False, 
                                        k=sing_values, sigma=sigma)
                evals -= eps
            if sigma is None: # that is, use the shift method 
                return np.sqrt(2 * max_eval - evals)
            else: # use of the shift invert method
                return np.sqrt(evals)
        else:
            try:
                evals, evecs = sp.linalg.eigsh(B, return_eigenvectors=True, 
                                                k=sing_values, sigma=sigma,
                                                which='LM')
            except RuntimeError:
                evals, evecs = sp.linalg.eigsh(B + eps*sp.eye(B.shape[0]), 
                                                which='LM', 
                                                return_eigenvectors=True, 
                                                k=sing_values, sigma=sigma)
                evals -= eps
            if sigma is None: # shift method
                return np.sqrt(2 * max_eval - evals), evecs.T, L_out_leads, X_out_leads
            else: # shift invert method
                # If an eigenvalue is negative due to numerical 
                # precision, then it is replaced by 0
                evals = np.clip(evals, 0, np.inf)
                return np.sqrt(evals), evecs.T, L_out_leads, X_out_leads
    else:
        if not uv:
            return la.svd(lhs, compute_uv=False)
        else:
            U, S, Wh = la.svd(lhs, compute_uv=True)
            return S, Wh, L_out_leads, X_out_leads

def setup_lhs(H_s, V_leads, transf, L_out_leads, X_out_leads, E, sparse=False):
    """
    Setup the left hand side of the equation to find bound states.
    """
    DBh_leads = hopping_svd(V_leads)
    DBh = block_diag(*DBh_leads)
    L_out = block_diag(*L_out_leads)
    X_out = block_diag(*X_out_leads)
    V = block_diag(*V_leads)

    if sparse: #Uses either scipy.sparse if sparse of numpy if dense
        d = sp.csr_matrix.dot
        lin = sp
        DBh, V = sp.csr_matrix(DBh), sp.csr_matrix(V)
        L_out, X_out = sp.csr_matrix(L_out), sp.csr_matrix(X_out)
        # V is a csc matrix because it is transposed later and it becomes a csr matrix
    else:
        d = np.dot
        lin = np

    Id_N = lin.eye(H_s.shape[0])
    DBh_ls = d(DBh, transf)
    V_ls = V.dot(transf)

    lhs = lin.hstack([H_s - E * Id_N, V_ls.conj().T.dot(d(X_out, L_out))])
    lhs = lin.vstack([lhs, lin.hstack([DBh_ls, -d(DBh, X_out)])])
    return lhs

def plot_singular_values(sys, e_min, e_max, N=500, sparse=False, sigma=None,
                        k=5):
    energies = np.linspace(e_min, e_max, N)

    H_s, H_leads, V_leads, transf = extract_kwant_matrices(sys, sparse=sparse)

    if sparse:
        dumb = np.zeros(shape=(N, k))
    else:
        # at maximum the number of singular value is the dimension of H_s
        # + the dimension of all the leads together
        dumb = np.zeros(shape=(N, H_s.shape[0] + sum(V.shape[0] for V in V_leads)))
        # the zeros are later replaced with NaN so they are not plotted.

    for i, e in enumerate(energies):
        s = singular_values(H_s, H_leads, V_leads, transf, e, sparse, 
                            uv=False, sigma=sigma, sing_values=k)

        dumb[i, :len(s)] = s
        dumb = fill_zero_with_nan(dumb)

    return (energies, dumb)

