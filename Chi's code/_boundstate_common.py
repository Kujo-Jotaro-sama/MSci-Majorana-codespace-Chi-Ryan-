from kwant.physics.leads import unified_eigenproblem, setup_linsys # should be a relative import when in kwant
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

def fill_zero_with_nan(array):
    """
    Replace all elements equal to 0 by nan so that it is 
    not plotted by pyplot
    """
    S = array.shape
    array = np.reshape(array, S[0] * S[1])
    for i, element in enumerate(array):
        if element != 0:
            array[i] = element
        else:
            array[i] = float('nan')
    return np.reshape(array, (S[0], S[1]))

def leads_modes(H_leads, V_leads, E, need_orth=True):
    """
    Compute the evanescent modes for every lead separetly
    """
    X_out_leads = []
    L_out_leads = []
    assert len(H_leads)==len(V_leads), 'Check the number of leads'

    for H, V in zip(H_leads, V_leads): #compute the modes of every lead
        L_out, X_out = extract_out_modes(H, V, E, need_orth=need_orth)

        X_out_leads.append(X_out)
        L_out_leads.append(L_out)
    return L_out_leads, X_out_leads

def block_diag(*arrs):
    """
    Copied from old version of scipy, v0.8.0
    Create a block diagonal matrix from the provided arrays.

    Given the inputs `A`, `B` and `C`, the output will have these
    arrays arranged on the diagonal::

        [[A, 0, 0],
         [0, B, 0],
         [0, 0, C]]

    If all the input arrays are square, the output is known as a
    block diagonal matrix.

    Parameters
    ----------
    A, B, C, ... : array-like, up to 2D
        Input arrays.  A 1D array or array-like sequence with length n is
        treated as a 2D array with shape (1,n).

    Returns
    -------
    D : ndarray
        Array with `A`, `B`, `C`, ... on the diagonal.  `D` has the
        same dtype as `A`.

    References
    ----------
    .. [1] Wikipedia, "Block matrix",
           http://en.wikipedia.org/wiki/Block_diagonal_matrix

    Examples
    --------
    >>> A = [[1, 0],
    ...      [0, 1]]
    >>> B = [[3, 4, 5],
    ...      [6, 7, 8]]
    >>> C = [[7]]
    >>> print(block_diag(A, B, C))
    [[1 0 0 0 0 0]
     [0 1 0 0 0 0]
     [0 0 3 4 5 0]
     [0 0 6 7 8 0]
     [0 0 0 0 0 7]]
    >>> block_diag(1.0, [2, 3], [[4, 5], [6, 7]])
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  2.,  3.,  0.,  0.],
           [ 0.,  0.,  0.,  4.,  5.],
           [ 0.,  0.,  0.,  6.,  7.]])

    """
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_2d(a) for a in arrs]

    bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
    if bad_args:
        raise ValueError("arguments in the following positions have dimension "
                            "greater than 2: %s" % bad_args) 

    shapes = np.array([a.shape for a in arrs])
    out = np.zeros(np.sum(shapes, axis=0), dtype=arrs[0].dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out

def extract_out_modes(H, V, E, return_mat=False, need_orth=True):
    """
    parameters:
    H: hamiltonian inside the unit cell of the lead
    V: hoppings between the cells
    E: Energy at which the mode is calculated
    need_orth: bool, if True it returns orthogonalized 
                eigenvectors Phi_e, only necessary for degenerate evs
    
    returns:
    modes: a list of evanescent right going modes
    Normalization of X_out is such that max(X_out[:, i]) = 1
    """
    n = H.shape[0]
    Id = np.eye(n)
    H = H - E * Id
    V_dag = V.conj().T

    matrices, v, extract = setup_linsys(H, V, stabilization=(True, True))
    all_lmb, evanselect, propselect, vec_gen, ord_schur = unified_eigenproblem(*(matrices))
    lmb_inv = all_lmb[evanselect]
    all_vecs = vec_gen(evanselect)
    Phi = np.zeros(shape=(n, len(lmb_inv)), dtype=complex)

    for i, l in enumerate(lmb_inv):
        phi = extract(all_vecs[:, i], l)
        phi /= np.sqrt(np.dot(phi.conj().T, phi))
        Phi[:, i] = phi

    Phi, arg_max, max_x = max_norm(Phi)

    if need_orth:
        if len(lmb_inv):
            Q, R = np.linalg.qr(Phi)
            Phi = Q
            R_inv = la.solve(R, np.eye(R.shape[0]))
            L = R.dot(np.diag(1 / lmb_inv)).dot(R_inv)
        else: L = np.zeros(shape=(len(lmb_inv), 0), dtype=complex)
    else:
        L = np.diag(1 / lmb_inv)

    if return_mat:
        return L, Phi, R
    else:
        return L, Phi

def extract_kwant_matrices(sys, sparse=False, args=(), params=None):
    """
    Function that takes a finalized kwant system (with at least one lead)
    and returns the matrices defining the onsite and hoppings in the 
    system.
    Parameters:
    sys: a kwant system with at least one lead
    sparse: whether to return sparse or dense matrices
    args : tuple
    Positional arguments to the system.
    params : dict
        Keyword arguments to the system.
    Returns:
    H_s: Hamiltonian of the scattering region
    H_leads: list countaining the hamiltonians of the unit cells of the 
             leads
    V_leads: list countaining the hoppings between the unit cells of the 
             leads
    transf: Rectangular matrix full of 0s and 1s that connects the lead 
            and the scattering region. transf[i, j] = 1 only if the site
            i from the lead and the site j from the scattering region 
            have a non-zero hopping.
    """

    assert(len(sys.leads)), print('The system should be infinite')

    H_s, norb, _ = sys.hamiltonian_submatrix(sparse=sparse,
                                             args=args, params=params,
                                             return_norb=True)
    H_leads = []
    V_leads = []
    coords = []
    for i, lead in enumerate(sys.leads):
        # H and V are not written in sparse at this point because the 
        # mode solver needs a numpy array
        H_leads.append(sys.leads[i].cell_hamiltonian(args=args, params=params))
        V_leads.append(sys.leads[i].inter_cell_hopping(args=args, params=params))
        coords = np.concatenate((coords, sys.lead_interfaces[i]))

    offsets = np.empty(norb.shape[0] + 1, int)
    offsets[0] = 0
    offsets[1 :] = np.cumsum(norb)
    ones = []
    for coord in coords:
        coord = int(coord)
        for i in range(offsets[coord], offsets[coord] + np.array(norb)[coord]):
            ones.append(i)

    if sparse:
        transf = sp.csr_matrix((np.ones(len(ones)), (range(len(ones)),ones)),
                                shape=(len(ones),sum(norb)))
    else:
        # most inefficient way to create a dense array, to change at some point
        transf = sp.csr_matrix((np.ones(len(ones)), (range(len(ones)),ones)),
                                shape=(len(ones),sum(norb))).toarray()

    return H_s, H_leads, V_leads, transf

def compute_wf(vals, vecs, L_leads, X_out_leads, scat_dim,
               tol=1e-4, schur_modes=False):
    """
    Extract and normalize the wavefunction from the eigenvectors of 
    the lhs (H_eff) matrix.

    vals: eigenvalues of H_eff computed at the energy of the bound state
    vecs: eigenvectors of H_eff computed at the energy of the bound state
    L_leads: list of matrices with the evanescent modes in
    X_leads:
    scat_dim: int, number of sites in the scattering region
    schur_modes: if True, bound state is returned in schur basis, it is 
                returned in the basis with Lambda diagonal otherwise
    """
    zero_schur = (abs(vals) < tol)
    # Psi_alpha_0 is the wavefunction of the bound states in the
    # system.
    psi_alpha_0, q_e = vecs[:scat_dim, zero_schur], vecs[scat_dim:, zero_schur]
    #print('hello!')
    #print('in spyder')
    #print(psi_alpha_0)
    #print(q_e)

    Q = block_diag(*X_out_leads)
    L_out = block_diag(*L_leads)

    # When modes are degenerated, Lambda is not diagonal anymore
    lmb, R = la.eig(L_out)
    # R is already computed in extract_out_modes, is it faster to keep it in memory?
    Phi = Q @ R
    # my changes
    q_e = np.linalg.inv(R)@q_e

    N = Phi.conj().T @ Phi
    #print('N')
    #print(N.shape)
    #print(N)
    
    #print(enumerate(lmb))

    for i, l1 in enumerate(lmb):
        for j, l2 in enumerate(lmb):
            ll = np.conj(l1) * l2
            N[i, j] *= ll / (1 - ll)
    #print('new N')
    #print(N)

    for i, psi in enumerate(psi_alpha_0.T): #loop if degeneracy
        #print(i)
        #print(psi)
        a = q_e[:, i]
        #print(a)
        norm = psi.conj().T @ psi
        #print(norm)
        norm += a.conj().T @ N @ a
        #print(norm)
        norm = np.sqrt(norm)
        #print(norm)
        psi_alpha_0[:, i] /= norm
        q_e[:, i] /= norm

    if schur_modes:
        return psi_alpha_0, q_e, L_out, Q
    else:
        return psi_alpha_0, q_e, np.diag(lmb), Phi

def max_norm(X):
    """
    Norm every column of X such that the maximum value of the column is equal to 1
    """
    X_tilde = np.zeros(X.shape, dtype=complex)
    arg_max, max_x = [], []

    for i, x in enumerate(X.T):
        x_max = np.argmax(abs(x)) # take the maximum to avoid division by small numbers
        X_tilde[:, i] = x / x[x_max]
        arg_max.append(x_max)
        max_x.append(x[x_max])
    return X_tilde, arg_max, max_x
