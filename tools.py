import numpy as np

from numba import njit


def do_Max_Overlap_Method( C, old_C, ao_overlap, n_elec ):
    # Find the best overlap between the new MOs and occupied subspace of old MOs
    old_C = old_C[:,:n_elec] # Only consider occupied orbitals of old MOs
    max_overlap = 0.0
    mo_overlap  = np.einsum( "ai,ab,bj->ij", old_C[:,:], ao_overlap, C[:,:] )
    p           = np.einsum( "ij->j", mo_overlap ) # j is the index of the new MO
    best_perm   = np.argsort( -p )[:n_elec] # Get n_elec indices of MOs with largest overlap
    return best_perm

@njit
def do_DAMP( F, old_F, DAMP=0.5 ):
    return DAMP * F + (1-DAMP) * old_F

@njit
def to_ortho_ao( U, M, shape ):
    if ( shape == 1 ):
        return U.T @ M
    elif ( shape == 2 ):
        return U.T @ M @ U

@njit
def from_ortho_ao( U, M, shape ):
    if ( shape == 1 ):
        return U @ M
    elif ( shape == 2 ):
        return U @ M @ U.T

@njit
def eigh( F ):
    return np.linalg.eigh( F )

#@njit
def make_RDM1_ao( C, occ_inds ):
    if ( type(occ_inds) == type(1) ):
        occ_inds = (np.arange(occ_inds))
    D = np.zeros( (len(C),len(C)) )
    for a in range( len(C) ):
        for b in range( len(C) ):
            D[a,b] = np.sum(C[a,occ_inds] * C[b,occ_inds] )
    return D

def get_spin_analysis( mo_a, mo_b, ao_overlap=None ):
    from functools import reduce
    nocc_a     = mo_a.shape[1] # Number of occupied alpha orbitals
    nocc_b     = mo_b.shape[1] # Number of occupied beta orbitals
    if ( ao_overlap is not None ):
        mo_overlap = np.einsum("ai,ab,bj->ij", mo_a.conj(), ao_overlap, mo_b)
    else:
        mo_overlap = np.einsum("ai,aj->ij", mo_a.conj(), mo_b)
    ssxy       = (nocc_a+nocc_b) / 2 - np.einsum('ij,ij->', mo_overlap.conj(), mo_overlap)
    ssz        = (nocc_b-nocc_a)**2 / 4
    ss         = (ssxy + ssz).real
    s          = np.sqrt(ss+0.25) - 0.5
    return ss, s*2+1