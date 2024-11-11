import numpy as np
from numba import njit

def do_Max_Overlap_Method( C, old_C, occ_inds ):
    # Find the best overlap between the new MOs and occupied subspace of old MOs
    old_C = old_C[:,occ_inds] # Only consider occupied orbitals of old MOs
    mo_overlap  = np.einsum( "ai,aj->ij", old_C[:,:], C[:,:] )
    p           = np.einsum( "ij->j", mo_overlap ) # j is the index of the new MO
    best_perm   = np.argsort( -np.abs(p) ) # Phase of C should not affect this # Get n_elec indices of MOs with largest overlap
    return best_perm

@njit
def do_DAMP( F, old_F, DAMP=0.75 ):
    return DAMP * F + (1-DAMP) * old_F

def make_RDM1_ao( C, occ_inds ):
    """
    D[:,:] = np.einsum( "ai,bi->ab", C[:,occ_inds], C[:,occ_inds] )
    """
    if ( type(occ_inds) == type(5) ):
        occ_inds = (np.arange(occ_inds))
    return np.einsum( "ai,bi->ab", C[:,occ_inds], C[:,occ_inds] )

def get_spin_analysis( mo_a, mo_b, ao_overlap=None ):
    from functools import reduce
    nocc_a     = mo_a.shape[1] # Number of occupied alpha orbitals
    nocc_b     = mo_b.shape[1] # Number of occupied beta orbitals
    if ( ao_overlap is not None ):
        mo_ab_overlap = np.einsum("ai,ab,bj->ij", mo_a.conj(), ao_overlap, mo_b)
    else:
        mo_ab_overlap = np.einsum("ai,aj->ij", mo_a.conj(), mo_b)
    ssxy       = (nocc_a+nocc_b) / 2 - np.einsum('ij,ij->', mo_ab_overlap.conj(), mo_ab_overlap)
    ssz        = (nocc_b-nocc_a)**2 / 4
    ss         = (ssxy + ssz).real
    s          = np.sqrt(ss+0.25) - 0.5
    return ss, s*2+1

def get_J( D, eri ):
    return np.einsum( 'rs,pqrs->pq', D, eri )

def get_K( D, eri ):
    return np.einsum( 'rs,psrq->pq', D, eri )

def get_JK( D, eri ):
    J = get_J( D, eri )
    K = get_K( D, eri )
    return J, K

def get_orbital_gradient( F, n_elec ):
    """
    Returns the orbital gradient, G = 2 * ( F - F.T )
    F: Fock matrix in the MO basis 
    G: Orbital gradient: G_ov = F_ov - F_vo
    """
    ov = F[:n_elec,n_elec:]
    vo = F[n_elec:,:n_elec]
    return 2 *( ov - vo )

@njit
def eigh( F ):
    return np.linalg.eigh( F )

