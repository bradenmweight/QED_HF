import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf, fci

from tools import get_JK, make_RDM1_ao, do_DAMP, eigh, do_Max_Overlap_Method
from ao_ints import get_ao_integrals
from DIIS import DIIS

def do_RHF( mol, initial_guess=None, return_wfn=False ):

    # Get ao integrals
    h1e, eri, n_elec_alpha, n_elec_beta, nuclear_repulsion_energy = get_ao_integrals( mol )

    if ( initial_guess is not None ):
        C = initial_guess
        D = make_RDM1_ao( C, n_elec_alpha )
        J,K = get_JK( D, eri )
        F = h1e + 2 * J - K
        eps, C = np.linalg.eigh( F )
        D = make_RDM1_ao( C, n_elec_alpha )
        old_energy = np.einsum("ab,ab->", D, 2*h1e + 2*J - K ) + nuclear_repulsion_energy
    else:
        # Choose core as guess for Fock
        F      = h1e
        eps, C = np.linalg.eigh( F )
        D      = np.einsum( "ai,bi->ab", C[:,:n_elec_alpha], C[:,:n_elec_alpha] )
        old_energy = np.einsum("ab,ab->", D, 2*h1e ) + nuclear_repulsion_energy

    e_convergence = 1e-8
    d_convergence = 1e-6
    maxiter       = 2000

    old_C = C.copy()
    old_D = D.copy()
    old_F = F.copy()

    myDIIS = DIIS()

    for iter in range( maxiter ):

        # Coulomb and Exchange matrix
        J, K = get_JK( D, eri )

        # Fock matrix for RHF
        F = h1e + 2 * J - K

        F = do_DAMP( F, old_F )
        
        if ( iter > 10 ):
            F = myDIIS.extrapolate( F, D )

        # Diagonalize Fock matrix
        eps, C = np.linalg.eigh( F )

        # Get density matrix in AO basis
        D  = np.einsum("ai,bi->ab", C[:,:n_elec_alpha], C[:,:n_elec_alpha])

        # Get current energy for RHF
        energy  = np.einsum("ab,ab->", D, 2*h1e + 2*J - K )
        energy += nuclear_repulsion_energy

        dE = energy - old_energy
        dD = np.linalg.norm( D - old_D )

        # if ( iter > 5 ):            
        #     inds = do_Max_Overlap_Method( C, old_C, (np.arange(n_elec_alpha)) )
        #     C    = C[:,inds]
        #     D    = make_RDM1_ao( C, (np.arange(n_elec_alpha)) )
        #     dD   = np.linalg.norm( D - old_D )

        print( '    Iteration %3d: Energy = %4.12f, dE = %1.5e, |dD| = %1.5e' % (iter, energy, dE, dD ) )

        old_energy = energy
        old_D      = D.copy()

        if ( iter > 2 and abs(dE) < e_convergence and dD < d_convergence ):
            break
        if ( iter == maxiter-1 ):
            print("FAILURE: RHF DID NOT CONVERGE")
            if ( return_wfn == True ):
                return float('nan'), C * 0 + 1
            return float('nan')

    print('    *     RHF Total Energy: %20.12f' % (energy))
    #print('    * RHF Wavefunction:', np.round( C[:,0],3))

    if ( return_wfn == True ):
        return energy, C
    return energy

if (__name__ == '__main__' ):

    mol = gto.Mole()
    mol.basis = "ccpvdz"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'H 0 0 0; H 0 0 2.0'
    mol.build()
    E    = do_RHF( mol )
    E, C = do_RHF( mol, return_wfn=True )
    E    = do_RHF( mol, initial_guess=C )

    mol.atom = 'H 0 0 0; H 0 0 10.0'
    mol.build()
    E = do_RHF( mol )
    E = do_RHF( mol, initial_guess=C )