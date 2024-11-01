import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf, fci

from tools import to_ortho_ao, from_ortho_ao, eigh, make_RDM1_ao, do_DAMP
from ao_ints import get_ao_integrals, get_dipole_quadrupole
from DIIS import DIIS

def do_RHF( mol ):

    # Get ao integrals
    S, Shalf, h1e, eri, n_elec_alpha, n_elec_beta, nuclear_repulsion_energy = get_ao_integrals( mol )

    # Choose core as guess for Fock
    F = h1e

    # Rotate Fock to orthogonal basis
    F_ORTHO = to_ortho_ao( Shalf, F, shape=2 )

    # Diagonalize Fock
    eps, C = np.linalg.eigh( F_ORTHO )

    # Rotate all MOs back to non-orthogonal AO basis
    C = from_ortho_ao( Shalf, C, shape=1 )

    # Get density matrix in AO basis
    D    = np.einsum( "ai,bi->ab", C[:,:n_elec_alpha], C[:,:n_elec_alpha] )

    e_convergence = 1e-8
    d_convergence = 1e-6
    maxiter       = 200

    old_energy = 0.5 * np.einsum("ab,ab->", D, h1e + F ) + nuclear_repulsion_energy
    old_D = D.copy()
    old_F = F.copy()

    myDIIS = DIIS( unrestricted=False, ao_overlap=S )

    #print("    Guess Energy: %20.12f" % old_energy)
    for iter in range( maxiter ):

        # Coulomb matrix
        J = np.einsum( 'rs,pqrs->pq', D, eri )

        # Exchange matrix
        K = np.einsum( 'rs,psrq->pq', D, eri )

        # Fock matrix for RHF
        F = h1e + 2 * J - K

        if ( iter < 5 ):
            F = do_DAMP( F, old_F )
        
        if ( iter > 5 ):
            myDIIS.extrapolate( F, D )

        # Transfom Fock matrix to orthogonal basis
        F_ORTHO = to_ortho_ao( Shalf, F, shape=2 )

        # Diagonalize Fock matrix
        eps, C = np.linalg.eigh( F_ORTHO )

        # Rotate MOs back to non-orthogonal AO basis
        C = from_ortho_ao( Shalf, C, shape=1 )

        # Get density matrix in AO basis
        D  = np.einsum("ai,bi->ab", C[:,:n_elec_alpha], C[:,:n_elec_alpha])

        # Get current energy for RHF
        energy  = np.einsum("ab,ab->", D, 2*h1e + 2*J - K )
        energy += nuclear_repulsion_energy

        dE = np.abs( energy - old_energy )
        dD = np.linalg.norm( D - old_D )

        #print( '    Iteration %3d: Energy = %4.12f, Energy change = %1.5e, Density change = %1.5e' % (iter, energy, dE, dD ) )

        old_energy = energy
        old_D      = D.copy()

        if ( iter > 2 and abs(dE) < e_convergence and dD < d_convergence ):
            break
        if ( iter == maxiter-1 ):
            print("FAILURE: QED-UHF DID NOT CONVERGE")
            break

    #myRHF = scf.RHF( mol )
    #e_rhf = myRHF.kernel()
    # Full configuration interaction
    #e_fci = fci.FCI( myRHF ).kernel()[0]
    #print('    * FCI Total Energy (PySCF): %20.12f' % (e_fci))
    #print('    * RHF Total Energy (PySCF) : %20.12f' % (e_rhf))
    print('    *     RHF Total Energy (Braden): %20.12f' % (energy))
    #print('    * RHF Wavefunction:', np.round( C[:,0],3))
    return energy#, e_rhf, e_fci

if (__name__ == '__main__' ):

    mol = gto.Mole()
    mol.basis = "ccpvdz"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'H 0 0 0; H 0 0 10.0'
    mol.build()
    E_RHF = do_RHF( mol )