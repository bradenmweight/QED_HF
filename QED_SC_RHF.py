import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from pyscf import gto, scf, fci

from tools import to_ortho_ao, from_ortho_ao, eigh, make_RDM1_ao, do_DAMP
from ao_ints import get_ao_integrals, get_dipole_quadrupole
from DIIS import DIIS

def do_QED_SC_RHF( mol, LAM, WC ):

    S, Shalf, h1e, eri, n_elec_alpha, n_elec_beta, nuclear_repulsion_energy = get_ao_integrals( mol )
    dip_ao, quad_ao = get_dipole_quadrupole( mol, Shalf.shape[0] )

    # Rotate all relevant integrals to the orthogonal AO basis
    h1e     = to_ortho_ao( Shalf, h1e, shape=2 )
    eri     = np.einsum( 'ap,bq,abcd,cr,ds->pqrs', Shalf, Shalf, eri, Shalf, Shalf )
    dip_ao  = to_ortho_ao( Shalf, dip_ao[-1,:,:], shape=2 )
    quad_ao = to_ortho_ao( Shalf, quad_ao[-1,-1,:,:], shape=2 )

    # # Diagonalize the dipole operator
    Emu, Umu = np.linalg.eigh( dip_ao )

    # # Rotate h2e and eri to the dipole basis
    h1e = np.einsum( "ap,ab,bq->pq", Umu, h1e, Umu )
    print("Starting eri dipole transformation.")
    eri = np.einsum( "ap,bq,abcd,cr,ds->pqrs", Umu, Umu, eri, Umu, Umu )

    # # Construct the X-operator (X. Li and Y. Zhang arXiv) in orthogonal AO basis
    Z    = LAM**2 / 4 / WC
    Emu_minus_Emu  = np.array([  p - q for p in Emu for q in Emu ]).reshape( (len(Emu),len(Emu)) )
    Emu_minus_Emu2 = np.array([  p - q + r - s for p in Emu for q in Emu for r in Emu for s in Emu ]).reshape( (len(Emu),len(Emu),len(Emu),len(Emu)) )
    h1e  = np.einsum( "pq,pq->pq", h1e, np.exp( -Z * Emu_minus_Emu**2 ) ) # <p,0|X.T.conj() @ h1e @ X|p,0>
    print("Starting eri X weighting.")
    eri  = np.einsum( "pqrs,pqrs->pqrs", eri, np.exp( -Z * Emu_minus_Emu2**2 ) ) # <p,0|X.T.conj() X.T.conj() @ eri @ X @ X|p,0>

    # Choose core as guess for Fock matrix
    F = h1e

    # Diagonalize Fock
    eps, C = eigh( F )

    # Get density matrix in AO basis
    D    = make_RDM1_ao( C, n_elec_alpha )

    e_convergence = 1e-8
    d_convergence = 1e-6
    maxiter       = 200

    old_energy = np.einsum("ab,ab->", D, 2*h1e ) + nuclear_repulsion_energy
    old_D = D.copy()
    old_F = F.copy()
    old_C = C.copy()
    DIIS_flag = False
    MOM_flag  = False

    myDIIS = DIIS( unrestricted=False, ao_overlap=np.eye(len(Emu)) )

    #print("    Guess Energy: %20.12f" % old_energy)
    for iter in range( maxiter ):

        # Coulomb and exchange matrix
        J     = np.einsum( 'rs,pqrs->pq', D, eri )
        K     = np.einsum( 'rs,prsq->pq', D, eri )

        # Fock matrix
        F  = h1e     + 2 * J     - K
        
        if ( iter < 5 ):
            F = do_DAMP( F, old_F )

        if ( iter > 2 and iter < 10 ):
            F = myDIIS.extrapolate(F, D)

        # Diagonalize Fock matrix
        eps, C = eigh( F )

        if ( MOM_flag == True ):
            occ_inds = do_Max_Overlap_Method( C, old_C, S, n_elec_alpha )
        else:
            occ_inds = (np.arange(n_elec_alpha))

        # Get density matrix in AO basis
        D = make_RDM1_ao( C, n_elec_alpha )

        # Get current energy for RHF
        energy  = np.einsum("ab,ab->", D, 2*h1e + 2*J - K )
        energy += nuclear_repulsion_energy
        energy += 0.5 * WC

        dE = energy - old_energy
        dD = np.linalg.norm( D - old_D )

        if ( iter > 50 ):
            print("    Iteration %3d: Energy = %1.12f, dE = %1.8f, dD = %1.6f" % (iter, energy, dE, dD))

        old_energy = energy
        old_D      = D.copy()

        if ( iter > 2 and abs(dE) < e_convergence and dD < d_convergence ):
            break
        if ( iter == maxiter-1 ):
            print("FAILURE: QED-RHF DID NOT CONVERGE")
            return float('nan')

        if ( iter > 100 ): # Try doing DIIS
            DIIS_flag = True
        if ( iter > 200 ): # Try doing MOM
            MOM_flag = True
            DIIS_flag = False

    print('    * SC-QED-RHF Total Energy: %20.12f' % (energy))

    return energy

if (__name__ == '__main__' ):
    mol = gto.Mole()
    mol.basis = "ccpvdz"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'H 0 0 0; H 0 0 2.0'
    mol.build()
    E_RHF = do_QED_SC_RHF( mol, 0.1, 0.1 )
