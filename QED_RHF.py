import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from pyscf import gto, scf, fci

from tools import to_ortho_ao, from_ortho_ao, eigh, make_RDM1_ao, do_DAMP
from ao_ints import get_ao_integrals, get_dipole_quadrupole
from DIIS import DIIS


def get_QED_HF_ZHY( mol, LAM, WC ):
    from openms import mqed
    cavity_freq     = np.array([WC]) # a.u.
    cavity_coupling = np.array([LAM])
    cavity_vec      = np.array([np.array([0,0,1])])
    cavity_mode     = np.einsum("m,md->md", cavity_coupling, cavity_vec ) / np.linalg.norm(cavity_vec,axis=-1)

    qedmf = mqed.HF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
    qedmf.max_cycle = 500
    qedmf.kernel()
    if ( qedmf.conv_check == False ):
        print("   Warning! QED-HF did not converge. Setting energy to NaN.")
        return float('nan')
    else:
        return qedmf.e_tot

def do_QED_RHF( mol, LAM, WC, do_coherent_state=True ):

    S, Shalf, h1e, eri, n_elec_alpha, n_elec_beta, nuclear_repulsion_energy = get_ao_integrals( mol )
    dip_ao, quad_ao = get_dipole_quadrupole( mol, Shalf.shape[0] )

    # Choose core as guess for Fock matrix
    F = h1e

    # Rotate Fock matrix to orthogonal ao basis
    F_ORTHO = to_ortho_ao( Shalf, F, shape=2 )

    # Diagonalize Fock
    eps, C = eigh( F_ORTHO )

    # Rotate all MOs back to non-orthogonal AO basis
    C = from_ortho_ao( Shalf, C, shape=1 )

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

    myDIIS = DIIS( unrestricted=False, ao_overlap=S )

    #print("    Guess Energy: %20.12f" % old_energy)
    for iter in range( maxiter ):

        # DSE
        if ( do_coherent_state == True ):
            AVEdipole = np.einsum( 'pq,pq->', D, dip_ao[-1,:,:] )
        else:
            AVEdipole = 0.0
        
        DSE_FACTOR = 0.5 * LAM**2
        h1e_DSE =     DSE_FACTOR * ( -2*AVEdipole * dip_ao[-1,:,:] + quad_ao[-1,-1,:,:] ) 
        eri_DSE = 2 * DSE_FACTOR  * np.einsum( 'pq,rs->pqrs', dip_ao[-1,:,:], dip_ao[-1,:,:] )

        # Coulomb matrix
        J     = np.einsum( 'rs,pqrs->pq', D, eri )
        DSE_J = np.einsum( 'rs,pqrs->pq', D, eri_DSE )

        # Exchange matrix
        K     = np.einsum( 'rs,prsq->pq', D, eri )
        DSE_K = np.einsum( 'rs,prsq->pq', D, eri_DSE )


        # Fock matrix
        F  = h1e     + 2 * J     - K
        F += h1e_DSE +     DSE_J - DSE_K
        
        if ( iter < 5 ):
            F = do_DAMP( F, old_F )

        if ( iter > 2 and iter < 10 ):
            F = myDIIS.extrapolate(F, D)

        # Transfom Fock matrix to orthogonal basis
        F_ORTHO = to_ortho_ao( Shalf, F, shape=2 )

        # Diagonalize Fock matrix
        eps, C = eigh( F_ORTHO )

        # Rotate MOs back to non-orthogonal AO basis
        C = from_ortho_ao( Shalf, C, shape=1 )

        if ( MOM_flag == True ):
            occ_inds = do_Max_Overlap_Method( C, old_C, S, n_elec_alpha )
        else:
            occ_inds = (np.arange(n_elec_alpha))

        # Get density matrix in AO basis
        D = make_RDM1_ao( C, n_elec_alpha )

        # Get current energy for RHF
        energy  = np.einsum("ab,ab->", D, 2*h1e + 2*J - K )
        energy += np.einsum("ab,ab->", D, 2*h1e_DSE + 2*DSE_J - DSE_K )
        energy += nuclear_repulsion_energy
        energy += DSE_FACTOR*AVEdipole**2
        energy += 0.5 * WC

        dE = energy - old_energy
        dD = np.linalg.norm( D - old_D )

        if ( iter > 50 ):
            print("    Iteration %3d: Energy = %1.12f, dE = %1.12f, dD = %1.12f" % (iter, energy, dE, dD))

        old_energy = energy
        old_D      = D.copy()

        if ( iter > 2 and abs(dE) < e_convergence and dD < d_convergence ):
            break
        if ( iter == maxiter-1 ):
            print("FAILURE: QED-UHF DID NOT CONVERGE")
            break

        if ( iter > 100 ): # Try doing DIIS
            DIIS_flag = True
        if ( iter > 200 ): # Try doing MOM
            MOM_flag = True
            DIIS_flag = False

    #myRHF   = scf.RHF( mol )
    #e_qedhf = get_QED_HF_ZHY( mol, LAM, WC )
    #e_rhf   = myRHF.kernel() + 0.5 * WC
    #e_fci   = fci.FCI( myRHF ).kernel()[0] + 0.5 * WC
    #print('    * FCI Total Energy (PySCF): %20.12f' % (e_fci))
    #print('    * RHF Total Energy (PySCF) : %20.12f' % (e_rhf))
    print('    * QED-RHF Total Energy: %20.12f' % (energy))
    #print('    * RHF Wavefunction:', np.round( C[:,0],3))

    return energy#, e_qedhf, e_rhf, e_fci

if (__name__ == '__main__' ):
    mol = gto.Mole()
    mol.basis = "ccpvdz"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'H 0 0 0; H 0 0 10.0'
    mol.build()
    E_RHF = do_QED_RHF( mol, 0.0, 0.1 )
