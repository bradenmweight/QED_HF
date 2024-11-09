import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from pyscf import gto, scf, fci

from tools import eigh, make_RDM1_ao, do_DAMP, get_JK, do_Max_Overlap_Method
from ao_ints import get_ao_integrals
from DIIS import DIIS


def do_QED_HF_ZHY( mol, LAM, WC ):
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

def do_QED_RHF( mol, LAM, WC, do_CS=True, return_wfn=False, initial_guess=None ):
    DSE_FACTOR = 0.5 * LAM**2

    h1e, eri, n_elec_alpha, n_elec_beta, nuclear_repulsion_energy, dip_ao, quad_ao = get_ao_integrals( mol, dipole_quadrupole=True )

    if ( initial_guess is not None ):
        C           = initial_guess
        D           = make_RDM1_ao( C, n_elec_alpha )
        AVEdipole   = (do_CS) * np.einsum( 'pq,pq->', D, dip_ao )
        h1e_DSE     =     DSE_FACTOR * ( -2*AVEdipole * dip_ao + quad_ao ) 
        eri_DSE     = 2 * DSE_FACTOR  * np.einsum( 'pq,rs->pqrs', dip_ao, dip_ao )
        J,K         = get_JK( D, eri )
        DSE_J,DSE_K = get_JK( D, eri_DSE )
        F           = h1e     + 2 * J -     K
        F          += h1e_DSE + DSE_J - DSE_K
        eps, C      = np.linalg.eigh( F )
        D           = make_RDM1_ao( C, n_elec_alpha )
        old_energy  = np.einsum("ab,ab->", D, 2*h1e + 2*J - K )
        old_energy += np.einsum("ab,ab->", D, 2*h1e_DSE + 2*DSE_J - DSE_K )
        old_energy += nuclear_repulsion_energy
        old_energy += 0.5 * WC
    else:
        # Choose core as guess for Fock
        F          = h1e
        eps, C     = eigh( F )
        D          = make_RDM1_ao( C, n_elec_alpha )
        old_energy = np.einsum("ab,ab->", D, 2*h1e )
        old_energy += nuclear_repulsion_energy
        old_energy += 0.5 * WC

    e_convergence = 1e-8
    d_convergence = 1e-6
    maxiter       = 200

    old_D  = D.copy()
    old_F  = F.copy()
    old_C  = C.copy()
    dE     = 0.0
    old_dE = 0.0

    myDIIS = DIIS()

    for iter in range( maxiter ):

        # DSE
        AVEdipole  = (do_CS) * np.einsum( 'pq,pq->', D, dip_ao )
        h1e_DSE    =     DSE_FACTOR * ( -2*AVEdipole * dip_ao + quad_ao ) 
        eri_DSE    = 2 * DSE_FACTOR  * np.einsum( 'pq,rs->pqrs', dip_ao, dip_ao )

        # Coulomb and exchange matrix
        J,K          = get_JK( D, eri )
        DSE_J, DSE_K = get_JK( D, eri_DSE )

        # Fock matrix
        F  = h1e     +     2 * J     - K
        F += h1e_DSE +     DSE_J - DSE_K
        
        F = do_DAMP( F, old_F )

        if ( iter > 1 ):
            F = myDIIS.extrapolate( F, D )

        # Diagonalize Fock matrix
        eps, C = eigh( F )

        # Get density matrix in AO basis
        D = make_RDM1_ao( C, (np.arange(n_elec_alpha)) )

        # Get current energy for RHF
        energy  = np.einsum("ab,ab->", D, 2*h1e + 2*J - K )
        energy += np.einsum("ab,ab->", D, 2*h1e_DSE + 2*DSE_J - DSE_K )
        energy += nuclear_repulsion_energy
        #energy += DSE_FACTOR*AVEdipole**2
        energy += 0.5 * WC

        old_dE = dE
        dE     = energy - old_energy
        dD     = np.linalg.norm( D - old_D )

        if ( iter > 2 and dD > 1.0 ):            
           inds = do_Max_Overlap_Method( C, old_C, (np.arange(n_elec_alpha)) )
           C    = C[:,inds]
           D    = make_RDM1_ao( C, (np.arange(n_elec_alpha)) )
           dD   = 2 * np.linalg.norm( D - old_D )

        #print("    Iteration %3d: Energy = %1.12f, dE = %1.8f, dD = %1.6f" % (iter, energy, dE, dD))

        if ( iter > 5 and abs(dE) < e_convergence and dD < d_convergence ):
            break
        if ( iter == maxiter-1 ):
            print("FAILURE: QED-RHF DID NOT CONVERGE")
            if ( return_wfn == True ):
                return float('nan'), C*0 + 1
            return float('nan')

        old_energy = energy
        old_D      = D.copy()
        old_C      = C.copy()

    print('    * QED-RHF Total Energy: %20.12f' % (energy))
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
    LAM  = 0.1
    WC   = 1.0
    E    = do_QED_RHF( mol, LAM, WC )
    E, C = do_QED_RHF( mol, LAM, WC, return_wfn=True )
    E    = do_QED_RHF( mol, LAM, WC, initial_guess=C )

    mol.atom = 'H 0 0 0; H 0 0 10.0'
    mol.build()
    E = do_QED_RHF( mol, LAM, WC )
    E = do_QED_RHF( mol, LAM, WC, initial_guess=C )
