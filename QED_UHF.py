import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from pyscf import gto, scf, fci

from ao_ints import get_ao_integrals
from tools import get_JK, eigh, make_RDM1_ao, get_spin_analysis, do_DAMP, do_Max_Overlap_Method
from DIIS import DIIS

def do_QED_UHF( mol, LAM, WC, do_CS=True, return_wfn=False, initial_guess=None ):
    DSE_FACTOR = 0.5 * LAM**2

    h1e, eri, n_elec_alpha, n_elec_beta, nuclear_repulsion_energy, dip_ao, quad_ao = get_ao_integrals( mol, dipole_quadrupole=True )

    if ( initial_guess is not None ):
        C = initial_guess
        if ( len(C) == 2 ):
            D_a     = make_RDM1_ao( C[0], n_elec_alpha )
            D_b     = make_RDM1_ao( C[1], n_elec_beta )
        else:
            D_a     = make_RDM1_ao( C, n_elec_alpha )
            D_b     = make_RDM1_ao( C, n_elec_beta )
        AVEdipole         = (do_CS) * np.einsum( 'pq,pq->', D_a + D_b, dip_ao )
        h1e_DSE           =     DSE_FACTOR * ( -2*AVEdipole * dip_ao + quad_ao ) 
        eri_DSE           = 2 * DSE_FACTOR  * np.einsum( 'pq,rs->pqrs', dip_ao, dip_ao )
        J_a, K_a          = get_JK( D_a, eri )
        J_b, K_b          = get_JK( D_b, eri )
        DSE_J_a, DSE_K_a  = get_JK( D_a, eri_DSE )
        DSE_J_b, DSE_K_b  = get_JK( D_b, eri_DSE )
        F_a               = h1e     + J_a + J_b - K_a
        F_a              += h1e_DSE + DSE_J_a + DSE_J_b - DSE_K_a
        F_b               = h1e     + J_a + J_b - K_b
        F_b              += h1e_DSE + DSE_J_a + DSE_J_b - DSE_K_b
        eps_a, C_a        = eigh( F_a )
        eps_b, C_b        = eigh( F_b )
        D_a               = make_RDM1_ao( C_a, n_elec_alpha )
        D_b               = make_RDM1_ao( C_b, n_elec_beta )
        old_energy        = np.einsum("ab,ab->", D_a, h1e + 0.5*(J_a + J_b - K_a) )
        old_energy       += np.einsum("ab,ab->", D_b, h1e + 0.5*(J_b + J_a - K_b) )
        old_energy       += np.einsum("ab,ab->", D_a, h1e_DSE + 0.5*(DSE_J_a + DSE_J_b - DSE_K_a) )
        old_energy       += np.einsum("ab,ab->", D_b, h1e_DSE + 0.5*(DSE_J_b + DSE_J_a - DSE_K_b) )
        old_energy       += nuclear_repulsion_energy
        old_energy       += 0.5 * WC
    else:
        F_a         = h1e
        F_b         = h1e
        eps_a, C_a  = np.linalg.eigh( F_a )
        eps_b, C_b  = np.linalg.eigh( F_b )
        D_a         = make_RDM1_ao( C_a, n_elec_alpha )
        D_b         = make_RDM1_ao( C_b, n_elec_beta )
        old_energy  = np.einsum("ab,ab->", D_a, h1e )
        old_energy += np.einsum("ab,ab->", D_b, h1e )
        old_energy += nuclear_repulsion_energy

    e_convergence = 1e-8
    d_convergence = 1e-6
    maxiter       = 2000

    old_C_a = C_a.copy()
    old_C_b = C_b.copy()
    old_D_a = D_a.copy()
    old_D_b = D_b.copy()
    old_F_a = F_a.copy()
    old_F_b = F_a.copy()
    old_dE  = 0.0
    old_dD  = 0.0

    myDIIS_a = DIIS()
    myDIIS_b = DIIS()

    for iter in range( maxiter ):

        AVEdipole = (do_CS) * np.einsum( 'pq,pq->', D_a + D_b, dip_ao[:,:] )
        h1e_DSE    =     DSE_FACTOR * ( -2*AVEdipole * dip_ao[:,:] + quad_ao[:,:] ) 
        eri_DSE    = 2 * DSE_FACTOR * np.einsum( 'pq,rs->pqrs', dip_ao[:,:], dip_ao[:,:] )

        # Coulomb and Exchange Matrices
        J_a, K_a         = get_JK( D_a, eri )
        J_b, K_b         = get_JK( D_b, eri )
        DSE_J_a, DSE_K_a = get_JK( D_a, eri_DSE )
        DSE_J_b, DSE_K_b = get_JK( D_b, eri_DSE )

        # Fock matrix
        F_a  = h1e     + J_a + J_b - K_a
        F_b  = h1e     + J_a + J_b - K_b
        F_a += h1e_DSE + DSE_J_a + DSE_J_b - DSE_K_a
        F_b += h1e_DSE + DSE_J_a + DSE_J_b - DSE_K_b
        
        F_a = do_DAMP( F_a, old_F_a )
        F_b = do_DAMP( F_b, old_F_b )

        if ( iter > 10 ):
            F_a = myDIIS_a.extrapolate( F_a, D_a )
            F_b = myDIIS_b.extrapolate( F_b, D_b )

        # Diagonalize Fock matrix
        eps_a, C_a = eigh( F_a )
        eps_b, C_b = eigh( F_b )

        if ( iter == 5 ):
            # Break symmetry by mixing a-HOMO and a-LUMO
            C_b    = C_a.copy()
            angle   = np.pi/5
            HOMO_a = C_a[:,n_elec_alpha-1]
            LUMO_a = C_a[:,n_elec_alpha+0]
            C_a[:,n_elec_alpha-1] = HOMO_a * np.cos(angle) + LUMO_a * np.sin(angle)
            C_b[:,n_elec_beta-1]  = HOMO_a * np.cos(angle) - LUMO_a * np.sin(angle)

        # Get density matrix in AO basis
        D_a = make_RDM1_ao( C_a, (np.arange(n_elec_alpha)) )
        D_b = make_RDM1_ao( C_b, (np.arange(n_elec_beta)) )

        # Get current energy for RHF
        energy  = np.einsum("ab,ab->", D_a, h1e + 0.5*(J_a + J_b - K_a) )
        energy += np.einsum("ab,ab->", D_b, h1e + 0.5*(J_b + J_a - K_b) )
        energy += np.einsum("ab,ab->", D_a, h1e_DSE + 0.5*(DSE_J_a + DSE_J_b - DSE_K_a) )
        energy += np.einsum("ab,ab->", D_b, h1e_DSE + 0.5*(DSE_J_b + DSE_J_a - DSE_K_b) )
        energy += nuclear_repulsion_energy
        #energy += DSE_FACTOR*AVEdipole**2
        energy += 0.5 * WC

        dE = energy - old_energy
        dD = np.linalg.norm( D_a - old_D_a ) + np.linalg.norm( D_b - old_D_b )

        # if ( iter > 5 and dD > 1.0 ):            
        #    inds_a = do_Max_Overlap_Method( C_a, old_C_a, (np.arange(n_elec_alpha)) )
        #    inds_b = do_Max_Overlap_Method( C_b, old_C_b, (np.arange(n_elec_beta)) )
        #    C_a  = C_a[:,inds_a]
        #    C_b  = C_b[:,inds_b]
        #    D_a  = make_RDM1_ao( C_a, (np.arange(n_elec_alpha)) )
        #    D_b  = make_RDM1_ao( C_b, (np.arange(n_elec_beta)) )
        #    dD   = 2 * (np.linalg.norm( D_a - old_D_a ) + np.linalg.norm( D_b - old_D_b ))

        old_energy = energy
        old_D_a    = D_a.copy()
        old_D_b    = D_b.copy()
        old_F_a    = F_a.copy()
        old_F_b    = F_b.copy()
        old_dE     = dE*1
        old_dD     = dD*1

        print("    QED-UHF Iteration %3d: Energy = %1.12f, dE = %1.8f, dD = %1.6f" % (iter, energy, dE, dD))

        if ( iter > 2 and abs(dE) < e_convergence and dD < d_convergence ):
            break
        if ( iter == maxiter-1 ):
            print("FAILURE: QED-UHF DID NOT CONVERGE")
            return float('nan'), float('nan'), float('nan')
        


    # Compute spin operators
    S2, ss1 = get_spin_analysis( C_a[:,:n_elec_alpha], C_b[:,:n_elec_beta] )
    print( "Spin Analsysis of UHF Wavefunction:" )
    print( "\t<S2>                = %1.4f" % (S2) )
    print( "\tMultiplicity s(s+1) = %1.4f" % (ss1) )

    print('    * QED-UHF Total Energy: %1.8f' % (energy))
    #print('    * RHF Wavefunction:', np.round( C[:,0],3))

    if ( return_wfn == True ):
        return energy, S2, ss1, np.array([C_a, C_b])
    else:
        return energy, S2, ss1


if (__name__ == '__main__' ):

    mol = gto.Mole()
    mol.basis = "ccpvdz"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'H 0 0 0; H 0 0 2.0'
    mol.build()
    LAM = 0.1
    WC  = 1.0
    E, S2, ss1    = do_QED_UHF( mol, LAM, WC )
    E, S2, ss1, C = do_QED_UHF( mol, LAM, WC, return_wfn=True )
    E, S2, ss1    = do_QED_UHF( mol, LAM, WC, initial_guess=C )

    mol.atom = 'H 0 0 0; H 0 0 10.0'
    mol.build()
    E, S2, ss1    = do_QED_UHF( mol, LAM, WC )
    E, S2, ss1    = do_QED_UHF( mol, LAM, WC, initial_guess=C )
