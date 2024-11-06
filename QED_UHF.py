import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from pyscf import gto, scf, fci

from tools import get_JK, to_ortho_ao, from_ortho_ao, eigh, make_RDM1_ao, get_spin_analysis, do_DAMP, do_Max_Overlap_Method
from ao_ints import get_ao_integrals, get_dipole_quadrupole
from DIIS import DIIS

def build_Fock_matrix( h1e, eri, D_a, D_b, dip_ao, quad_ao, do_CS=True, LAM=0.0 ):


    return F_a, F_b, h1e_DSE, J_a, J_b, DSE_J_a, DSE_J_b, K_a, K_b, DSE_K_a, DSE_K_b

def do_QED_UHF( mol, LAM, WC, do_CS=True, return_wfn=False, initial_guess=None ):

    S, Shalf, h1e, eri, n_elec_alpha, n_elec_beta, nuclear_repulsion_energy = get_ao_integrals( mol )
    dip_ao, quad_ao = get_dipole_quadrupole( mol, Shalf.shape[0] )

    if ( initial_guess is None ):
        # Choose core as guess for Fock matrix
        F_a = h1e
        F_ORTHO_a  = to_ortho_ao( Shalf, F_a, shape=2 )
        eps_a, C_a = eigh( F_ORTHO_a )
        C_a        = from_ortho_ao( Shalf, C_a, shape=1 )
        C_b        = C_a.copy()
        D_a        = make_RDM1_ao( C_a, n_elec_alpha )
        D_b        = make_RDM1_ao( C_b, n_elec_beta )
    else:
        # Use initial guess to construct the Fock matrix
        C_a, C_b   = initial_guess
        D_a        = make_RDM1_ao( C_a, n_elec_alpha )
        D_b        = make_RDM1_ao( C_b, n_elec_beta )
        if ( do_CS == True ):
            AVEdipole = np.einsum( 'pq,pq->', D_a + D_b, dip_ao[-1,:,:] )
        else:
            AVEdipole = 0.0
        
        DSE_FACTOR = 0.5 * LAM**2
        h1e_DSE =     DSE_FACTOR * ( -2*AVEdipole * dip_ao[-1,:,:] + quad_ao[-1,-1,:,:] ) 
        eri_DSE = 2 * DSE_FACTOR  * np.einsum( 'pq,rs->pqrs', dip_ao[-1,:,:], dip_ao[-1,:,:] )

        # Coulomb and Exchange Matrices
        J_a, K_a         = get_JK( D_a, eri )
        J_b, K_b         = get_JK( D_b, eri )
        DSE_J_a, DSE_K_a = get_JK( D_a, eri_DSE )
        DSE_J_b, DSE_K_b = get_JK( D_b, eri_DSE )

        # Fock matrix
        F_a  = h1e     + J_a + J_b - K_a
        F_a += h1e_DSE + DSE_J_a + DSE_J_b - DSE_K_a
        F_b  = h1e     + J_a + J_b - K_b
        F_b += h1e_DSE + DSE_J_a + DSE_J_b - DSE_K_b

        F_ORTHO_a  = to_ortho_ao( Shalf, F_a, shape=2 )
        F_ORTHO_b  = to_ortho_ao( Shalf, F_b, shape=2 )
        eps_a, C_a = eigh( F_ORTHO_a )
        eps_b, C_b = eigh( F_ORTHO_b )
        C_a        = from_ortho_ao( Shalf, C_a, shape=1 )
        C_b        = from_ortho_ao( Shalf, C_b, shape=1 )
        D_a        = make_RDM1_ao( C_a, n_elec_alpha )
        D_b        = make_RDM1_ao( C_b, n_elec_beta )

    e_convergence = 1e-8
    d_convergence = 1e-6
    maxiter       = 500

    old_energy  = np.einsum("ab,ab->", D_a, h1e )
    old_energy += np.einsum("ab,ab->", D_b, h1e )
    old_energy += nuclear_repulsion_energy
    old_C_a = C_a.copy()
    old_C_b = C_b.copy()
    old_D_a = D_a.copy()
    old_D_b = D_b.copy()
    old_F_a = F_a.copy()
    old_F_b = F_a.copy()
    old_dE  = 0.0
    old_dD  = 0.0
    DIIS_flag = False
    MOM_flag  = False

    myDIIS = DIIS( ao_overlap=S, unrestricted=True )

    #print("    Guess Energy: %20.12f" % old_energy)
    for iter in range( maxiter ):

        if ( do_CS == True ):
            AVEdipole = np.einsum( 'pq,pq->', D_a + D_b, dip_ao[-1,:,:] )
        else:
            AVEdipole = 0.0
        
        DSE_FACTOR = 0.5 * LAM**2
        h1e_DSE    =     DSE_FACTOR * ( -2*AVEdipole * dip_ao[-1,:,:] + quad_ao[-1,-1,:,:] ) 
        eri_DSE    = 2 * DSE_FACTOR * np.einsum( 'pq,rs->pqrs', dip_ao[-1,:,:], dip_ao[-1,:,:] )

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
        
        if ( iter < 5 ): # Do before DIIS
            F_a = do_DAMP( F_a, old_F_a )
            F_b = do_DAMP( F_b, old_F_b )

        if ( DIIS_flag == True ):
           F_a, F_b = myDIIS.extrapolate(F_a, D_a, F_b=F_b, D_b=D_b)

        # Transfom Fock matrix to orthogonal basis
        F_ORTHO_a = to_ortho_ao( Shalf, F_a, shape=2 )
        F_ORTHO_b = to_ortho_ao( Shalf, F_b, shape=2 )

        # Diagonalize Fock matrix
        eps_a, C_a = eigh( F_ORTHO_a )
        eps_b, C_b = eigh( F_ORTHO_b )

        if ( iter == 5 ):
            # Break symmetry by mixing a-HOMO and a-LUMO
            C_b    = C_a.copy()
            angle   = np.pi/5
            HOMO_a = C_a[:,n_elec_alpha-1]
            LUMO_a = C_a[:,n_elec_alpha+0]
            C_a[:,n_elec_alpha-1] = HOMO_a * np.cos(angle) + LUMO_a * np.sin(angle)
            C_b[:,n_elec_beta-1]  = HOMO_a * np.cos(angle) - LUMO_a * np.sin(angle)

        # Rotate MOs back to non-orthogonal AO basis
        C_a = from_ortho_ao( Shalf, C_a, shape=1 )
        C_b = from_ortho_ao( Shalf, C_b, shape=1 )

        if ( MOM_flag == True ):
            occ_inds_a = do_Max_Overlap_Method( C_a, old_C_a, S, n_elec_alpha )
            occ_inds_b = do_Max_Overlap_Method( C_b, old_C_b, S, n_elec_beta )
        else:
            occ_inds_a = (np.arange(n_elec_alpha))
            occ_inds_b = (np.arange(n_elec_beta))

        # Get density matrix in AO basis
        D_a = make_RDM1_ao( C_a, occ_inds_a )
        D_b = make_RDM1_ao( C_b, occ_inds_b )

        # Get current energy for RHF
        energy  = np.einsum("ab,ab->", D_a, h1e + 0.5*(J_a + J_b - K_a) )
        energy += np.einsum("ab,ab->", D_b, h1e + 0.5*(J_b + J_a - K_b) )
        energy += np.einsum("ab,ab->", D_a, h1e_DSE + 0.5*(DSE_J_a + DSE_J_b - DSE_K_a) )
        energy += np.einsum("ab,ab->", D_b, h1e_DSE + 0.5*(DSE_J_b + DSE_J_a - DSE_K_b) )
        energy += nuclear_repulsion_energy
        energy += DSE_FACTOR*AVEdipole**2
        energy += 0.5 * WC

        dE = energy - old_energy
        dD = np.linalg.norm( D_a - old_D_a ) + np.linalg.norm( D_b - old_D_b )

        old_energy = energy
        old_D_a    = D_a.copy()
        old_D_b    = D_b.copy()
        old_F_a    = F_a.copy()
        old_F_b    = F_b.copy()
        old_dE     = dE*1
        old_dD     = dD*1

        if ( iter > 50 ):
            print("    QED-UHF Iteration %3d: Energy = %1.12f, dE = %1.8f, dD = %1.6f" % (iter, energy, dE, dD))

        if ( iter > 2 and abs(dE) < e_convergence and dD < d_convergence ):
            break
        if ( iter == maxiter-1 ):
            print("FAILURE: QED-UHF DID NOT CONVERGE")
            return float('nan')
        
        if ( iter > 100 ): # Try doing DIIS
            DIIS_flag = True
        if ( iter > 150 ): # Try doing MOM
            MOM_flag = True
            DIIS_flag = False

    # Compute spin operators
    S2, ss1 = get_spin_analysis( C_a[:,:n_elec_alpha], C_b[:,:n_elec_beta], ao_overlap=S )
    print( "Spin Analsysis of UHF Wavefunction:" )
    print( "\t<S2>                = %1.4f" % (S2) )
    print( "\tMultiplicity s(s+1) = %1.4f" % (ss1) )

    # myRHF   = scf.RHF( mol )
    # e_qedhf = float('NaN') # get_QED_HF_ZHY( mol, LAM, WC )
    # e_rhf   = myRHF.kernel() + 0.5 * WC
    # # UHF -- BMW: Need to break symmetry of initial guess to get right solution
    # mf1 = scf.UHF(mol)
    # dm_alpha, dm_beta = mf1.get_init_guess()
    # dm_beta[:2,:2] = 0 # BMW: Set some of the beta coefficients to zero to break alpha/beta symmetry
    # dm = (dm_alpha,dm_beta)
    # mf1.kernel(dm) # BMW: Pass in modified initial guess
    # mf2 = scf.UHF(mol)
    # dm_alpha, dm_beta = mf2.get_init_guess()
    # dm_beta[:,:] = 0 # BMW: Set some of the beta coefficients to zero to break alpha/beta symmetry
    # dm = (dm_alpha,dm_beta)
    # mf2.kernel(dm) # BMW: Pass in modified initial guess
    # if ( mf1.e_tot < mf2.e_tot ): # BMW: Check which symmetry breaking works... H2 is mf1 but LiH is mf2
    #     e_uhf = mf1.e_tot + 0.5 * WC
    # else:
    #     e_uhf = mf2.e_tot + 0.5 * WC
    # e_fci   = fci.FCI( myRHF ).kernel()[0] + 0.5 * WC
    #print('    * FCI Total Energy (PySCF): %20.12f' % (e_fci))
    #print('    * RHF Total Energy (PySCF) : %20.12f' % (e_rhf))
    print('\tQED-UHF Total Energy: %1.8f' % (energy))
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
    mol.atom = 'Li 0 0 0; H 0 0 15.0'
    mol.build()
    E_UHF, S2, ss1 = do_QED_UHF( mol, 0.0, 0.0 )
