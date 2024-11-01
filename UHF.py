import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf, fci

from tools import to_ortho_ao, from_ortho_ao, eigh, make_RDM1_ao, get_spin_analysis, do_DAMP
from ao_ints import get_ao_integrals, get_dipole_quadrupole
from DIIS import DIIS

def do_UHF( mol ):

    # Get ao integrals
    S, Shalf, h1e, eri, n_elec_alpha, n_elec_beta, nuclear_repulsion_energy = get_ao_integrals( mol )

    # Choose core as guess for Fock
    F_a = h1e

    # Rotate Fock to orthogonal basis
    F_ORTHO_a = to_ortho_ao( Shalf, F_a, shape=2 )

    # Diagonalize Fock
    eps_a, C_a = np.linalg.eigh( F_ORTHO_a )
    
    # Rotate all MOs back to non-orthogonal AO basis
    C_a = from_ortho_ao( Shalf, C_a, shape=1 )
    C_b = C_a.copy()

    # Get density matrix in AO basis C, n_elec_alpha
    D_a      = make_RDM1_ao(C_a, n_elec_alpha)
    D_b      = make_RDM1_ao(C_b, n_elec_beta )

    e_convergence = 1e-8
    d_convergence = 1e-6
    maxiter       = 200

    old_energy  = np.einsum("ab,ab->", D_a, 2*h1e )
    old_energy += nuclear_repulsion_energy
    old_D_a     = D_a.copy()
    old_D_b     = D_b.copy()
    old_F_a     = F_a.copy()
    old_F_b     = F_a.copy()

    myDIIS = DIIS( unrestricted=True, ao_overlap=S )

    for iter in range( maxiter ):

        # Coulomb matrix
        J_a = np.einsum( 'rs,pqrs->pq', D_a, eri )
        J_b = np.einsum( 'rs,pqrs->pq', D_b, eri )

        # Exchange matrix
        K_a = np.einsum( 'rs,prsq->pq', D_a, eri )
        K_b = np.einsum( 'rs,prsq->pq', D_b, eri )

        # Fock matrix for RHF
        F_a = h1e + J_a + J_b - K_a
        F_b = h1e + J_b + J_a - K_b

        if ( iter < 5 ):
            F = do_DAMP( F_a, old_F_a )
            F = do_DAMP( F_b, old_F_b )

        if ( iter > 2 and iter < 10 ):
            F_a, F_b = myDIIS.extrapolate( F_a, D_a, F_b=F_b, D_b=D_b )

        # Transfom Fock matrix to orthogonal basis
        F_ORTHO_a = Shalf.T @ F_a @ Shalf
        F_ORTHO_b = Shalf.T @ F_b @ Shalf

        # Diagonalize Fock matrix
        eps_a, C_a = np.linalg.eigh( F_ORTHO_a )
        eps_b, C_b = np.linalg.eigh( F_ORTHO_b )

        if ( iter == 5 ):
            # Break symmetry by mixing a-HOMO and a-LUMO
            C_b    = C_a.copy()
            angle  = np.pi/5
            HOMO_a = C_a[:,n_elec_alpha-1]
            LUMO_a = C_a[:,n_elec_alpha+0]
            C_a[:,n_elec_alpha-1] = HOMO_a * np.cos(angle) + LUMO_a * np.sin(angle)
            C_b[:,n_elec_beta-1]  = HOMO_a * np.cos(angle) - LUMO_a * np.sin(angle)

        # Rotate MOs back to non-orthogonal AO basis
        C_a = Shalf @ C_a
        C_b = Shalf @ C_b

        # Get density matrix in AO basis
        D_a  = np.einsum("ai,bi->ab", C_a[:,:n_elec_alpha], C_a[:,:n_elec_alpha])
        D_b  = np.einsum("ai,bi->ab", C_b[:,:n_elec_beta], C_b[:,:n_elec_beta])

        # Get current energy for RHF
        energy  = np.einsum("ab,ab->", D_a, h1e + 0.5*(J_a + J_b - K_a) )
        energy += np.einsum("ab,ab->", D_b, h1e + 0.5*(J_b + J_a - K_b) )
        energy += nuclear_repulsion_energy


        dE = energy - old_energy
        dD = np.linalg.norm( D_a - old_D_a ) + np.linalg.norm( D_b - old_D_b )

        if ( iter > 50 ):
            print ("    Iteration %d  Energy = %1.6f  dE = %1.6f" % (iter, energy, dE))


        old_energy   = energy
        old_D_a      = D_a.copy()
        old_D_b      = D_b.copy()

        if ( iter > 2 and abs(dE) < e_convergence and dD < d_convergence ):
            break
        if ( iter == maxiter-1 ):
            print("FAILURE: QED-UHF DID NOT CONVERGE")
            break

    # Compute spin operators
    S2, ss1 = get_spin_analysis( C_a[:,:n_elec_alpha], C_b[:,:n_elec_beta], ao_overlap=S )
    print( "Spin Analsysis of UHF Wavefunction:" )
    print( "\t<S2>                = %1.4f" % (S2) )
    print( "\tMultiplicity s(s+1) = %1.4f" % (ss1) )

    # # RHF
    # myRHF = scf.RHF( mol )
    # e_rhf = myRHF.kernel()
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
    #     e_uhf = mf1.e_tot
    # else:
    #     e_uhf = mf2.e_tot

    # # Full configuration interaction
    # e_fci = fci.FCI( myRHF ).kernel()[0]
    # print('    * FCI Total Energy (PySCF): %20.12f' % (e_fci))
    # print('    * RHF Total Energy (PySCF) : %20.12f' % (e_rhf))
    print('\tUHF Total Energy:    %1.8f' % (energy))
    # print('    * RHF Wavefunction:', np.round( C_a[:,0],3))
    # print('    * RHF Wavefunction:', np.round( C_b[:,0],3))
    
    
    return energy, S2, ss1 #, e_rhf, e_uhf, e_fci

if (__name__ == '__main__' ):

    mol = gto.Mole()
    mol.basis = "ccpvdz"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'H 0 0 0; H 0 0 6'
    mol.build()
    E_UHF, S2, ss1 = do_UHF( mol )