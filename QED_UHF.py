import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from pyscf import gto, scf, fci

from openms import mqed

from DIIS import DIIS

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


def get_QED_HF_ZHY( mol, LAM, WC ):
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


def get_dipole_quadrupole( mol, n_ao ):
    # Get dipole matrix elements in AO basis with nuclear contribution
    charges    = mol.atom_charges()
    coords     = mol.atom_coords()
    nuc_dipole = np.einsum("a,ad->d", charges, coords) / charges.sum()
    with mol.with_common_orig(nuc_dipole):
        dipole_ao  = mol.intor_symmetric("int1e_r", comp=3)

    # Get quadrupole matrix elements in AO basis
    with mol.with_common_orig(nuc_dipole):
        quadrupole_ao  = mol.intor_symmetric("int1e_rr", comp=9)#.reshape(3,3,n_ao,n_ao)
    quadrupole_ao = quadrupole_ao.reshape(3,3,n_ao,n_ao)

    return dipole_ao, quadrupole_ao

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

def make_RDM1_ao_einsum( C, n_elec_alpha ):
    return np.einsum("ai,bi->ab", C[:,:n_elec_alpha], C[:,:n_elec_alpha] )

@njit
def make_RDM1_ao( C, n_elec_alpha ):
    D = np.zeros( (len(C),len(C)) )
    for a in range( len(C) ):
        for b in range( len(C) ):
            D[a,b] = np.sum(C[a,:n_elec_alpha] * C[b,:n_elec_alpha] )
    return D

def get_ao_integrals( mol ):

    # Get overlap matrix and orthogonalizing transformation matrix
    S        = mol.intor('int1e_ovlp')
    s,u      = eigh(S) 
    Shalf    = u @ np.diag(1/np.sqrt(s)) @ u.T

    # Get nuclear repulsion energy
    nuclear_repulsion_energy = mol.energy_nuc()

    # Get number of atomic orbitals
    n_ao = S.shape[0]

    # Get number of electrons
    n_elec_alpha, n_elec_beta = mol.nelec

    # Get kinetic energy matrix
    T_AO = mol.intor('int1e_kin')

    # Get electron-nuclear matrix
    V_en = mol.intor('int1e_nuc')

    # Get electron-electron repulsion matrix
    eri = mol.intor('int2e', aosym='s1' ) # Symmetry is turned off to get all possible integrals, (NAO,NAO,NAO,NAO)

    # Get dipole and quadrupole integrals
    dip_ao, quad_ao = get_dipole_quadrupole( mol, n_ao )

    # Construct core electronic Hamiltonian
    h1e = T_AO + V_en

    return S, Shalf, h1e, eri, dip_ao, quad_ao, n_elec_alpha, n_elec_beta, n_ao, nuclear_repulsion_energy

def do_QED_UHF( mol, LAM, WC, do_coherent_state=True ):

    S, Shalf, h1e, eri, dip_ao, quad_ao, n_elec_alpha, n_elec_beta, n_ao, nuclear_repulsion_energy = get_ao_integrals( mol )

    # Choose core as guess for Fock matrix
    F_a = h1e

    # Rotate Fock matrix to orthogonal ao basis
    F_ORTHO_a = to_ortho_ao( Shalf, F_a, shape=2 )

    # Diagonalize Fock
    eps_a, C_a = eigh( F_ORTHO_a )

    # Rotate all MOs back to non-orthogonal AO basis
    C_a = from_ortho_ao( Shalf, C_a, shape=1 )
    C_b = C_a.copy()

    # Get density matrix in AO basis
    D_a = make_RDM1_ao( C_a, n_elec_alpha )
    D_b = make_RDM1_ao( C_b, n_elec_beta )

    e_convergence = 1e-8
    d_convergence = 1e-6
    maxiter       = 200
    doDAMP        = True
    DAMP          = 0.5

    old_energy  = np.einsum("ab,ab->", D_a, h1e )
    old_energy += np.einsum("ab,ab->", D_b, h1e )
    old_energy += nuclear_repulsion_energy
    old_D_a = D_a.copy()
    old_D_b = D_b.copy()
    old_F_a = F_a.copy()
    old_F_b = F_a.copy()

    myDIIS = DIIS( ao_overlap=S, unrestricted=True, N_DIIS=5 )

    #print("    Guess Energy: %20.12f" % old_energy)
    for iter in range( maxiter ):

        # DSE
        if ( do_coherent_state == True ):
            AVEdipole = np.einsum( 'pq,pq->', D_a + D_b, dip_ao[-1,:,:] )
        else:
            AVEdipole = 0.0
        
        DSE_FACTOR = 0.5 * LAM**2
        h1e_DSE =     DSE_FACTOR * ( -2*AVEdipole * dip_ao[-1,:,:] + quad_ao[-1,-1,:,:] ) 
        eri_DSE = 2 * DSE_FACTOR  * np.einsum( 'pq,rs->pqrs', dip_ao[-1,:,:], dip_ao[-1,:,:] )

        # Coulomb matrix
        J_a     = np.einsum( 'rs,pqrs->pq', D_a, eri )
        J_b     = np.einsum( 'rs,pqrs->pq', D_b, eri )
        DSE_J_a = np.einsum( 'rs,pqrs->pq', D_a, eri_DSE )
        DSE_J_b = np.einsum( 'rs,pqrs->pq', D_b, eri_DSE )

        # Exchange matrix
        K_a     = np.einsum( 'rs,prsq->pq', D_a, eri )
        K_b     = np.einsum( 'rs,prsq->pq', D_b, eri )
        DSE_K_a = np.einsum( 'rs,prsq->pq', D_a, eri_DSE )
        DSE_K_b = np.einsum( 'rs,prsq->pq', D_b, eri_DSE )

        # Fock matrix
        F_a  = h1e     + J_a + J_b - K_a
        F_a += h1e_DSE + DSE_J_a + DSE_J_b - DSE_K_a
        F_b  = h1e     + J_a + J_b - K_b
        F_b += h1e_DSE + DSE_J_a + DSE_J_b - DSE_K_b

        if ( doDAMP == True ): # Do before DIIS, else DIIS will give singular B matrix
            F_a = DAMP * F_a + (1-DAMP) * old_F_a
            F_b = DAMP * F_b + (1-DAMP) * old_F_b

        if ( iter > 2 ):
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

        # Get density matrix in AO basis
        D_a = make_RDM1_ao( C_a, n_elec_alpha )
        D_b = make_RDM1_ao( C_b, n_elec_beta )

        # Get current energy for RHF
        energy  = np.einsum("ab,ab->", D_a, h1e + 0.5*(J_a + J_b - K_a) )
        energy += np.einsum("ab,ab->", D_b, h1e + 0.5*(J_b + J_a - K_b) )
        energy += np.einsum("ab,ab->", D_a, h1e_DSE + 0.5*(DSE_J_a + DSE_J_b - DSE_K_a) )
        energy += np.einsum("ab,ab->", D_b, h1e_DSE + 0.5*(DSE_J_a + DSE_J_b - DSE_K_b) )
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

        print("    Iteration %3d: Energy = %20.12f, dE = %1.5e, dD = %1.5e" % (iter, energy, dE, dD))

        if ( iter > 2 and abs(dE) < e_convergence and dD < d_convergence ):
            break
        if ( iter == maxiter-1 ):
            print("FAILURE: QED-UHF (Braden) DID NOT CONVERGE")
            break

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

    return energy, S2, ss1 #, e_qedhf, e_rhf, e_uhf, e_fci


if (__name__ == '__main__' ):

    mol = gto.Mole()
    mol.basis = "ccpvdz"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'Li 0 0 0; H 0 0 8.0'
    mol.build()
    E_UHF, S2, ss1 = do_QED_UHF( mol, 0.2, 0.1 )
