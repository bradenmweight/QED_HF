import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from pyscf import gto, scf, fci

from openms import mqed

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
    overlap  = mol.intor('int1e_ovlp')
    s,u      = eigh(overlap) 
    Shalf    = u @ np.diag(1/np.sqrt(s)) @ u.T

    # Get nuclear repulsion energy
    nuclear_repulsion_energy = mol.energy_nuc()

    # Get number of atomic orbitals
    n_ao = overlap.shape[0]

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

    return Shalf, h1e, eri, dip_ao, quad_ao, n_elec_alpha, nuclear_repulsion_energy

def do_QED_RHF( mol, LAM, WC, doCS=True ):

    Shalf, h1e, eri, dip_ao, quad_ao, n_elec_alpha, nuclear_repulsion_energy = get_ao_integrals( mol )

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
    maxiter       = 500

    old_energy = 0.5 * np.einsum("ab,ab->", D, 2*h1e ) + nuclear_repulsion_energy
    old_D = D.copy()

    #print("    Guess Energy: %20.12f" % old_energy)
    for iter in range( maxiter ):

        # DSE
        if ( doCS == True ):
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

        # Transfom Fock matrix to orthogonal basis
        F_ORTHO = to_ortho_ao( Shalf, F, shape=2 )

        # Diagonalize Fock matrix
        eps, C = eigh( F_ORTHO )

        # Rotate MOs back to non-orthogonal AO basis
        C = from_ortho_ao( Shalf, C, shape=1 )

        # Get density matrix in AO basis
        D = make_RDM1_ao( C, n_elec_alpha )

        # Get current energy for RHF
        energy  = np.einsum("ab,ab->", D, 2*h1e + 2*J - K ) + nuclear_repulsion_energy
        energy += np.einsum("ab,ab->", D, 2*h1e_DSE + 2*DSE_J - DSE_K )
        energy += 0* DSE_FACTOR*AVEdipole**2 + 0.5 * WC

        dE = energy - old_energy
        dD = np.linalg.norm( D - old_D )

        old_energy = energy
        old_D      = D.copy()

        if ( iter > 2 and abs(dE) < e_convergence and dD < d_convergence ):
            break
        if ( iter > maxiter ):
            print("FAILURE: QED-HF (Braden) DID NOT CONVERGE")
            break

    myRHF   = scf.RHF( mol )
    e_qedhf = get_QED_HF_ZHY( mol, LAM, WC )
    e_rhf   = myRHF.kernel() + 0.5 * WC
    e_fci   = fci.FCI( myRHF ).kernel()[0] + 0.5 * WC
    #print('    * FCI Total Energy (PySCF): %20.12f' % (e_fci))
    #print('    * RHF Total Energy (PySCF) : %20.12f' % (e_rhf))
    print('    * RHF Total Energy (Braden): %20.12f' % (energy))
    #print('    * RHF Wavefunction:', np.round( C[:,0],3))

    return energy, e_qedhf, e_rhf, e_fci


if (__name__ == '__main__' ):
    mol = gto.Mole()
    mol.basis = 'ccpvdz'
    mol.unit = 'Bohr'
    mol.symmetry = False

    dL           = 0.05
    LAM_LIST     = np.arange(0.0, 0.2+dL, dL)
    #R_LIST   = np.arange(0.75, 10.25, 0.25)
    R_LIST   = np.arange(2, 7.25, 0.25)
    QEDHF_BRADEN = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDHF_ZHY    = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    EHF_PYSCF    = np.zeros( (len(R_LIST)) )
    EFCI_PYSCF   = np.zeros( (len(R_LIST)) )
    for LAMi,LAM in enumerate( LAM_LIST ):
        for Ri,R in enumerate( R_LIST ):
            print("Working on R = %1.2f" % R)
            mol.atom = 'Li 0 0 0; H 0 0 %1.8f' % R
            mol.build()
            QEDHF_BRADEN[LAMi,Ri], QEDHF_ZHY[LAMi,Ri], EHF_PYSCF[Ri], EFCI_PYSCF[Ri] = do_QED_RHF( mol, LAM, 0.1 )
    
    plt.plot( R_LIST, EHF_PYSCF, c='black', lw=8, alpha=0.5, label="RHF (PySCF) + $\\frac{\\hbar \\omega}{2}$" )
    plt.plot( R_LIST, EFCI_PYSCF, c='blue', lw=8, alpha=0.5, label="FCI (PySCF) + $\\frac{\\hbar \\omega}{2}$" )
    for LAMi,LAM in enumerate( LAM_LIST ):
        plt.plot( R_LIST, QEDHF_ZHY[LAMi,:], "o", c="black", label="QED-RHF (Yu)" * (LAMi==0) )
        plt.plot( R_LIST, QEDHF_BRADEN[LAMi,:], "-", c="red", label="QED-RHF (Braden)" * (LAMi==0) )
    plt.xlabel("Nuclear Separation, $R$ (a.u.)", fontsize=15)
    plt.ylabel("Energy, $E_0$ (a.u.)", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig("H2_dissociation_curve.jpg", dpi=300)
    plt.clf()

    exit()



    dL           = 0.1
    LAM_LIST     = np.arange(0.0, 1+dL, dL)
    QEDHF_BRADEN = np.zeros_like(LAM_LIST)
    QEDHF_ZHY    = np.zeros_like(LAM_LIST)
    RHH          = 2.8
    WC           = 0.1
    #mol.atom     = 'H 0 0 0; H 0 0 %1.3f' % ( RHH )
    mol.atom     = 'Li 0 0 %1.3f; H 0 0 %1.3f' % ( -3/4*RHH, 1/4*RHH )
    #mol.atom     = 'Li 0 0 %1.3f; Li 0 0 %1.3f' % ( -2, 2 )
    mol.build()

    for LAMi,LAM in enumerate( LAM_LIST ):
        print("Working on LAM = %1.2f" % LAM)
        QEDHF_BRADEN[LAMi], QEDHF_ZHY[LAMi], e_rhf, e_fci = do_QED_RHF( mol, LAM, WC )
    
    plt.plot( LAM_LIST, LAM_LIST*0 + e_rhf, "-", c='black', lw=5, alpha=0.5, label="RHF (PySCF) + $\\frac{\\hbar \\omega}{2}$" )
    # plt.plot( LAM_LIST, LAM_LIST*0 + e_fci, "-", c='blue', lw=5, alpha=0.5, label="FCI (PySCF) + $\\frac{\\hbar \\omega}{2}$" )
    plt.plot( LAM_LIST, QEDHF_BRADEN, "-", c='black', label="QED-RHF (Braden)" )
    plt.plot( LAM_LIST, QEDHF_ZHY, "o", c='black', label="QED-RHF (Yu)" )
    # plt.plot( LAM_LIST, QEDHF_BRADEN - QEDHF_ZHY, "-", c='black', label="QED-RHF Yu" )
    plt.legend()
    plt.xlabel("Coupling Strength, $\\lambda$ (a.u.)", fontsize=15)
    plt.ylabel("Energy, $E_0$ (a.u.)", fontsize=15)
    plt.tight_layout()
    plt.savefig("H2_LAM_SCAN.jpg", dpi=300)
    plt.clf()
