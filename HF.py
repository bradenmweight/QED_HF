import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf, fci



def do_RHF( mol ):

    # Get overlap matrix and orthogonalizing transformation matrix
    overlap  = mol.intor('int1e_ovlp')
    s,u      = np.linalg.eigh(overlap) 
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

    # # Get dipole matrix elements in AO basis with nuclear contribution
    # charges    = mol.atom_charges()
    # coords     = mol.atom_coords()
    # nuc_dipole = np.einsum("a,ad->d", charges, coords) #/ charges.sum()
    # dipole_ao  = mol.intor_symmetric("int1e_r", comp=3)
    # dipole_ao  = np.array([-1*dipole_ao[d,:,:] + nuc_dipole[d]*np.eye(n_ao) for d in range(3)])

    # # Get quadrupole matrix elements in AO basis
    # quadrupole_ao  = mol.intor_symmetric("int1e_rr", comp=9).reshape(3,3,n_ao,n_ao)
    # nuc_quadrupole = np.einsum("a,ax,by,b->xy", charges, coords, coords, charges) #/ charges.sum()
    # quadrupole_ao  = np.array([-1*quadrupole_ao[x,y,:,:] + nuc_quadrupole[x,y]*np.eye(n_ao) for x in range(3) for y in range(3)])
    # quadrupole_ao  = quadrupole_ao.reshape(3,3,n_ao,n_ao)





    # Get core Hamiltonian
    h1e = T_AO + V_en

    # Choose core as guess for Fock
    F = h1e

    # Rotate Fock to orthogonal basis
    F_ORTHO = Shalf.T @ F @ Shalf

    # Diagonalize Fock
    eps, C = np.linalg.eigh( F_ORTHO )

    # Rotate all MOs back to non-orthogonal AO basis
    C = Shalf @ C

    # Get density matrix in AO basis
    D    = np.einsum( "ai,bi->ab", C[:,:n_elec_alpha], C[:,:n_elec_alpha] )

    e_convergence = 1e-8
    d_convergence = 1e-6
    maxiter       = 100

    old_energy = 0.5 * np.einsum("ab,ab->", D, h1e + F ) + nuclear_repulsion_energy
    old_D = D.copy()

    #print("    Guess Energy: %20.12f" % old_energy)
    for iter in range( maxiter ):

        # Coulomb matrix
        J = np.einsum( 'rs,pqrs->pq', D, eri )

        # Exchange matrix
        K = np.einsum( 'rs,psrq->pq', D, eri )
        #K = np.einsum( 'rs,prqs->pq', D, eri )

        # Fock matrix for RHF
        F = h1e + 2 * J - K

        # Transfom Fock matrix to orthogonal basis
        F_ORTHO = Shalf.T @ F @ Shalf

        # Diagonalize Fock matrix
        eps, C = np.linalg.eigh( F_ORTHO )

        # Rotate MOs back to non-orthogonal AO basis
        C = Shalf @ C

        # Get density matrix in AO basis
        D  = np.einsum("ai,bi->ab", C[:,:n_elec_alpha], C[:,:n_elec_alpha])

        # Get current energy for RHF
        # H = 0.5*(h + F) = h + 0.5 * (J - K) = h + 0.5 * V_ee
        energy = np.einsum("ab,ab->", D, 2*h1e + 2*J - K ) + nuclear_repulsion_energy

        dE = np.abs( energy - old_energy )
        dD = np.linalg.norm( D - old_D )

        #print( '    Iteration %3d: Energy = %4.12f, Energy change = %1.5e, Density change = %1.5e' % (iter, energy, dE, dD ) )

        old_energy = energy
        old_D      = D.copy()

        if ( iter > 2 and dE < e_convergence and dD < d_convergence ):
            #print('    SCF iterations converged!')
            break
        else :
            if ( iter > maxiter ):
                #print('    SCF iterations did not converge...')
                break

    myRHF = scf.RHF( mol )
    e_rhf = myRHF.kernel()
    # Full configuration interaction
    e_fci = fci.FCI( myRHF ).kernel()[0]
    print('    * FCI Total Energy (PySCF): %20.12f' % (e_fci))
    print('    * RHF Total Energy (PySCF) : %20.12f' % (e_rhf))
    print('    * RHF Total Energy (Braden): %20.12f' % (energy))
    #print('    * RHF Wavefunction:', np.round( C[:,0],3))
    return energy, e_rhf, e_fci


if (__name__ == '__main__' ):
    # From PySCF, get all required AO integrals
    mol = gto.Mole()
    mol.basis = 'ccpvtz'
    mol.unit = 'Bohr'
    mol.symmetry = False

    RHH_LIST   = np.arange(0.75, 10.05, 0.05)
    EHF_BRADEN = np.zeros_like(RHH_LIST)
    EHF_PYSCF  = np.zeros_like(RHH_LIST)
    EFCI_PYSCF  = np.zeros_like(RHH_LIST)
    for Ri,R in enumerate( RHH_LIST ):
        print("Working on R = %1.2f" % R)
        mol.atom = 'H 0 0 0; H 0 0 %1.8f' % R
        mol.build()
        EHF_BRADEN[Ri], EHF_PYSCF[Ri], EFCI_PYSCF[Ri] = do_RHF( mol )
    
    plt.plot( RHH_LIST, EHF_PYSCF, label="RHF PySCF" )
    plt.plot( RHH_LIST, EHF_BRADEN, "--", label="RHF Braden" )
    plt.plot( RHH_LIST, EFCI_PYSCF, label="FCI PySCF" )
    plt.legend()
    plt.savefig("H2_dissociation_curve.jpg", dpi=300)