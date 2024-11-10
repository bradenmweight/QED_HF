import numpy as np
from tools import eigh
import psutil
from opt_einsum import contract as opt_einsum # Very fast library for tensor contractions

def get_electric_dipole_ao( mol ):
    charges    = mol.atom_charges()
    coords     = mol.atom_coords()
    nuc_dipole = np.einsum("a,ad->d", charges, coords) / charges.sum()
    with mol.with_common_orig(nuc_dipole):
        dipole_ao  = mol.intor_symmetric("int1e_r", comp=3)
    return dipole_ao

def get_magnetic_dipole_ao( mol ):
    charges    = mol.atom_charges()
    coords     = mol.atom_coords()
    nuc_dipole = np.einsum("a,ad->d", charges, coords) / charges.sum()
    with mol.with_common_orig(nuc_dipole):
        dipole_ao  = mol.intor_symmetric("int1e_cg_irxp", comp=3)
    return dipole_ao

def get_electric_quadrupole_ao( mol ):
    charges    = mol.atom_charges()
    coords     = mol.atom_coords()
    nuc_dipole = np.einsum("a,ad->d", charges, coords) / charges.sum()
    with mol.with_common_orig(nuc_dipole):
        quadrupole_ao  = mol.intor_symmetric("int1e_rr", comp=9)
    quadrupole_ao  = quadrupole_ao.reshape(3,3,-1)
    n_ao = int(np.sqrt( quadrupole_ao.shape[-1] ) )
    return quadrupole_ao.reshape(3,3,n_ao,n_ao)




def get_ao_integrals( mol ):

    # Get overlap matrix and orthogonalizing transformation matrix
    S     = mol.intor('int1e_ovlp')
    s,u   = eigh(S) 
    Shalf = u @ np.diag(1/np.sqrt(s)) @ u.T

    # Get nuclear repulsion energy
    nuclear_repulsion_energy = mol.energy_nuc()

    # Get number of atomic orbitals
    n_ao = S.shape[0]

    eri_memory_size = 3 * n_ao**4 * 16 / 10**(9) # GB # Factor 3 for QED-versions
    avail_memory = psutil.virtual_memory().available / (1024 ** 3)
    if ( eri_memory_size > 2 ): 
        print("\tERI memory size: %1.6f GB of %1.3f GB available > 2.0" % (eri_memory_size, avail_memory) )
        print("\tExitting.")
        exit()
    elif ( eri_memory_size > 1 ):
        print("\tERI memory size ~ %1.3f GB of %1.3f GB available" % (eri_memory_size, avail_memory) )

    # Get number of electrons
    n_elec_alpha, n_elec_beta = mol.nelec

    # Get kinetic energy matrix
    T_AO = mol.intor('int1e_kin')

    # Get electron-nuclear matrix
    V_en = mol.intor('int1e_nuc')

    # Get electron-electron repulsion matrix
    eri = mol.intor('int2e', aosym='s1' ) # Symmetry is turned off to get all possible integrals, (NAO,NAO,NAO,NAO)

    # Construct core electronic Hamiltonian
    h1e = T_AO + V_en

    # Rotate all relevant integrals to the orthogonal AO basis
    h1e     = opt_einsum( 'ap,ab,bq->pq', Shalf, h1e, Shalf )
    eri     = opt_einsum( 'ap,bq,abcd,cr,ds->pqrs', Shalf, Shalf, eri, Shalf, Shalf ) # This can be expensive. Consider reverting back to rotating only Fock matrix
    
    return h1e, eri, n_elec_alpha, n_elec_beta, nuclear_repulsion_energy

