import numpy as np

from pyscf import gto, scf, tdscf

from ao_ints import get_electric_dipole_ao, get_magnetic_dipole_ao, get_electric_quadrupole_ao


# def get_cis_JJ_density_matrix(td, state_id):
#     '''
#     Taking the TDA amplitudes as the CIS coefficients, calculate the density
#     matrix (in AO basis) of the excited states
#     '''
#     # G-to-J transition density matrix (only TDA contribution -- X, Assume B=0)
#     cis_t1 = td.xy[state_id][0]
    
#     nocc, nvir = cis_t1.shape
    
#     dm_oo =-np.einsum('ia,ka->ik', cis_t1.conj(), cis_t1)
#     dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t1.conj())

#     # The ground state density matrix in mo_basis
#     mf = td._scf
#     dm = np.diag(mf.mo_occ)

#     # Add CIS contribution
#     nocc = cis_t1.shape[0]
#     dm[:nocc,:nocc] += dm_oo * 2
#     dm[nocc:,nocc:] += dm_vv * 2

#     # Transform density matrix to AO basis
#     mo = mf.mo_coeff
#     dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())
#     return dm

def get_cis_JK_density_matrix(td, state_id1, state_id2=None):
    '''
    Taking the TDA amplitudes as the CIS coefficients, calculate the density
    matrix (in AO basis) of the excited states
    '''
    
    # The ground state density matrix in mo_basis
    mf = td._scf
    dm = np.diag(mf.mo_occ)

    # G-to-J transition density matrix (only TDA contribution -- X, Assume B=0)
    cis_t1 = td.xy[state_id1][0]
    if ( state_id2 is not None ): cis_t2 = td.xy[state_id2][0]

    nocc, nvir = cis_t1.shape

    dm_oo1 =-np.einsum('ia,ka->ik', cis_t1.conj(), cis_t1)
    dm_vv1 = np.einsum('ia,ic->ac', cis_t1, cis_t1.conj())

    if ( state_id2 is not None ): 
        dm_oo2 =-np.einsum('ia,ka->ik', cis_t2.conj(), cis_t2)
        dm_vv2 = np.einsum('ia,ic->ac', cis_t2, cis_t2.conj())

    # Add CIS contribution to ground state (unperturbed dm)
    # Note: "* 2" comes from double occupation
    dm[:nocc,:nocc] += dm_oo1 * 2
    dm[nocc:,nocc:] += dm_vv1 * 2
    if ( state_id2 is not None ): 
        dm[:nocc,:nocc] += dm_oo2 * 2 # Should one of them be subtracted ??? ~BMW
        dm[nocc:,nocc:] += dm_vv2 * 2 # Should one of them be subtracted ??? ~BMW

    # Transform density matrix to AO basis
    mo = mf.mo_coeff
    dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())
    return dm



if ( __name__ == "__main__" ):


    # Example usage
    mol = gto.Mole()
    mol.atom = 'Li 0 0 0; H 0 0 1.4'
    mol.basis = '6-311G*'
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    td = tdscf.TDHF(mf)
    td.nstates = 4
    td.kernel()

    E_el = np.append( [mf.e_tot], mf.e_tot + td.e)

    dipole_ao              = get_electric_dipole_ao( mol )
    magnetic_dipole_ao     = get_magnetic_dipole_ao( mol )
    electric_quadrupole_ao = get_electric_quadrupole_ao( mol )

    dipole_matrix = np.zeros( (1+td.nstates, 1+td.nstates, 3) )
    # Excited State Dipoles
    for j in range(td.nstates):
        for k in range(td.nstates):
            dm = get_cis_JK_density_matrix(td, j, k)
            dipole_matrix[1+j, 1+k] = np.einsum('ab,dab->d', dm, dipole_ao)

    # Ground State Dipole
    dipole_matrix[0,0] = np.einsum( 'dab,ab->d', dipole_ao, mf.make_rdm1() )

    # Ground-to-excited Dipoles
    nocc, nvir = td.xy[0][0].shape
    for j in range(td.nstates):
        X                     = td.xy[j][0]
        Y                     = td.xy[j][1]
        X_ao                  = np.einsum('aj,jk,bk->ab', mf.mo_coeff.conj()[:,:nocc], X + Y, mf.mo_coeff[:,nocc:])
        dipole_matrix[0, 1+j] = 2 * np.einsum('ab,dab->d', X_ao, dipole_ao)
        dipole_matrix[1+j, 0] = dipole_matrix[0, 1+j]
    # transition_dipoles = td.transition_dipole()

    # print("Dipole matrix (X):\n", np.round( dipole_matrix[0,1:,0], 3))
    # print("PySCF Transition dipoles (X):\n", np.round( transition_dipoles[:,0], 3))
    # print("Dipole matrix (Y):\n", np.round( dipole_matrix[0,1:,1], 3))
    # print("PySCF Transition dipoles (Y):\n", np.round( transition_dipoles[:,1], 3))
    # print("Dipole matrix (Z):\n", np.round( dipole_matrix[0,1:,2], 3))
    # print("PySCF Transition dipoles (Z):\n", np.round( transition_dipoles[:,2], 3))

    # Build Pauli-Fierz Hamiltonian
    evec  = np.array([1,1,1])
    A0    = 0.01
    NFOCK = 2
    NEL   = 2 #1 + td.nstates
    NPOL  = NEL * NFOCK
    EMOL  = np.append( [mf.e_tot], mf.e_tot + td.e[:NEL-1] )
    print(EMOL - EMOL[0])
    WC    = EMOL[1] - EMOL[0]
    print( WC )
    dipole_matrix_projected = dipole_matrix[:NEL,:NEL,:].dot(evec)
    a_op  = np.diag( np.sqrt(np.arange(1,NFOCK)), k=1 )
    HPF   = np.zeros( (NPOL,NPOL) )
    HPF  += np.kron( np.diag(EMOL), np.eye(NFOCK) )
    HPF  += np.kron( np.eye(NEL), WC * np.diag(np.arange(NFOCK)) )
    HPF  += WC * A0 * np.kron( dipole_matrix_projected, a_op.T + a_op )
    HPF  += WC * A0**2 * np.kron(dipole_matrix_projected @ dipole_matrix_projected, np.eye(NFOCK) )
    E, U  = np.linalg.eigh( HPF )
    print( (E[:] - E[0]) )
