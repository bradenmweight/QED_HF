import numpy as np
from time import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh as sparse_eigsh

from pyscf import gto, scf, tdscf, lib
lib.num_threads(8)

from ao_ints import get_electric_dipole_ao, get_magnetic_dipole_ao, get_electric_quadrupole_ao

def get_cis_JJ_density_matrix(td, state_id): # THIS FUNCTION IS NOT USED
    '''
    Taking the TDA amplitudes as the CIS coefficients, calculate the density
    matrix (in AO basis) of the excited states
    '''
    # G-to-J transition density matrix (only TDA contribution -- X, Assume B=0)
    cis_t1 = td.xy[state_id][0]
    
    nocc, nvir = cis_t1.shape
    
    dm_oo =-np.einsum('ia,ka->ik', cis_t1.conj(), cis_t1)
    dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t1.conj())

    # The ground state density matrix in mo_basis
    mf = td._scf
    dm = np.diag(mf.mo_occ)

    # Add CIS contribution
    nocc = cis_t1.shape[0]
    dm[:nocc,:nocc] += dm_oo * 2
    dm[nocc:,nocc:] += dm_vv * 2

    # Transform density matrix to AO basis
    mo = mf.mo_coeff
    dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())
    return dm

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

    dm_oo1 =-np.einsum('ia,ka->ik', cis_t1.conj(), cis_t1, optimize=True)
    dm_vv1 = np.einsum('ia,ic->ac', cis_t1, cis_t1.conj(), optimize=True)

    if ( state_id2 is not None ): 
        dm_oo2 =-np.einsum('ia,ka->ik', cis_t2.conj(), cis_t2, optimize=True)
        dm_vv2 = np.einsum('ia,ic->ac', cis_t2, cis_t2.conj(), optimize=True)

    # Add CIS contribution to ground state (unperturbed dm)
    # Note: "* 2" comes from double occupation
    dm[:nocc,:nocc] += dm_oo1 * 2
    dm[nocc:,nocc:] += dm_vv1 * 2
    if ( state_id2 is not None ): 
        dm[:nocc,:nocc] += dm_oo2 * 2 # Should one of them be subtracted ??? ~BMW
        dm[nocc:,nocc:] += dm_vv2 * 2 # Should one of them be subtracted ??? ~BMW

    # Transform density matrix to AO basis
    mo = mf.mo_coeff
    dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj(), optimize=True)
    return dm

def build_transition_electric_dipole_matrix(mol, mf, td):
    electric_dipole_ao     = get_electric_dipole_ao( mol )
    # magnetic_dipole_ao     = get_magnetic_dipole_ao( mol )
    # electric_quadrupole_ao = get_electric_quadrupole_ao( mol )

    dipole_matrix = np.zeros( (1+td.nstates, 1+td.nstates, 3) )
    # Excited State Dipoles
    for j in range(td.nstates):
        for k in range(td.nstates):
            dm = get_cis_JK_density_matrix(td, j, k)
            dipole_matrix[1+j, 1+k] = np.einsum('ab,dab->d', dm, electric_dipole_ao, optimize=True)

    # Ground State Dipole
    dipole_matrix[0,0] = np.einsum( 'dab,ab->d', electric_dipole_ao, mf.make_rdm1(), optimize=True )

    # Ground-to-excited Dipoles
    nocc, nvir = td.xy[0][0].shape
    for j in range(td.nstates):
        X                     = td.xy[j][0]
        Y                     = td.xy[j][1]
        X_ao                  = np.einsum('aj,jk,bk->ab', mf.mo_coeff.conj()[:,:nocc], X + Y, mf.mo_coeff[:,nocc:], optimize=True)
        dipole_matrix[0, 1+j] = 2 * np.einsum('ab,dab->d', X_ao, electric_dipole_ao, optimize=True)
        dipole_matrix[1+j, 0] = dipole_matrix[0, 1+j]
    # transition_dipoles = td.transition_dipole()
    # print("Dipole matrix (X):\n", np.round( dipole_matrix[0,1:,0], 3))
    # print("PySCF Transition dipoles (X):\n", np.round( transition_dipoles[:,0], 3))
    # print("Dipole matrix (Y):\n", np.round( dipole_matrix[0,1:,1], 3))
    # print("PySCF Transition dipoles (Y):\n", np.round( transition_dipoles[:,1], 3))
    # print("Dipole matrix (Z):\n", np.round( dipole_matrix[0,1:,2], 3))
    # print("PySCF Transition dipoles (Z):\n", np.round( transition_dipoles[:,2], 3))

    return dipole_matrix

def build_Pauli_Fierz_Hamiltonian( mol, mf, td, evec, NEL, dipole_matrix, A0, WC, NFOCK):
    NPOL = NEL * NFOCK
    EMOL = np.append( [mf.e_tot], mf.e_tot + td.e)[:NEL]
    dipole_matrix_projected = dipole_matrix[:NEL,:NEL,:].dot(evec)
    a_op  = np.diag( np.sqrt(np.arange(1,NFOCK)), k=1 )
    # HPF   = np.zeros( (NPOL,NPOL) )
    # HPF  += np.kron( np.diag(EMOL), np.eye(NFOCK) )
    # HPF  += np.kron( np.eye(NEL), WC * np.diag(np.arange(NFOCK)) )
    # HPF  += WC * A0 * np.kron( dipole_matrix_projected, a_op.T + a_op )
    # HPF  += WC * A0**2 * np.kron(dipole_matrix_projected @ dipole_matrix_projected, np.eye(NFOCK) )
    # E, U  = np.linalg.eigh( HPF )
    

    # Do sparse version for memory efficiency
    from scipy.sparse import diags, kron, eye
    from scipy.sparse import csc_matrix as sparse_matrix
    from scipy.sparse import csr_matrix as sparse_csr_matrix
    HPF = sparse_matrix( (NPOL, NPOL) )
    HPF += kron( diags(EMOL), eye(NFOCK, format='csc') )
    HPF += kron( eye(NEL, format='csc'), diags(np.arange(NFOCK), format='csc') * WC )
    HPF += WC * A0 * kron( sparse_csr_matrix(dipole_matrix_projected), a_op.T + a_op )
    HPF += WC * A0**2 * kron(sparse_csr_matrix(dipole_matrix_projected @ dipole_matrix_projected), eye(NFOCK, format='csc'))
    HPF = HPF.tocsr()  # Convert to CSR format for efficient arithmetic operations

    return HPF

if ( __name__ == "__main__" ):
    import subprocess as sp
    DATA_DIR = "DATA_PLOTS_pQED"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)

    do_Coherent_State = False

    # Example usage
    mol = gto.Mole()
    mol.atom = 'Li 0 0 0; H 0 0 1.4'
    mol.basis = 'aug-ccpvdz'
    mol.verbose = 0
    mol.build()
    NAO = mol.nao_nr()
    nelec = mol.nelectron
    NOCC = nelec // 2
    NVIR = NAO - NOCC
    NCIS_MAX = NOCC * NVIR
    print("NCIS_MAX:", NCIS_MAX)

    T0 = time()
    mf = scf.RHF(mol)
    mf.kernel()
    td = tdscf.TDHF(mf)
    td.nstates = NOCC * NVIR
    td.kernel()
    print( "Electronic Structure Time: %1.4f" % (time() - T0) )

    T0 = time()
    electric_dipole_CIS = build_transition_electric_dipole_matrix(mol, mf, td)
    if ( do_Coherent_State == True ):
        print("Doing coherent state shift")
        electric_dipole_CIS[:,:,:] -= np.eye(1+td.nstates)[:,:,None] * electric_dipole_CIS[0,0,:]

    print( "Building Dipoles     Time: %1.4f" % (time() - T0) )

    # Build Pauli-Fierz Hamiltonian
    evec  = np.array([1,1,1])
    A0    = 0.01
    WC    = td.e[1]

    NEL_LIST   = np.arange(4, NCIS_MAX + 1)
    NFOCK_LIST = np.arange(2, 11)
    NPOL_KEEP  = 5
    E_LIST     = np.zeros( (len(NEL_LIST), len(NFOCK_LIST), NPOL_KEEP) )
    PHOT_LIST  = np.zeros( (len(NEL_LIST), len(NFOCK_LIST), NPOL_KEEP) )
    for NELi, NEL in enumerate(NEL_LIST):
        Iel = np.eye( NEL )
        for NFOCKi, NFOCK in enumerate(NFOCK_LIST):
            N_op = np.diag( np.arange(NFOCK) )
            print(f"Building Pauli-Fierz Hamiltonian for NEL={NEL}, NFOCK={NFOCK}, WC={WC:.3f}")
            N_op_full = np.kron( Iel, N_op )
            H_PF = build_Pauli_Fierz_Hamiltonian(mol, mf, td, evec, NEL, electric_dipole_CIS, A0, WC, NFOCK)
            E, U = sparse_eigsh(H_PF, k=np.min([NPOL_KEEP,NEL*NFOCK]), which='SA', return_eigenvectors=True, tol=1e-10)
            #E, U = np.linalg.eigh( H_PF )
            E_LIST[NELi, NFOCKi, :np.min([NPOL_KEEP,NEL*NFOCK])] = E[:np.min([NPOL_KEEP,NEL*NFOCK])]
            PHOT_LIST[NELi, NFOCKi, :np.min([NPOL_KEEP,NEL*NFOCK])] = np.einsum("aj,ab,bj->j", U[:,:np.min([NPOL_KEEP,NEL*NFOCK])], N_op_full[:,:], U[:,:np.min([NPOL_KEEP,NEL*NFOCK])], optimize=True)

            np.savetxt(f'{DATA_DIR}/pauli_fierz_energy_NEL_{NEL}_NFOCK_{NFOCK}_CD_{do_Coherent_State}.dat', E, fmt='%1.8f')

    E_LIST *= 1000 * 27.2114  # Convert to meV
    for pi in range( 5 ):
        if ( pi == 0 ):
            plt.imshow( E_LIST[:, :, pi] - E_LIST[-1, -1, pi], aspect='auto', origin='lower', cmap='bwr', extent=[NFOCK_LIST[0]-0.5, NFOCK_LIST[-1]+0.5, NEL_LIST[0]-0.5, NEL_LIST[-1]+0.5], norm=mpl.colors.CenteredNorm(vcenter=0.0))
        else:
            plt.imshow( E_LIST[:, :, pi] - E_LIST[:, :, 0] - (E_LIST[-1, -1, pi] - E_LIST[-1,-1, 0]), aspect='auto', origin='lower', cmap='bwr', extent=[NFOCK_LIST[0]-0.5, NFOCK_LIST[-1]+0.5, NEL_LIST[0]-0.5, NEL_LIST[-1]+0.5], norm=mpl.colors.CenteredNorm(vcenter=0.0))
        plt.colorbar(label='Energy (meV)', pad=0.01)
        plt.xlabel('Number of Fock States, $N_\\mathrm{F}$', fontsize=15)
        plt.ylabel('Number of Electronic States, $N_\\mathrm{el}$', fontsize=15)
        plt.tight_layout()
        plt.savefig('%s/pauli_fierz_energy_convergence_%d_%s_A0_%1.4f_WC_%1.4f_CS_%s.jpg' % (DATA_DIR, pi, mol.basis, A0, WC, do_Coherent_State), dpi=300)
        plt.clf()
        plt.close()

        plt.imshow( PHOT_LIST[:,:,pi], aspect='auto', origin='lower', cmap='winter', extent=[NFOCK_LIST[0]-0.5, NFOCK_LIST[-1]+0.5, NEL_LIST[0]-0.5, NEL_LIST[-1]+0.5], vmin=np.min(PHOT_LIST[:,:,pi]))
        plt.colorbar(label='Average Photon Number', pad=0.01)
        plt.xlabel('Number of Fock States, $N_\\mathrm{F}$', fontsize=15)
        plt.ylabel('Number of Electronic States, $N_\\mathrm{el}$', fontsize=15)
        plt.tight_layout()
        plt.savefig('%s/pauli_fierz_photon_convergence_%d_%s_A0_%1.4f_WC_%1.4f_CS_%s.jpg' % (DATA_DIR, pi, mol.basis, A0, WC, do_Coherent_State), dpi=300)
        plt.clf()
        plt.close()

        for NFi, NF in enumerate( NFOCK_LIST[:-1] ):
            # plt.plot( NEL_LIST, E_LIST[:, NFi, pi], "-o", label='$N_\\mathrm{F}$ = %d'% (NF) )
            plt.semilogy( NEL_LIST, E_LIST[:, NFi, pi] - E_LIST[:, -1, pi], "-o", label='$N_\\mathrm{F}$ = %d'% (NF) )
        plt.xlabel('Number of Electronic States, $N_\\mathrm{el}$', fontsize=15)
        plt.ylabel('Error in Energy, $E(N_\\mathrm{F}, N_\\mathrm{el}) - E(%d, N_\\mathrm{el})$ (eV)' % (NFOCK_LIST[-1]), fontsize=15)
        # plt.title('Converged Value: %1.4f' % (E_LIST[-1, -1, pi]), fontsize=15)
        plt.ylim(1e-10)
        plt.legend()
        plt.savefig('%s/pauli_fierz_energy_convergence_%d_%s_A0_%1.4f_WC_%1.4f_NFOCK_lines_CS_%s.jpg' % (DATA_DIR, pi, mol.basis, A0, WC, do_Coherent_State), dpi=300)
        plt.clf()
        plt.close()

        for NFi, NF in enumerate( NFOCK_LIST[:-1] ):
            plt.semilogy( NEL_LIST, np.abs( PHOT_LIST[:, NFi, pi] - PHOT_LIST[:, -1, pi] ), "-o", label='$N_\\mathrm{F}$ = %d'% (NF) )
            # plt.semilogy( NEL_LIST, np.abs( PHOT_LIST[:, NFi, pi] - PHOT_LIST[-1, -1, pi] ), "-o", label='$N_\\mathrm{F}$ = %d'% (NF) )
        plt.xlabel('Number of Electronic States, $N_\\mathrm{el}$', fontsize=15)
        plt.ylabel('Error in Photon Number, $\\langle \\hat{N} \\rangle (N_\\mathrm{F}, N_\\mathrm{el}) - \\langle \\hat{N} \\rangle (%d, N_\\mathrm{el})$' % (NFOCK_LIST[-1]), fontsize=12)
        plt.title('Converged Value: %1.4f' % (PHOT_LIST[-1, -1, pi]), fontsize=15)
        plt.ylim(1e-10)
        plt.legend()
        plt.tight_layout()
        plt.savefig('%s/pauli_fierz_photon_convergence_%d_%s_A0_%1.4f_WC_%1.4f_NFOCK_lines_CS_%s.jpg' % (DATA_DIR, pi, mol.basis, A0, WC, do_Coherent_State), dpi=300)
        plt.clf()
        plt.close()





    plt.imshow( E_LIST[:, :, 3] - E_LIST[:, :, 2] - (E_LIST[-1, -1, 3] - E_LIST[-1,-1, 2]), aspect='auto', origin='lower', cmap='bwr', extent=[NFOCK_LIST[0]-0.5, NFOCK_LIST[-1]+0.5, NEL_LIST[0]-0.5, NEL_LIST[-1]+0.5], norm=mpl.colors.CenteredNorm(vcenter=0.0))
    plt.colorbar(label='Energy (meV)', pad=0.01)
    plt.xlabel('Number of Fock States, $N_\\mathrm{F}$', fontsize=15)
    plt.ylabel('Number of Electronic States, $N_\\mathrm{el}$', fontsize=15)
    plt.tight_layout()
    plt.savefig('%s/pauli_fierz_Rabi_convergence_%s_A0_%1.4f_WC_%1.4f_CS_%s.jpg' % (DATA_DIR, mol.basis, A0, WC, do_Coherent_State), dpi=300)
    plt.clf()
    plt.close()

    output = [NEL_LIST]
    for NFi, NF in enumerate( NFOCK_LIST[:-1] ):
        output.append( E_LIST[:, NFi, 3] - E_LIST[:, NFi, 2] )
    np.savetxt('%s/pauli_fierz_Rabi_convergence_%s_A0_%1.4f_WC_%1.4f_NFOCK_lines.dat' % (DATA_DIR, mol.basis, A0, WC), np.array(np.c_[NEL_LIST, E_LIST[:, :, 3] - E_LIST[:, :, 2] - (E_LIST[-1, -1, 3] - E_LIST[-1,-1, 2])]).T)


    for NFi, NF in enumerate( NFOCK_LIST[:-1] ):
        plt.plot( NEL_LIST, E_LIST[:, NFi, 3] - E_LIST[:, NFi, 2] , "-o", label='$N_\\mathrm{F}$ = %d'% (NF) )
    plt.xlabel('Number of Electronic States, $N_\\mathrm{el}$', fontsize=15)
    plt.ylabel('Rabi Splitting, $\\omega_\\mathrm{R}$ (eV)' % (NFOCK_LIST[-1]), fontsize=15)
    #plt.ylim(1e-10)
    plt.legend()
    plt.savefig('%s/pauli_fierz_Rabi_convergence_%s_A0_%1.4f_WC_%1.4f_NFOCK_lines_CS_%s.jpg' % (DATA_DIR, mol.basis, A0, WC, do_Coherent_State), dpi=300)
    plt.clf()
    plt.close()

    output = [NEL_LIST]
    for NFi, NF in enumerate( NFOCK_LIST[:-1] ):
        output.append( E_LIST[:, NFi, 3] - E_LIST[:, NFi, 2] )
    np.savetxt('%s/pauli_fierz_Rabi_convergence_%s_A0_%1.4f_WC_%1.4f_NFOCK_lines_CS_%s.dat' % (DATA_DIR, mol.basis, A0, WC, do_Coherent_State), np.array(output).T, header="NEL, NF=2, NF=3, ...")


    for NFi, NF in enumerate( NFOCK_LIST[:-1] ):
        plt.semilogy( NEL_LIST, np.abs(E_LIST[:, NFi, 3] - E_LIST[:, NFi, 2] - (E_LIST[-1, -1, 3] - E_LIST[-1, -1, 2])), "-o", label='$N_\\mathrm{F}$ = %d'% (NF) )
    plt.xlabel('Number of Electronic States, $N_\\mathrm{el}$', fontsize=15)
    plt.ylabel('Rabi Splitting, $\\omega_\\mathrm{R}$ (eV)' % (NFOCK_LIST[-1]), fontsize=15)
    plt.ylim(1e-10)
    plt.legend()
    plt.savefig('%s/pauli_fierz_Rabi_convergence_%s_A0_%1.4f_WC_%1.4f_NFOCK_lines_ERROR_CS_%s.jpg' % (DATA_DIR, mol.basis, A0, WC, do_Coherent_State), dpi=300)
    plt.clf()
    plt.close()

    output = [NEL_LIST]
    for NFi, NF in enumerate( NFOCK_LIST[:-1] ):
        output.append( np.abs(E_LIST[:, NFi, 3] - E_LIST[:, NFi, 2] - (E_LIST[-1, -1, 3] - E_LIST[-1, -1, 2])) )
    np.savetxt('%s/pauli_fierz_Rabi_convergence_%s_A0_%1.4f_WC_%1.4f_NFOCK_lines_ERROR_CS_%s.dat' % (DATA_DIR, mol.basis, A0, WC, do_Coherent_State), np.array(output).T, header="NEL, NF=2, NF=3, ...")

