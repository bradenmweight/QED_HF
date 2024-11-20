import numpy as np
from opt_einsum import contract as opt_einsum # Very fast library for tensor contractions

from pyscf import gto, scf, tdscf

from QED_UHF import do_QED_UHF
from QED_RHF import do_QED_RHF
from QED_RCIS import do_QED_RCIS

from tools import eigh, make_RDM1_ao
from ao_ints import get_ao_integrals, get_electric_dipole_ao, get_magnetic_dipole_ao, get_electric_quadrupole_ao


def do_QED_UCIS( mol, LAM, WC, do_CS=True, C_HF=None, eps_HF=None, return_wfn=False, nstates=1, calc_moments=False, spin_flipping=False ):
    DSE_FACTOR = 0.5 * LAM**2

    # Get RHF wavefunction
    if ( C_HF is None or len(C_HF.shape) != 3 ):
        E_HF, S2_HF, ss1_HF, C_HF, eps_HF = do_QED_UHF( mol, LAM=LAM, WC=WC, return_wfn=True, return_MO_energies=True )

    # Get the h1e and ERIs in orthogonal AO basis
    h1e, eri, n_elec_alpha, n_elec_beta, nuclear_repulsion_energy = get_ao_integrals( mol )

    # Get the electric dipole and quadrupole in the orthogonal AO basis
    dip_ao, quad_ao = get_electric_dipole_ao( mol ), get_electric_quadrupole_ao( mol )
    EPOL    = np.array([0,0,1])
    dip_ao  = np.einsum("x,xpq->pq", EPOL, dip_ao)
    quad_ao = np.einsum("x,xypq,y->pq", EPOL, quad_ao, EPOL)
    #eri_DSE = 2 * DSE_FACTOR  * np.einsum( 'pq,rs->pqrs->ijkl', dip_ao, dip_ao )

    # Get average dipole moment for CS shift, if used
    D_a        = make_RDM1_ao( C_HF[0], (np.arange(n_elec_alpha)) )
    D_b        = make_RDM1_ao( C_HF[0], (np.arange(n_elec_beta)) )
    AVEdipole  = (do_CS) * np.einsum( 'pq,pq->', D_a + D_b, dip_ao )

    # Get number of occupied and virtual orbitals
    n_occ_a = n_elec_alpha
    n_vir_a = len(h1e) - n_occ_a
    n_occ_b = n_elec_beta
    n_vir_b = len(h1e) - n_occ_b

    # Get the indices of the occupied and virtual orbitals
    o_a = slice( 0, n_occ_a )
    v_a = slice( n_occ_a, n_occ_a + n_vir_a )
    o_b = slice( 0, n_occ_b )
    v_b = slice( n_occ_b, n_occ_b + n_vir_b )

    # Get the slices of the occupied and virtual orbitals energies
    eps_occ_a = eps_HF[0,o_a]
    eps_vir_a = eps_HF[0,v_a]
    eps_occ_b = eps_HF[1,o_b]
    eps_vir_b = eps_HF[1,v_b]

    # Obtain all needed eris
    tmp       = opt_einsum( "pi,qj,pqrs,rk,sl->ijkl", C_HF[0], C_HF[0], eri, C_HF[0], C_HF[0] )
    ovov_aaaa = tmp[o_a,v_a,o_a,v_a]
    oovv_aaaa = tmp[o_a,o_a,v_a,v_a]
    tmp       = opt_einsum( "pi,qj,pqrs,rk,sl->ijkl", C_HF[1], C_HF[1], eri, C_HF[1], C_HF[1] )
    ovov_bbbb = tmp[o_b,v_b,o_b,v_b]
    oovv_bbbb = tmp[o_b,o_b,v_b,v_b]
    tmp       = opt_einsum( "pi,qj,pqrs,rk,sl->ijkl", C_HF[0], C_HF[0], eri, C_HF[1], C_HF[1] )
    ovov_aabb = tmp[o_a,v_a,o_b,v_b]
    oovv_aabb = tmp[o_a,o_a,v_b,v_b]
    tmp       = opt_einsum( "pi,qj,pqrs,rk,sl->ijkl", C_HF[1], C_HF[1], eri, C_HF[0], C_HF[0] )
    ovov_bbaa = tmp[o_b,v_b,o_a,v_a]
    oovv_bbaa = tmp[o_b,o_b,v_a,v_a]

    # Obtain all needed eri_DSEs
    tmp           = 2 * DSE_FACTOR * opt_einsum( "pi,qj,pq,rs,rk,sl->ijkl", C_HF[0], C_HF[0], dip_ao, dip_ao, C_HF[0], C_HF[0] )
    ovov_aaaa_DSE = tmp[o_a,v_a,o_a,v_a]
    oovv_aaaa_DSE = tmp[o_a,o_a,v_a,v_a]
    tmp           = 2 * DSE_FACTOR * opt_einsum( "pi,qj,pq,rs,rk,sl->ijkl", C_HF[1], C_HF[1], dip_ao, dip_ao, C_HF[1], C_HF[1] )
    ovov_bbbb_DSE = tmp[o_b,v_b,o_b,v_b]
    oovv_bbbb_DSE = tmp[o_b,o_b,v_b,v_b]
    tmp           = 2 * DSE_FACTOR * opt_einsum( "pi,qj,pq,rs,rk,sl->ijkl", C_HF[0], C_HF[0], dip_ao, dip_ao, C_HF[1], C_HF[1] )
    ovov_aabb_DSE = tmp[o_a,v_a,o_b,v_b]
    oovv_aabb_DSE = tmp[o_a,o_a,v_b,v_b]
    tmp           = 2 * DSE_FACTOR * opt_einsum( "pi,qj,pq,rs,rk,sl->ijkl", C_HF[1], C_HF[1], dip_ao, dip_ao, C_HF[0], C_HF[0] )
    ovov_bbaa_DSE = tmp[o_b,v_b,o_a,v_a]
    oovv_bbaa_DSE = tmp[o_b,o_b,v_a,v_a]

    # Initialize the CIS Hamiltonian parts
    H_aaaa = np.zeros( (n_occ_a,n_vir_a,n_occ_a,n_vir_a) )
    H_aabb = np.zeros( (n_occ_a,n_vir_a,n_occ_b,n_vir_b) )
    H_bbaa = np.zeros( (n_occ_b,n_vir_b,n_occ_a,n_vir_a) )
    H_bbbb = np.zeros( (n_occ_b,n_vir_b,n_occ_b,n_vir_b) )
    H_aaP  = np.zeros( (n_occ_a,n_vir_a,1,1) ) # Only single mode for now
    H_bbP  = np.zeros( (n_occ_b,n_vir_b,1,1) ) # Only single mode for now
    H_PP   = np.zeros( (1,1,1,1) )             # Only single mode for now

    # Build CIS Hamiltonian parts
    E_aaaa     = np.diag((eps_vir_a[:,None] - eps_occ_a).flatten() ).reshape((n_occ_a,n_vir_a,n_occ_a,n_vir_a))
    E_bbbb     = np.diag((eps_vir_b[:,None] - eps_occ_b).flatten() ).reshape((n_occ_b,n_vir_b,n_occ_b,n_vir_b))
    H_aaaa[:]  = E_aaaa + opt_einsum('iajb->iajb', ovov_aaaa ) - opt_einsum('ijba->iajb', oovv_aaaa )
    H_aabb[:]  = opt_einsum('iajb->iajb', ovov_aabb ) # No exchange since i != j spin and a != b spin
    H_bbaa[:]  = opt_einsum('iajb->iajb', ovov_bbaa ) # No exchange since i != j spin and a != b spin
    H_bbbb[:]  = E_bbbb + opt_einsum('iajb->iajb', ovov_bbbb ) - opt_einsum('ijba->iajb', oovv_bbbb )

    H_aaaa[:] += 0.5 * (opt_einsum('iajb->iajb', ovov_aaaa_DSE ) - opt_einsum('ijba->iajb', oovv_aaaa_DSE ) )
    H_aabb[:] += 0.5 * (opt_einsum('iajb->iajb', ovov_aabb_DSE ) ) # No exchange since i != j spin and a != b spin
    H_bbaa[:] += 0.5 * (opt_einsum('iajb->iajb', ovov_bbaa_DSE ) ) # No exchange since i != j spin and a != b spin
    H_bbbb[:] += 0.5 * (opt_einsum('iajb->iajb', ovov_bbbb_DSE ) - opt_einsum('ijba->iajb', oovv_bbbb_DSE ) )

    dip_mo_a = np.einsum( "pi,pq,qj->ij", C_HF[0], dip_ao, C_HF[0] )
    dip_mo_b = np.einsum( "pi,pq,qj->ij", C_HF[1], dip_ao, C_HF[1] )
    H_aaP[:,:,0,0] = np.sqrt(WC / 2) * LAM * dip_mo_a[o_a,v_a] # Only single mode for now
    H_bbP[:,:,0,0] = np.sqrt(WC / 2) * LAM * dip_mo_b[o_b,v_b] # Only single mode for now
    H_PP[0,0,0,0]  = WC # Only single mode for now

    # Reshape all blocks
    H_aaaa = H_aaaa.reshape((n_occ_a*n_vir_a,n_occ_a*n_vir_a))
    H_aabb = H_aabb.reshape((n_occ_a*n_vir_a,n_occ_b*n_vir_b))
    H_bbaa = H_bbaa.reshape((n_occ_b*n_vir_b,n_occ_a*n_vir_a))
    H_bbbb = H_bbbb.reshape((n_occ_b*n_vir_b,n_occ_b*n_vir_b))
    H_aaP  = H_aaP.reshape((n_occ_a*n_vir_a,1))
    H_bbP  = H_bbP.reshape((n_occ_b*n_vir_b,1))
    H_PP   = H_PP.reshape((1,1))

    # Build CIS Hamiltonian
    H = np.block( [[H_aaaa,H_aabb,H_aaP],[H_bbaa,H_bbbb,H_bbP],[H_aaP,H_bbP,H_PP]] )

    #print( "H_UCIS (WC = %1.3f)\n" % WC, np.round(H, 3) )

    E_CIS, C_CIS = eigh( H )
    #print( E_CIS )

    #E_CIS = E_CIS[:nstates]
    #C_CIS = C_CIS[:,:nstates]
    #print("IO Excitation:  %1.5f " % (eps_HF[1] - eps_HF[0]) )
    #print("RCIS " + "Triplets" * (not do_Singlets) + "Singlets" * (do_Singlets) + "          ", np.round(E_CIS,5) )

    #if ( nstates > len(E_CIS) ):
    #    nstates = len(E_CIS)

    ##### Normalize the RCIS wavefunction #####
    NORM   = np.einsum("xS,xS->S", C_CIS, C_CIS)
    C_CIS *= np.sqrt( 2 / NORM ) # sqrt(2) * psi

    out_list = [E_CIS]
    if ( return_wfn == True ):
        out_list.append([C_CIS,C_HF])

    #if ( calc_moments == True ):
    #    EL_DIP  = get_electric_dipole_moments( mol, C_HF, C_CIS, o, v, do_Singlets  )
    #    MAG_DIP = get_magnetic_dipole_moments( mol, C_HF, C_CIS, o, v, do_Singlets  )
    #    EL_QUAD = get_electric_quadrupole_moments( mol, C_HF, C_CIS, o, v, do_Singlets  )
    #    out_list.append( [EL_DIP, MAG_DIP, EL_QUAD] )
    
    if ( len(out_list) == 1 ):
        return E_CIS
    return out_list

if ( __name__ == "__main__"):
    import matplotlib.pyplot as plt
    
    from RHF import do_RHF
    from RCIS import do_RCIS

    LAM = 0.05
    WC  = 0.6

    mol = gto.Mole()
    mol.basis = "sto3g"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'H 0 0 0; H 0 0 2.0'
    mol.verbose = 0
    mol.build()

    E_UHF, S2, ss1, C_UHF = do_QED_UHF( mol, LAM, WC, return_wfn=True )

    nstates = 2
    E_UCIS_S = do_QED_UCIS(mol, LAM, WC, nstates=nstates )

    print( "  QED-UHF      : %1.5f" % (E_UHF) )
    print( "  QED-UCIS (eV):", ((E_UCIS_S[:nstates])*27.2114))






    R_LIST  = np.linspace(0.5,3.5,400)/0.529 # Angstrom to Bohr
    nstates = 3
    E_LIST_R_S    = np.zeros( (len(R_LIST),   nstates+1-1) )
    E_LIST_R_T    = np.zeros( (len(R_LIST),   nstates+1-1) )
    E_LIST_U      = np.zeros( (len(R_LIST),   nstates+1) )
    for Ri,R in enumerate(R_LIST):
        print("R = %1.2f A" % (R*0.528))
        mol.atom = 'H 0 0 0; H 0 0 %1.2f' % (R)
        mol.build()
        E_RHF, C_UHF, eps_HF          = do_QED_RHF ( mol, LAM, WC, return_wfn=True, return_MO_energies=True )
        E_RHF, C_RHF, eps_HF          = do_QED_RHF ( mol, LAM, WC, return_wfn=True, return_MO_energies=True )
        E_RCIS                        = do_QED_RCIS( mol, LAM, WC, C_HF=C_RHF, eps_HF=eps_HF, nstates=nstates, symmetry="Singlet" )
        E_LIST_R_S[Ri,0]              = E_RHF
        E_LIST_R_S[Ri,1:]             = E_RCIS + E_RHF
        E_RCIS                        = do_QED_RCIS( mol, LAM, WC, C_HF=C_RHF, eps_HF=eps_HF, nstates=nstates, symmetry="Triplet" )
        E_LIST_R_T[Ri,0]              = E_RHF
        E_LIST_R_T[Ri,1:]             = E_RCIS + E_RHF
        E_UHF, S2, ss1, C_UHF, eps_HF = do_QED_UHF( mol, LAM, WC, return_wfn=True, return_MO_energies=True )
        E_UCIS                        = do_QED_UCIS( mol, LAM, WC, C_HF=C_UHF, eps_HF=eps_HF, nstates=nstates, spin_flipping=False )
        E_LIST_U[Ri,0]                = E_UHF
        E_LIST_U[Ri,1:]               = E_UCIS + E_UHF



    for i in range(E_LIST_U.shape[1]):
        plt.plot( R_LIST*0.529, E_LIST_U[:,i] - WC/2, c="black", label="QED-UHF/UCIS" * (i==0) )

    #plt.plot( R_LIST*0.529, E_LIST_R_S[:,0], "-", c="red", label="QED-RHF/RCIS (S)" )
    #for i in range(1,E_LIST_R_S.shape[1]):
    #    plt.plot( R_LIST*0.529, E_LIST_R_S[:,i], "-", c="red" )
    #    plt.plot( R_LIST*0.529, E_LIST_R_T[:,i], "-", c="cyan", label="QED-RHF/RCIS (T)" * (i==1) )

    plt.legend()
    plt.xlim(R_LIST[0]*0.529,R_LIST[-1]*0.529)
    #plt.ylim(-1.2,-0.2)
    #plt.ylim(-0.9,0.4)
    plt.ylim(-1.2,-0.2)
    plt.xlabel("$R_\\mathrm{HH}$ ($\\AA$)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$\\lambda$ = %1.3f a.u.  $\\omega_\\mathrm{c}$ = %1.3f a.u." % (LAM, WC), fontsize=15)
    plt.tight_layout()
    plt.savefig("QED_UCIS_H2.jpg", dpi=300)

