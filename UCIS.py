import numpy as np
from opt_einsum import contract as opt_einsum # Very fast library for tensor contractions

from pyscf import gto, scf, tdscf

from pyscf_results import do_UHF_pyscf, get_UCIS_pyscf

from UHF import do_UHF
from ao_ints import get_ao_integrals, get_electric_dipole_ao, get_magnetic_dipole_ao, get_electric_quadrupole_ao

def get_off_diagonal_overlap( mol_1, mol_2 ):
    # Combine coordinates of two molecules
    mol = gto.Mole()
    mol.atom = mol_1.atom + "; " + mol_2.atom
    mol.basis = mol_1.basis
    mol.unit = mol_1.unit
    mol.symmetry = mol_1.symmetry
    mol.verbose = 0
    mol.build()
    S      = mol.intor('int1e_ovlp')
    n_ao = S.shape[0]//2
    return S[:n_ao,n_ao:] # Only consider the overlap between the two molecules

def compute_spatial_overlap( CIS_1, CIS_2, mo_1, mo_2, mol_1, mol_2 ):
    """
    Assume that mo_i and CIS_i are in the orthogonal ao basis.
    Therefore, we need to know the non-orthogonal overlap matrix.
    """
    n_occ,n_vir,nstates = CIS_1.shape
    o,v                 = slice(0,n_occ), slice(n_occ,n_occ+n_vir)
    S_1                 = mol_1.intor('int1e_ovlp')
    S_2                 = mol_2.intor('int1e_ovlp')
    s_1, u_1            = np.linalg.eigh(S_1)
    s_2, u_2            = np.linalg.eigh(S_2)
    Shalf_1             = u_1 @ np.diag(1/np.sqrt(s_1)) @ u_1.T
    Shalf_2             = u_2 @ np.diag(1/np.sqrt(s_2)) @ u_2.T

    mo_1_o = mo_1[:,o]
    mo_1_v = mo_1[:,v]
    mo_2_o = mo_2[:,o]
    mo_2_v = mo_2[:,v]

    S_AB       = get_off_diagonal_overlap( mol_1, mol_2 )
    S_AB_ORTHO = opt_einsum( "pi,pq,qj->ij", Shalf_1, S_AB, Shalf_2 ) # Overlap between the two molecules in their orthogonal bases
    OVLP       = 0.500 * opt_einsum( "ovJ,po,pv,pq,qo,qv,ovK->JK", CIS_1, mo_1_o, mo_1_v, S_AB_ORTHO, mo_2_o, mo_2_v, CIS_2 )
    #OVLP       = 0.500 * opt_einsum( "ovJ,ovK->JK", CIS_1, CIS_2 ) # Does not acount for changes in AO or MO basis
    print( np.round( OVLP, 3 ) )
    print("Overlap code is not working right...")
    



def get_electric_dipole_moments( mol, C_HF, C_CIS, o, v, do_Singlets, doPrint=True ):
    DIP           = np.zeros( (nstates+1,nstates+1,3) )
    S             = mol.intor('int1e_ovlp')
    s,u           = np.linalg.eigh(S)
    Shalf         = u @ np.diag(1/np.sqrt(s)) @ u.T
    dip_ao        = get_electric_dipole_ao( mol )
    dip_ao        = opt_einsum( "pi,dij,qj->dpq", Shalf, dip_ao, Shalf )
    dip_ao       *= -2 # Take electron to be negative and count 2 electrons for RCIS
    dip_mo        = opt_einsum( "pi,dpq,qj->dij", C_HF, dip_ao, C_HF )
    DIP[0,0,:]    = opt_einsum( "pi,dij,pj->d", C_HF[:,o], dip_mo[:,o,o], C_HF[:,o] )
    DIP[0,1:,:]   = (do_Singlets) * opt_einsum( "dov,ovS->Sd", dip_mo[:,o,v], C_CIS )
    DIP[1:,0,:]   = DIP[0,1:]
    DIP[1:,1:,:]  = opt_einsum( "dvk,ovA,okB->ABd", dip_mo[:,v,v], C_CIS[:,:,:], C_CIS[:,:,:] )
    DIP[1:,1:,:] -= opt_einsum( "dou,ovA,uvB->ABd", dip_mo[:,o,o], C_CIS[:,:,:], C_CIS[:,:,:] )

    if ( doPrint == True ):
        print("Transition Electric Dipole Elements (a.u.):")
        for state in range(nstates+1):
            print("0 --> %d: %2.5f %2.5f %2.5f" % (state, DIP[0,state,0], DIP[0,state,1], DIP[0,state,2]) )
    return DIP

def get_magnetic_dipole_moments( mol, C_HF, C_CIS, o, v, do_Singlets, doPrint=True  ):
    DIP           = np.zeros( (nstates+1,nstates+1,3) )
    S             = mol.intor('int1e_ovlp')
    s,u           = np.linalg.eigh(S)
    Shalf         = u @ np.diag(1/np.sqrt(s)) @ u.T
    dip_ao        = get_magnetic_dipole_ao( mol )
    dip_ao        = opt_einsum( "pi,dij,qj->dpq", Shalf, dip_ao, Shalf )
    dip_ao       *= -2 # Take electron to be negative and count 2 electrons for RCIS
    dip_mo        = opt_einsum( "pi,dpq,qj->dij", C_HF, dip_ao, C_HF )
    DIP[0,0,:]    = opt_einsum( "pi,dij,pj->d", C_HF[:,o], dip_mo[:,o,o], C_HF[:,o] )
    DIP[0,1:,:]   = (do_Singlets) * opt_einsum( "dov,ovS->Sd", dip_mo[:,o,v], C_CIS )
    DIP[1:,0,:]   = DIP[0,1:]
    DIP[1:,1:,:]  = opt_einsum( "dvk,ovA,okB->ABd", dip_mo[:,v,v], C_CIS[:,:,:], C_CIS[:,:,:] )
    DIP[1:,1:,:] -= opt_einsum( "dou,ovA,uvB->ABd", dip_mo[:,o,o], C_CIS[:,:,:], C_CIS[:,:,:] )

    if ( doPrint == True ):
        print("Transition Magnetic Dipole Elements (a.u.):")
        for state in range(nstates+1):
            print("0 --> %d: %1.5f %1.5f %1.5f" % (state, DIP[0,state,0], DIP[0,state,1], DIP[0,state,2]) )
    return DIP

def get_electric_quadrupole_moments( mol, C_HF, C_CIS, o, v, do_Singlets, doPrint=False  ):
    QUAD             = np.zeros( (nstates+1,nstates+1,3,3) )
    S                = mol.intor('int1e_ovlp')
    s,u              = np.linalg.eigh(S)
    Shalf            = u @ np.diag(1/np.sqrt(s)) @ u.T
    quad_ao          = get_electric_quadrupole_ao( mol )
    quad_ao          = opt_einsum( "pi,xyij,qj->xypq", Shalf, quad_ao, Shalf )
    quad_ao         *= -2 # Take electron to be negative and count 2 electrons for RCIS
    quad_mo          = opt_einsum( "pi,xypq,qj->xyij", C_HF, quad_ao, C_HF )
    QUAD[0,0,  :,:]  = opt_einsum( "pi,xyij,pj->xy", C_HF[:,o], quad_mo[:,:,o,o], C_HF[:,o] )
    QUAD[0,1:, :,:]  = QUAD[0,1:,:] = (do_Singlets) * opt_einsum( "xyov,ovS->Sxy", quad_mo[:,:,o,v], C_CIS )
    QUAD[1:,1:,:,:]  = opt_einsum( "xyvk,ovA,okB->ABxy", quad_mo[:,:,v,v], C_CIS[:,:,:], C_CIS[:,:,:] )
    QUAD[1:,1:,:,:] -= opt_einsum( "xyou,ovA,uvB->ABxy", quad_mo[:,:,o,o], C_CIS[:,:,:], C_CIS[:,:,:] )

    if ( doPrint == True ):
        print("Transition Electric Quadrupole Elements (a.u.):")
        for state in range(nstates+1):
            print(" 0 --> %d: " % (state) )
            print("          X          Y          Z" )
            XYZ     = ["X", "Y", "Z"]
            for x in range(3):
                string = "%s          " % (XYZ[x])
                for y in range(3):
                    if ( x < y ): continue
                    string += "%2.4f          " % (QUAD[0,state,x,y])
                print(string)
        
    return QUAD


def do_UCIS( mol, C_HF=None, eps_HF=None, return_wfn=False, nstates=1, calc_moments=False, spin_flipping=False ):

    # Get RHF wavefunction
    if ( C_HF is None or len(C_HF.shape) != 3 ):
        E_HF, S2_HF, ss1_HF, C_HF, eps_HF = do_UHF( mol, return_wfn=True, return_MO_energies=True )

    # Get the h1e and ERIs in orthogonal AO basis
    h1e, eri, n_elec_alpha, n_elec_beta, nuclear_repulsion_energy = get_ao_integrals( mol )

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
    if ( spin_flipping == True ):
        tmp       = opt_einsum( "pi,qj,pqrs,rk,sl->ijkl", C_HF[0], C_HF[1], eri, C_HF[0], C_HF[1] )
        ovov_abab = tmp[o_a,v_b,o_a,v_b]
        oovv_abab = tmp[o_a,o_b,v_b,v_a]
        tmp       = opt_einsum( "pi,qj,pqrs,rk,sl->ijkl", C_HF[1], C_HF[0], eri, C_HF[1], C_HF[0] )
        ovov_baba = tmp[o_b,v_a,o_b,v_a]
        oovv_baba = tmp[o_b,o_a,v_a,v_b]

    # Initialize the CIS Hamiltonian parts
    H_aaaa = np.zeros( (n_occ_a,n_vir_a,n_occ_a,n_vir_a) )
    H_aabb = np.zeros( (n_occ_a,n_vir_a,n_occ_b,n_vir_b) )
    H_bbaa = np.zeros( (n_occ_b,n_vir_b,n_occ_a,n_vir_a) )
    H_bbbb = np.zeros( (n_occ_b,n_vir_b,n_occ_b,n_vir_b) )
    if ( spin_flipping == True ):
        H_abab = np.zeros( (n_occ_a,n_vir_b,n_occ_a,n_vir_b) )
        H_baba = np.zeros( (n_occ_b,n_vir_a,n_occ_b,n_vir_a) )

    # Build CIS Hamiltonian parts
    E_aaaa     = np.diag((eps_vir_a[:,None] - eps_occ_a).flatten() ).reshape((n_occ_a,n_vir_a,n_occ_a,n_vir_a))
    E_bbbb     = np.diag((eps_vir_b[:,None] - eps_occ_b).flatten() ).reshape((n_occ_b,n_vir_b,n_occ_b,n_vir_b))
    H_aaaa[:]  = opt_einsum('iajb->iajb', ovov_aaaa ) - opt_einsum('ijba->iajb', oovv_aaaa )
    H_aabb[:]  = opt_einsum('iajb->iajb', ovov_aabb ) # No exchange since i != j spin and a != b spin
    H_bbaa[:]  = opt_einsum('iajb->iajb', ovov_bbaa ) # No exchange since i != j spin and a != b spin
    H_bbbb[:]  = opt_einsum('iajb->iajb', ovov_bbbb ) - opt_einsum('ijba->iajb', oovv_bbbb )
    if ( spin_flipping == True ):
        E_abab     = np.diag((eps_vir_b[:,None] - eps_occ_a).flatten() ).reshape((n_occ_a,n_vir_b,n_occ_a,n_vir_b))
        E_baba     = np.diag((eps_vir_a[:,None] - eps_occ_b).flatten() ).reshape((n_occ_b,n_vir_a,n_occ_b,n_vir_a))
        H_abab[:]  = opt_einsum('iajb->iajb', ovov_abab ) #- opt_einsum('ijba->iajb', oovv_abab ) # Do we need exchange terms?
        H_baba[:]  = opt_einsum('iajb->iajb', ovov_baba ) #- opt_einsum('ijba->iajb', oovv_baba ) # Do we need exchange terms?

    # Define Hamiltonian index slices for alpha and beta blocks
    Naa  = n_occ_a * n_vir_a
    Nbb  = n_occ_b * n_vir_b
    Nab  = n_occ_a * n_vir_b
    Nba  = n_occ_b * n_vir_a
    aaaa = (slice(0,Naa), slice(0,Naa))
    aabb = (slice(0,Naa), slice(Naa,Naa+Nbb))
    bbaa = (slice(Naa,Naa+Nbb), slice(0,Naa))
    bbbb = (slice(Naa,Naa+Nbb), slice(Naa,Naa+Nbb))
    if ( spin_flipping == True ):
        abab = (slice(Naa+Nbb,Naa+Nbb+Nab), slice(Naa+Nbb,Naa+Nbb+Nab))
        baba = (slice(Naa+Nbb+Nab,Naa+Nbb+Nab+Nba), slice(Naa+Nbb+Nab,Naa+Nbb+Nab+Nba))

    # Build CIS Hamiltonian
    if ( spin_flipping == True ):
        H       = np.zeros( (Naa+Nbb+Nab+Nba, Naa+Nbb+Nab+Nba) )
    else:
        H       = np.zeros( (Naa+Nbb, Naa+Nbb) )
    H[aaaa] = (H_aaaa + E_aaaa).reshape( (Naa, Naa) )
    H[aabb] = H_aabb.reshape( (Naa, Nbb) )
    H[bbaa] = H_bbaa.reshape( (Nbb, Naa) )
    H[bbbb] = (H_bbbb + E_bbbb).reshape( (Nbb, Nbb) )
    if ( spin_flipping == True ):
        H[abab] = (H_abab + E_abab).reshape( (Nab, Nab) )
        H[baba] = (H_baba + E_baba).reshape( (Nba, Nba) )
        Hoff = np.block( np.array([H[abab],H[baba]]) )

    E_CIS, C_CIS = np.linalg.eigh( H )
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

    mol = gto.Mole()
    mol.basis = "sto3g"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'H 0 0 0; H 0 0 10.0'
    mol.verbose = 0
    mol.build()

    E_UHF, S2, ss1, C_UHF = do_UHF( mol, return_wfn=True )

    nstates = 2
    E_UHF_pyscf, E_UCIS_S_pyscf = get_UCIS_pyscf( mol, nstates=nstates )
    E_UCIS_S = do_UCIS(mol, nstates=nstates )

    print( "  BMW UHF                       : %1.5f" % (E_UHF) )
    print( "PySCF UHF                       : %1.5f" % (E_UHF_pyscf) )
    print( "PySCF Transition Energies (eV)  :", ((E_UCIS_S_pyscf[:nstates]-E_UHF_pyscf)*27.2114))
    print( "  BMW Transition Energies (eV)  :", ((E_UCIS_S[:nstates])*27.2114))





    from RHF import do_RHF
    from RCIS import do_RCIS
    R_LIST  = np.linspace(0.5,3.5,400)/0.529 # Angstrom to Bohr
    nstates = 2
    E_LIST_U    = np.zeros( (len(R_LIST), nstates+1) )
    E_LIST_R_S  = np.zeros( (len(R_LIST), nstates+1) )
    E_LIST_R_T  = np.zeros( (len(R_LIST), nstates+1) )
    for Ri,R in enumerate(R_LIST):
        print("R = %1.2f A" % (R*0.528))
        mol.atom = 'H 0 0 0; H 0 0 %1.2f' % (R)
        mol.build()
        E_UHF, S2, ss1, C_UHF, eps_HF = do_UHF( mol, return_wfn=True, return_MO_energies=True )
        E_UCIS                        = do_UCIS( mol, C_HF=C_UHF, eps_HF=eps_HF, nstates=nstates, spin_flipping=False )
        E_LIST_U[Ri,0]                = E_UHF
        E_LIST_U[Ri,1:]               = E_UCIS + E_UHF
        E_RHF, C_UHF, eps_HF          = do_RHF( mol, return_wfn=True, return_MO_energies=True )
        # E_RCIS                        = do_RCIS( mol, C_HF=C_UHF, eps_HF=eps_HF, nstates=nstates )
        E_LIST_R_S[Ri,0]              = E_RHF
        # E_LIST_R_S[Ri,1:]             = E_RCIS + E_RHF
        # E_RCIS                        = do_RCIS( mol, C_HF=C_UHF, eps_HF=eps_HF, nstates=nstates, symmetry="t" )
        # E_LIST_R_T[Ri,0]              = E_RHF
        # E_LIST_R_T[Ri,1:]             = E_RCIS + E_RHF
        # print( E_LIST_R_S[Ri,1:], E_LIST_R_T[Ri,1:] )
        print( E_LIST_U[Ri,:] )


    import matplotlib.pyplot as plt
    plt.plot( R_LIST*0.529, E_LIST_U[:,0], c="black", label="UHF" )
    plt.plot( R_LIST*0.529, E_LIST_R_S[:,0], "--", c="red", label="RHF" )
    for i in range(1,nstates+1):
        plt.plot( R_LIST*0.529, E_LIST_U[:,i], "-", c="black" )
        #plt.plot( R_LIST*0.529, E_LIST_R_S[:,i], "--", c="red" )
        #plt.plot( R_LIST*0.529, E_LIST_R_T[:,i], "--", c="red" )
    plt.legend()
    plt.xlim(R_LIST[0]*0.529,R_LIST[-1]*0.529)
    plt.ylim(-1.2,-0.2)
    plt.xlabel("$R_\\mathrm{HH}$ ($\\AA$)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.tight_layout()
    plt.savefig("UCIS_H2.jpg", dpi=300)

