import numpy as np
from opt_einsum import contract as opt_einsum # Very fast library for tensor contractions

from pyscf import gto

from pyscf_results import get_RCIS_pyscf

from RHF import do_RHF
from QED_RHF import do_QED_RHF
from tools import make_RDM1_ao
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

def compute_CIS_overlap( CIS_1, CIS_2, mo_1, mo_2, mol_1, mol_2 ):
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
 
def get_electric_dipole_moments( mol, C_HF, C_RCIS, o, v, do_Singlets, doPrint=True ):
    DIP           = np.zeros( (nstates+1,nstates+1,3) )
    S             = mol.intor('int1e_ovlp')
    s,u           = np.linalg.eigh(S)
    Shalf         = u @ np.diag(1/np.sqrt(s)) @ u.T
    dip_ao        = get_electric_dipole_ao( mol )
    dip_ao        = opt_einsum( "pi,dij,qj->dpq", Shalf, dip_ao, Shalf )
    dip_ao       *= -2 # Take electron to be negative and count 2 electrons for RCIS
    dip_mo        = opt_einsum( "pi,dpq,qj->dij", C_HF, dip_ao, C_HF )
    DIP[0,0,:]    = opt_einsum( "pi,dij,pj->d", C_HF[:,o], dip_mo[:,o,o], C_HF[:,o] )
    DIP[0,1:,:]   = (do_Singlets) * opt_einsum( "dov,ovS->Sd", dip_mo[:,o,v], C_RCIS )
    DIP[1:,0,:]   = DIP[0,1:]
    DIP[1:,1:,:]  = opt_einsum( "dvk,ovA,okB->ABd", dip_mo[:,v,v], C_RCIS[:,:,:], C_RCIS[:,:,:] )
    DIP[1:,1:,:] -= opt_einsum( "dou,ovA,uvB->ABd", dip_mo[:,o,o], C_RCIS[:,:,:], C_RCIS[:,:,:] )

    if ( doPrint == True ):
        print("Transition Electric Dipole Elements (a.u.):")
        for state in range(nstates+1):
            print("0 --> %d: %2.5f %2.5f %2.5f" % (state, DIP[0,state,0], DIP[0,state,1], DIP[0,state,2]) )
    return DIP

def get_magnetic_dipole_moments( mol, C_HF, C_RCIS, o, v, do_Singlets, doPrint=True  ):
    DIP           = np.zeros( (nstates+1,nstates+1,3) )
    S             = mol.intor('int1e_ovlp')
    s,u           = np.linalg.eigh(S)
    Shalf         = u @ np.diag(1/np.sqrt(s)) @ u.T
    dip_ao        = get_magnetic_dipole_ao( mol )
    dip_ao        = opt_einsum( "pi,dij,qj->dpq", Shalf, dip_ao, Shalf )
    dip_ao       *= -2 # Take electron to be negative and count 2 electrons for RCIS
    dip_mo        = opt_einsum( "pi,dpq,qj->dij", C_HF, dip_ao, C_HF )
    DIP[0,0,:]    = opt_einsum( "pi,dij,pj->d", C_HF[:,o], dip_mo[:,o,o], C_HF[:,o] )
    DIP[0,1:,:]   = (do_Singlets) * opt_einsum( "dov,ovS->Sd", dip_mo[:,o,v], C_RCIS )
    DIP[1:,0,:]   = DIP[0,1:]
    DIP[1:,1:,:]  = opt_einsum( "dvk,ovA,okB->ABd", dip_mo[:,v,v], C_RCIS[:,:,:], C_RCIS[:,:,:] )
    DIP[1:,1:,:] -= opt_einsum( "dou,ovA,uvB->ABd", dip_mo[:,o,o], C_RCIS[:,:,:], C_RCIS[:,:,:] )

    if ( doPrint == True ):
        print("Transition Magnetic Dipole Elements (a.u.):")
        for state in range(nstates+1):
            print("0 --> %d: %1.5f %1.5f %1.5f" % (state, DIP[0,state,0], DIP[0,state,1], DIP[0,state,2]) )
    return DIP

def get_electric_quadrupole_moments( mol, C_HF, C_RCIS, o, v, do_Singlets, doPrint=False  ):
    QUAD             = np.zeros( (nstates+1,nstates+1,3,3) )
    S                = mol.intor('int1e_ovlp')
    s,u              = np.linalg.eigh(S)
    Shalf            = u @ np.diag(1/np.sqrt(s)) @ u.T
    quad_ao          = get_electric_quadrupole_ao( mol )
    quad_ao          = opt_einsum( "pi,xyij,qj->xypq", Shalf, quad_ao, Shalf )
    quad_ao         *= -2 # Take electron to be negative and count 2 electrons for RCIS
    quad_mo          = opt_einsum( "pi,xypq,qj->xyij", C_HF, quad_ao, C_HF )
    QUAD[0,0,  :,:]  = opt_einsum( "pi,xyij,pj->xy", C_HF[:,o], quad_mo[:,:,o,o], C_HF[:,o] )
    QUAD[0,1:, :,:]  = QUAD[0,1:,:] = (do_Singlets) * opt_einsum( "xyov,ovS->Sxy", quad_mo[:,:,o,v], C_RCIS )
    QUAD[1:,1:,:,:]  = opt_einsum( "xyvk,ovA,okB->ABxy", quad_mo[:,:,v,v], C_RCIS[:,:,:], C_RCIS[:,:,:] )
    QUAD[1:,1:,:,:] -= opt_einsum( "xyou,ovA,uvB->ABxy", quad_mo[:,:,o,o], C_RCIS[:,:,:], C_RCIS[:,:,:] )

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

def do_QED_RCIS( mol, LAM, WC, do_CS=True, C_HF=None, eps_HF=None, return_wfn=False, nstates=1, symmetry="s", calc_moments=False, calc_photon_number=False ):
    DSE_FACTOR = 0.5 * LAM**2

    if ( symmetry[0].lower() == "s" ):
        do_Singlets = True
    elif ( symmetry[0].lower() == "t" ):
        do_Singlets = False

    # Get RHF wavefunction
    if ( C_HF is None or len(C_HF.shape) == 3 ):
        E_HF, C_HF, eps_HF = do_QED_RHF( mol, LAM=LAM, WC=WC, return_wfn=True, return_MO_energies=True )

    # Get the h1e and ERIs in orthogonal AO basis
    h1e, eri, n_elec_alpha, n_elec_beta, nuclear_repulsion_energy = get_ao_integrals( mol )
    assert( n_elec_alpha == n_elec_beta ), "RCIS only implemented for closed-shell systems"
    
    # Get the electric dipole and quadrupole in the orthogonal AO basis
    dip_ao, quad_ao = get_electric_dipole_ao( mol ), get_electric_quadrupole_ao( mol )
    EPOL    = np.array([0,0,1])
    dip_ao  = np.einsum("x,xpq->pq", EPOL, dip_ao)
    quad_ao = np.einsum("x,xypq,y->pq", EPOL, quad_ao, EPOL)

    # Get average dipole moment for CS shift, if used
    D          = make_RDM1_ao( C_HF, (np.arange(n_elec_alpha)) )
    AVEdipole  = (do_CS) * np.einsum( 'pq,pq->', D, dip_ao )

    # Convert the h1e and ERIs integrals to canonical MO basis
    #h1e = opt_einsum( "pi,pq,qj->ij", C_HF, h1e, C_HF ) # I guess we don't need this if we use eps_HF
    eri     = opt_einsum( "pi,qj,pqrs,rk,sl->ijkl", C_HF, C_HF, eri, C_HF, C_HF )
    dip_mo  = opt_einsum( "pi,pq,qj->ij", C_HF, dip_ao, C_HF )
    quad_mo = opt_einsum( "pi,pq,qj->ij", C_HF, quad_ao, C_HF )
    eri_DSE = 2 * DSE_FACTOR  * np.einsum( 'ij,kl->ijkl', dip_ao, dip_ao )

    # Get number of occupied and virtual orbitals
    n_occ = n_elec_alpha
    n_vir = len(h1e) - n_occ

    # Get the indices of the occupied and virtual orbitals
    o = slice( 0, n_occ )
    v = slice( n_occ, n_occ + n_vir )

    # Get the slices of the occupied and virtual orbitals energies
    eps_occ = eps_HF[o]
    eps_vir = eps_HF[v]

    ###### HERE IS THE EDUCATIONAL APPROACH ######
    ###### Build the Hamiltonian using for-loops ######
    ovov     = eri[o,v,o,v]
    oovv     = eri[o,o,v,v]
    ovov_DSE = eri_DSE[o,v,o,v]
    oovv_DSE = eri_DSE[o,o,v,v]

    if ( do_Singlets == True ):

        H_MM  = np.zeros( (n_occ, n_vir, n_occ, n_vir) )
        H_MM += np.diag((eps_vir[:,None] - eps_occ).flatten() ).reshape((n_occ,n_vir,n_occ,n_vir))
        H_MM += 2.0 * np.einsum( 'iajb->iajb', eri    [o,v,o,v] ) - np.einsum( 'ijba->iajb', eri    [o,o,v,v] )
        H_MM += 2.0 * np.einsum( 'iajb->iajb', eri_DSE[o,v,o,v] ) - np.einsum( 'ijba->iajb', eri_DSE[o,o,v,v] )    
        H_MM  = H_MM.reshape( (n_occ*n_vir, n_occ*n_vir) )

        H_MP           = np.zeros( (n_occ, n_vir, 1, 1) ) # Only single mode for now
        H_MP[:,:,0,0] += np.sqrt(WC / 2) * LAM * dip_mo[o,v]
        H_MP           = H_MP.reshape( (n_occ*n_vir, 1*1) )

        H_PP  = np.zeros( (1, 1, 1, 1) ) # Only single mode for now
        H_PP[0,0,0,0] = WC # Singlet ground state energy (Egs = 0) + 1 photon frequency
        H_PP  = H_PP.reshape( (1*1,1*1) )

        H = np.block( [[H_MM,H_MP],[H_MP.T,H_PP]] )

    else:
        H_MM  = np.zeros( (n_occ, n_vir, n_occ, n_vir) )
        H_MM += np.diag((eps_vir[:,None] - eps_occ).flatten() ).reshape((n_occ,n_vir,n_occ,n_vir))
        H_MM += -1 * np.einsum( 'ijba->iajb', eri    [o,o,v,v] )
        H_MM += -1 * np.einsum( 'ijba->iajb', eri_DSE[o,o,v,v] )    
        H_MM  = H_MM.reshape( (n_occ*n_vir, n_occ*n_vir) )
        H     = H_MM

    E_RCIS, C_RCIS = np.linalg.eigh( H )
    if ( nstates > len(E_RCIS) ):
        print("Only %d states are available with current basis set. Reducing output." % (len(E_RCIS) ) )
        nstates = len(E_RCIS)

    E_RCIS = E_RCIS[:nstates]
    C_RCIS = C_RCIS[:,:nstates]
    #print("IO Excitation:  %1.5f " % (eps_HF[1] - eps_HF[0]) )
    #print("RCIS " + "Triplets" * (not do_Singlets) + "Singlets" * (do_Singlets) + "          ", np.round(E_RCIS,5) )
    #print( E_RCIS )


    ##### Normalize the RCIS wavefunction #####
    NORM    = np.einsum("aS,aS->S", C_RCIS, C_RCIS)
    #C_RCIS *= np.sqrt( 2 / NORM ) # sqrt(2) * psi

    out_list = [E_RCIS]
    if ( return_wfn == True ):
        out_list.append([C_RCIS,C_HF])

    if ( calc_moments == True ):
        EL_DIP  = get_electric_dipole_moments( mol, C_HF, C_RCIS, o, v, do_Singlets  )
        MAG_DIP = get_magnetic_dipole_moments( mol, C_HF, C_RCIS, o, v, do_Singlets  )
        EL_QUAD = get_electric_quadrupole_moments( mol, C_HF, C_RCIS, o, v, do_Singlets  )
        out_list.append( [EL_DIP, MAG_DIP, EL_QUAD] )
    
    if ( calc_photon_number == True ):
        if ( do_Singlets == False ):
            print("\tTriplet calculations do not include the photon excitation.")
            print("\tLeft for future development to inclde |T_0, 1> basis states")
            out_list.append( np.zeros( (nstates) ) )
        else:
            start = n_occ*n_vir
            PHOT  = np.einsum("xi->i", np.abs(C_RCIS[start:,:])**2 )
            out_list.append( PHOT )

    if ( len(out_list) == 1 ):
        return E_RCIS
    return out_list

if ( __name__ == "__main__"):
    from RCIS import do_RCIS
    import matplotlib.pyplot as plt

    LAM = 0.02

    mol = gto.Mole()
    mol.basis = "sto3g"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'H 0 0 0; H 0 0 2.0'
    mol.verbose = 0
    mol.build()

    E_RHF, C_RHF = do_RHF( mol, return_wfn=True )
    E_RCIS       = do_RCIS( mol )

    nstates = 2
    WC_LIST = np.arange(0.5, 0.85, 0.01 )
    E_LIST  = np.zeros( (len(WC_LIST), nstates+1) )

    for WCi,WC in enumerate( WC_LIST ):
        E_QED_RHF, C_QED_RHF     = do_QED_RHF( mol, LAM, WC, return_wfn=True )
        E_CIS_S, (C_CIS_S, C_HF) = do_QED_RCIS( mol, LAM=LAM, WC=WC, nstates=nstates, symmetry="s", return_wfn=True )
        E_LIST[WCi,0]            = E_QED_RHF
        E_LIST[WCi,1:]           = E_CIS_S[:] + E_QED_RHF

    for state in range(len(E_RCIS)):
        plt.plot( WC_LIST, WC_LIST*0 + E_RCIS[state], c='black', alpha=0.5, lw=8, label="RCIS" )
    for state in range(1,nstates+1):
        plt.plot( WC_LIST, E_LIST[:,state] - E_LIST[:,0], c='red', label="QED-RCIS" * (state==1) )
    plt.legend()
    plt.xlabel("Cavity Frequency, $\\omega_\\mathrm{c}$ (a.u.)", fontsize=15)
    plt.ylabel("Transition Energy (a.u.)", fontsize=15)
    plt.title("$\\lambda$ = %1.3f a.u." % (LAM), fontsize=15)
    plt.tight_layout()
    plt.savefig("QED_RCIS_WC.jpg", dpi=300)
    plt.clf()


    nstates  = 1
    E_RCIS   = do_RCIS( mol, nstates=nstates )
    WC       = E_RCIS[0]
    nstates  = 2
    LAM_LIST = np.arange(0.0, 0.3, 0.01 )
    E_LIST   = np.zeros( (len(LAM_LIST), nstates+1) )
    for LAMi,LAM in enumerate( LAM_LIST ):
        E_QED_RHF, C_QED_RHF      = do_QED_RHF( mol, LAM, WC, return_wfn=True )
        E_QED_RCIS                = do_QED_RCIS( mol, LAM=LAM, WC=WC, nstates=nstates )
        E_LIST[LAMi,0]            = E_QED_RHF
        E_LIST[LAMi,1:]           = E_QED_RCIS[:] + E_QED_RHF

    plt.plot( LAM_LIST, E_LIST[:,0], label="QED-RHF %d" % (0) )
    for state in range(1,nstates+1):
        plt.plot( LAM_LIST, E_LIST[:,state], label="QED-RCIS %d" % (state) )
    plt.legend()
    plt.xlabel("Cavity Frequency, $\\omega_\\mathrm{c}$ (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$\\omega_\\mathrm{c}$ = %1.3f a.u. (Resonance)" % (WC), fontsize=15)
    plt.tight_layout()
    plt.savefig("QED_RCIS_LAM.jpg", dpi=300)
    plt.clf()



    LAM = 0.2
    WC  = 0.60
    nstates = 1
    R_LIST       = np.arange(1.0, 5.0, 0.05 )
    E_RCIS_S      = np.zeros( (len(R_LIST), nstates+1) )
    E_RCIS_T      = np.zeros( (len(R_LIST), nstates+1) )
    E_QED_RCIS_S  = np.zeros( (len(R_LIST), nstates+2) )
    E_QED_RCIS_T  = np.zeros( (len(R_LIST), nstates+1) )

    for Ri,R in enumerate( R_LIST ):
        mol.atom = 'H 0 0 0; H 0 0 %1.8f' % (R)
        mol.build()
        E_RCIS_S[Ri,0]      = do_RHF( mol )[0]
        E_RCIS_S[Ri,1:]     = E_RCIS_S[Ri,0] + do_RCIS( mol, nstates=nstates )
        E_QED_RCIS_S[Ri,0]  = do_QED_RHF( mol, LAM=LAM, WC=WC )[0]
        E_QED_RCIS_S[Ri,1:] = E_QED_RCIS_S[Ri,0] + do_QED_RCIS( mol, LAM=LAM, WC=WC, nstates=nstates+1 )

        E_RCIS_T[Ri,0]      = E_RCIS_S[Ri,0]
        E_RCIS_T[Ri,1:]     = E_RCIS_T[Ri,0] + do_RCIS( mol, nstates=nstates, symmetry="t" )
        E_QED_RCIS_T[Ri,0]  = E_QED_RCIS_S[Ri,0]
        E_QED_RCIS_T[Ri,1:] = E_QED_RCIS_T[Ri,0] + do_QED_RCIS( mol, LAM=LAM, WC=WC, nstates=nstates, symmetry="t" )

    for state in range(nstates+1):
        plt.plot( R_LIST, E_RCIS_S[:,state], c='black', alpha=0.3, lw=8, label="RCIS (S)"*(state==0) )
    
    for state in range(nstates+2):
        plt.plot( R_LIST, E_QED_RCIS_S[:,state] - WC/2, c='black', lw=2, label="QED-RCIS (S)"*(state==0) )
    
    for state in range(1,nstates+1):
        plt.plot( R_LIST, E_RCIS_T[:,state], "-", c='blue', alpha=0.3, lw=8, label="RCIS (T)"*(state==1) )
    
    for state in range(1,nstates+1):
        plt.plot( R_LIST, E_QED_RCIS_T[:,state] - WC/2, "-", c='blue', lw=2, label="QED-RCIS (T)"*(state==1) )
    
    plt.legend()
    plt.xlabel("Interatomic Distance (Bohr)", fontsize=15)
    plt.ylabel("Transition Energy (a.u.)", fontsize=15)
    plt.title("$\\omega_\\mathrm{c}$ = %1.3f a.u.    $\\lambda$ = %1.3f a.u." % (WC,LAM), fontsize=15)
    plt.tight_layout()
    plt.savefig("QED_RCIS_HH.jpg", dpi=300)