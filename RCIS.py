import numpy as np
from opt_einsum import contract as opt_einsum # Very fast library for tensor contractions

from pyscf import gto, scf, tdscf

from RHF import do_RHF
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
    S_AB                = get_off_diagonal_overlap( mol_1, mol_2 )
    S_1                 = mol_1.intor('int1e_ovlp')
    S_2                 = mol_2.intor('int1e_ovlp')
    s_1, u_1            = np.linalg.eigh(S_1)
    s_2, u_2            = np.linalg.eigh(S_2)
    Shalf_1             = u_1 @ np.diag(1/np.sqrt(s_1)) @ u_1.T
    Shalf_2             = u_2 @ np.diag(1/np.sqrt(s_2)) @ u_2.T

    n_ao   = mo_1.shape[0]
    mo_1_o = mo_1[:,o]
    mo_1_v = mo_1[:,v]
    mo_2_o = mo_2[:,o]
    mo_2_v = mo_2[:,v]

    S_AB_ORTHO = opt_einsum( "pi,pq,qj->ij", Shalf_1, S_AB, Shalf_2 )
    #OVLP       = 0.500 * opt_einsum( "ovJ,po,pv,pq,qo,qv,ovK->JK", CIS_1, mo_1_o, mo_1_v, S_AB_ORTHO, mo_2_o, mo_2_v, CIS_2 )
    OVLP       = 0.500 * opt_einsum( "ovJ,ovK->JK", CIS_1, CIS_2 ) # Does not acount for changes in AO or MO basis
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


def do_RCIS( mol, C_HF=None, eps_HF=None, return_wfn=False, nstates=1, symmetry="s", calc_moments=False ):

    if ( symmetry[0].lower() == "s" ):
        do_Singlets = True
    elif ( symmetry[0].lower() == "t" ):
        do_Singlets = False

    # Get RHF wavefunction
    if ( C_HF is None or len(C_HF.shape) == 3 ):
        E_HF, C_HF, eps_HF = do_RHF( mol, return_wfn=True, return_MO_energies=True )

    # Get the h1e and ERIs (and other integrals) in orthogonal AO basis
    h1e, eri, n_elec_alpha, n_elec_beta, nuclear_repulsion_energy = get_ao_integrals( mol )
    assert( n_elec_alpha == n_elec_beta ), "RCIS only implemented for closed-shell systems"
    
    # Convert the h1e and ERIs to canonical MO basis
    h1e = opt_einsum( "pi,pq,qj->ij", C_HF, h1e, C_HF ) # I guess we don't need this if we use eps_HF
    eri = opt_einsum( "pi,qj,pqrs,rk,sl->ijkl", C_HF, C_HF, eri, C_HF, C_HF )

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
    # ovov = eri[o,v,o,v]
    # oovv = eri[o,o,v,v]
    # H    = np.zeros( (n_occ,n_vir,n_occ,n_vir) )
    # for o1 in range(n_occ):
    #     for v1 in range(n_vir):
    #         H[o1,v1,o1,v1] += eps_vir[v1] - eps_occ[o1]
    #         for o2 in range(n_occ):
    #            for v2 in range(n_vir):
    #                H[o1,v1,o2,v2] += (do_Singlets) * 2.0 * ovov[o1,v1,o2,v2] - oovv[o1,o2,v1,v2]
    # H = H.reshape( (n_occ*n_vir, n_occ*n_vir) )
    ###############################################

    ###### HERE IS THE FAST APPROACH ######
    ###### Build the Hamiltonian without using for-loops ######
    H = np.diag((eps_vir[:,None] - eps_occ).flatten() )
    H += (do_Singlets) * 2.0 * eri[o,v,o,v].reshape( (n_occ*n_vir,n_occ*n_vir) )
    H -= eri[o,o,v,v].reshape( (n_occ*n_vir,n_occ*n_vir) )
    ###############################################

    E_RCIS, C_RCIS = np.linalg.eigh( H )
    #if ( nstates > len(E_RCIS) ):
    #    print("\tWarning!!!\n\tNumber of states requested is larger than the number of states available")
    #    print("\tSetting {nstates} to the number of states available")
    #    nstates = len(E_RCIS) # This already happens by default in numpy
    E_RCIS = E_RCIS[:nstates]
    C_RCIS = C_RCIS[:,:nstates]
    #print("IO Excitation:  %1.5f " % (eps_HF[1] - eps_HF[0]) )
    #print("RCIS " + "Triplets" * (not do_Singlets) + "Singlets" * (do_Singlets) + "          ", np.round(E_RCIS,5) )

    if ( nstates > len(E_RCIS) ):
        nstates = len(E_RCIS)

    ##### Normalize the RCIS wavefunction #####
    C_RCIS  = C_RCIS.reshape( (n_occ,n_vir,nstates) )
    NORM    = np.einsum("abS,abS->S", C_RCIS, C_RCIS)
    C_RCIS *= np.sqrt( 2 / NORM ) # sqrt(2) * psi

    out_list = [E_RCIS]
    if ( return_wfn == True ):
        out_list.append([C_RCIS,C_HF])

    if ( calc_moments == True ):
        EL_DIP  = get_electric_dipole_moments( mol, C_HF, C_RCIS, o, v, do_Singlets  )
        MAG_DIP = get_magnetic_dipole_moments( mol, C_HF, C_RCIS, o, v, do_Singlets  )
        EL_QUAD = get_electric_quadrupole_moments( mol, C_HF, C_RCIS, o, v, do_Singlets  )
        out_list.append( [EL_DIP, MAG_DIP, EL_QUAD] )
    
    if ( len(out_list) == 1 ):
        return E_RCIS
    return out_list


def get_CIS_pyscf( mol, nstates=1, symmetry="s" ):
    if ( symmetry[0].lower() == "s" ):
        singlet = True
    elif ( symmetry[0].lower() == "t" ):
        singlet = False
    # CIS calculation using PySCF
    mf = scf.RHF( mol )
    mf.kernel()
    myci = tdscf.TDHF( mf )
    myci.nroots = nstates
    myci.singlet = singlet
    myci.kernel()
    return mf.e_tot, myci.e_tot 

if ( __name__ == "__main__"):
    

    mol = gto.Mole()
    mol.basis = "sto3g"
    mol.unit = 'Bohr'
    mol.symmetry = False
    #mol.atom = 'H 0 0 0; H 0 0 2.0'
    mol.atom = 'Li 0 0 0; H 0 0 2.0'
    mol.verbose = 0
    mol.build()
    E, C = do_RHF( mol, return_wfn=True )
    #print( "RHF    (BMW)         : %1.5f" % (E) )

    nstates = 5

    E_RHF_pyscf, E_RCIS_S_pyscf = get_CIS_pyscf( mol, nstates=nstates, symmetry="s" )
    E_RHF_pyscf, E_RCIS_T_pyscf = get_CIS_pyscf( mol, nstates=nstates, symmetry="t" )
    E_RCIS_S = do_RCIS( mol, nstates=nstates, symmetry="s" )
    E_RCIS_T = do_RCIS( mol, nstates=nstates, symmetry="t" )

    print( "PySCF Singlet Transition Energy (eV): %1.6f" % ((E_RCIS_S_pyscf[0]-E_RHF_pyscf)*27.2114))
    print( "  BMW Singlet Transition Energy (eV): %1.6f" % ((E_RCIS_S[0])      *27.2114))
    if ( mol.basis == "sto3g" ):
        print( "  G16 Singlet Transition Energy (eV): %1.6f" % (18.3805))
    print("")
    print( "PySCF Triplet Transition Energy (eV): %1.6f" % ((E_RCIS_T_pyscf[0]-E_RHF_pyscf)*27.2114))
    print( "  BMW Triplet Transition Energy (eV): %1.6f" % ((E_RCIS_T[0])      *27.2114))
    if ( mol.basis == "sto3g" ):
        print( "  G16 Triplet Transition Energy (eV): %1.6f" % (7.4688))
    
    E_CIS_S, (C_CIS_S, C_HF), moments = do_RCIS( mol, nstates=nstates, symmetry="s", return_wfn=True, calc_moments=True )
    E_CIS_T, (C_CIS_T, C_HF)          = do_RCIS( mol, nstates=nstates, symmetry="t", return_wfn=True )

    compute_spatial_overlap( C_CIS_S, C_CIS_S, C_HF, C_HF, mol, mol )