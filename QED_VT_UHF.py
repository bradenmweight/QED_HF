import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from opt_einsum import contract as opt_einsum # Very fast library for tensor contractions

from pyscf import gto, scf, fci

from tools import get_spin_analysis, get_JK, to_ortho_ao, from_ortho_ao, eigh, make_RDM1_ao, do_DAMP, do_Max_Overlap_Method
from ao_ints import get_ao_integrals, get_dipole_quadrupole
from DIIS import DIIS


def get_h1e_eri_f( f, WC, Emu, h1e, eri ):
    # # Construct and apply the X-operator (X. Li and Y. Zhang arXiv) in orthogonal AO basis
    Z    = f**2 / 4 / WC
    Emu_minus_Emu  = np.array([  p - q for p in Emu for q in Emu ]).reshape( (len(Emu),len(Emu)) )
    Emu_minus_Emu2 = np.array([  p - q + r - s for p in Emu for q in Emu for r in Emu for s in Emu ]).reshape( (len(Emu),len(Emu),len(Emu),len(Emu)) )
    G2   = np.exp( -Z * Emu_minus_Emu **2 )
    G4   = np.exp( -Z * Emu_minus_Emu2**2 )
    h1e  = np.einsum( "pq,pq->pq", h1e, G2 ) # <p,0|X.T.conj() @ h1e @ X|p,0>
    eri  = np.einsum( "pqrs,pqrs->pqrs", eri, G4 ) # <p,0|X.T.conj() X.T.conj() @ eri @ X @ X|p,0>
    return h1e, eri, G2, G4

# def get_Gradient( E0, mol, LAM, WC, f ):
#     df = 1e-3
#     EF = __do_QED_VT_RHF_f( mol, LAM, WC, f=f+df )
#     GRAD = (EF - E0) / df
#     if ( abs(GRAD) > 1 ):
#         print("\tFound large gradient: %1.4f" % GRAD)
#         print("\tTrying to refine gradient with central difference")
#         EF = __do_QED_VT_RHF_f( mol, LAM, WC, f=f+2*df )
#         EB = __do_QED_VT_RHF_f( mol, LAM, WC, f=f-2*df )
#         GRAD = (EF - EB) / 2 / (2*df)
#         print("\tRefined gradient %1.4f" % GRAD)
#     if ( abs(GRAD) > 1 ):
#         print("\t\tTrying second-order central difference...")
#         EBB = __do_QED_VT_RHF_f( mol, LAM, WC, f=f-2*df )
#         EFF = __do_QED_VT_RHF_f( mol, LAM, WC, f=f+2*df )
#         EB  = __do_QED_VT_RHF_f( mol, LAM, WC, f=f-df )
#         EF  = __do_QED_VT_RHF_f( mol, LAM, WC, f=f+df )
#         GRAD = (-EFF + 8*EF - 8 * EB + EFF) / 12 / df
#         print("\t\tFinal Refined gradient %1.4f" % GRAD)
    
#     return GRAD

def get_Gradient( E0, mol, LAM, WC, f ):
    df = 1e-3
    EF = __do_QED_VT_RHF_f( mol, LAM, WC, f=f+df )
    EB = __do_QED_VT_RHF_f( mol, LAM, WC, f=f-df )
    GRAD = (EF - EB) / 2 / df
    return GRAD, EF, EB

def get_Hessian( E0, EF, EB, mol, LAM, WC, f ):
    df = 1e-3
    #EF = __do_QED_VT_RHF_f( mol, LAM, WC, f=f+df )
    #EB = __do_QED_VT_RHF_f( mol, LAM, WC, f=f-df )
    HESS = (EF - 2*E0 + EB) / df**2
    return HESS

def do_gradient_descent( mol, LAM, WC ):
    E_list = []
    f_list = []
    f = LAM/2 # Set to good initial guess
    E0 = __do_QED_VT_RHF_f( mol, LAM, WC, f=f )
    E_list.append( E0 )
    f_list.append( f )
    iteration = 1
    while ( True ):
        GRAD, _, _ = get_Gradient( E0, mol, LAM, WC, f )
        f    = f - 0.01 * GRAD
        if ( f/LAM > 1.0 or f/LAM < 0.0 ):
            f = f - 0.01 * (-1)*np.sign(GRAD) # Walk the other way this time without using GRAD's value in case it is bogus

        E1 = __do_QED_VT_RHF_f( mol, LAM, WC, f=f )
        print( "fold / LAM = %1.6f, E = %1.8f, dE = %1.8f" % (f/LAM, E1, E1-E0) )
        if ( abs(E1 - E0) < 1e-8 ):
            print( "f / LAM = %1.4f, E = %1.12f" % (f / LAM, E1) )
            return E_list, f_list
        E_list.append( E1 )
        f_list.append( f )
        E0 = E1
        iteration += 1
    return E_list, f_list

def do_Newton_Raphson( mol, LAM, WC ):
    """
    It seems that the energy is always nearly quadratic in the shift parameter f.
    Newton-Raphson should be a good method to minimize the energy in this case since it is exact for a parabola.
    """
    E_list = []
    f_list = []
    f = LAM/2 # Set to good initial guess
    E0 = __do_QED_VT_RHF_f( mol, LAM, WC, f=f )
    E_list.append( E0 )
    f_list.append( f )
    iteration = 1
    print( "f / LAM = %1.6f, E = %1.8f" % (f/LAM, E0) )
    while ( True ):
        GRAD, EF, EB = get_Gradient( E0, mol, LAM, WC, f )
        HESS         = get_Hessian( E0, EF, EB, mol, LAM, WC, f )
        f    = f - GRAD / HESS * ( np.random.rand() * (iteration > 5) + 1 * (iteration <=5) )
        if ( f/LAM > 1.0 or f/LAM < 0.0 ):
            f = f - 0.01 * (-1)*np.sign(GRAD) # Walk the other way this time without using GRAD's value in case it is bogus

        E1 = __do_QED_VT_RHF_f( mol, LAM, WC, f=f )
        print( "f / LAM = %1.6f, E = %1.8f, dE = %1.8f" % (f/LAM, E1, E1-E0) )
        if ( abs(E1 - E0) < 1e-8 ):
            print( "f / LAM = %1.4f, E = %1.12f" % (f / LAM, E1) )
            return E_list, f_list
        E_list.append( E1 )
        f_list.append( f )
        E0 = E1
        iteration += 1
    return E_list, f_list

def do_QED_VT_UHF( mol, LAM, WC, f=None ):

    if ( f is None ):
        if ( LAM == 0.0 ): return [[__do_QED_VT_RHF_f( mol, LAM, WC, f=0.0 )], [0.0]]
        #return do_gradient_descent( mol, LAM, WC )
        return do_Newton_Raphson( mol, LAM, WC )
    else:
        print("Doing single point calculation with f = %1.4f" % f)
        return __do_QED_VT_RHF_f( mol, LAM, WC, f=f )

def __do_QED_VT_RHF_f( mol, LAM, WC, f=None, do_CS=True, initial_guess=None ):

    S, Shalf, h1e, eri, n_elec_alpha, n_elec_beta, nuclear_repulsion_energy = get_ao_integrals( mol )
    dip_ao, quad_ao = get_dipole_quadrupole( mol, Shalf.shape[0] )

    # Rotate all relevant integrals to the orthogonal AO basis
    h1e     = to_ortho_ao( Shalf, h1e, shape=2 )
    eri     = opt_einsum( 'ap,bq,abcd,cr,ds->pqrs', Shalf, Shalf, eri, Shalf, Shalf )
    dip_ao  = to_ortho_ao( Shalf, dip_ao[-1,:,:], shape=2 )
    quad_ao = to_ortho_ao( Shalf, quad_ao[-1,-1,:,:], shape=2 )

    # # Diagonalize the dipole operator
    Emu, Umu = np.linalg.eigh( dip_ao )

    # # Rotate all relevant integrals to the dipole basis
    h1e     = opt_einsum( "ap,ab,bq->pq", Umu, h1e, Umu )
    eri     = opt_einsum( "ap,bq,abcd,cr,ds->pqrs", Umu, Umu, eri, Umu, Umu )
    dip_ao  = opt_einsum( "ap,ab,bq->pq", Umu, dip_ao, Umu )
    quad_ao = opt_einsum( "ap,ab,bq->pq", Umu, quad_ao, Umu )

    # # Apply VT-transformation to h1e and eri
    h1e, eri, G2, G4 = get_h1e_eri_f( f, WC, Emu, h1e, eri )

    if ( initial_guess is None ):
        # Choose core as guess for Fock matrix
        F_a        = h1e
        eps_a, C_a = eigh( F_a )
        C_b        = C_a.copy()
        D_a        = make_RDM1_ao( C_a, n_elec_alpha )
        D_b        = make_RDM1_ao( C_b, n_elec_beta )
    else:
        # Use initial guess to construct the Fock matrix
        C_a, C_b   = initial_guess
        D_a        = make_RDM1_ao( C_a, n_elec_alpha )
        D_b        = make_RDM1_ao( C_b, n_elec_beta )
        AVEdipole  = (do_CS) * np.einsum( 'pq,pq->', D_a + D_b, dip_ao[:,:] )
        
        DSE_FACTOR = 0.5 * LAM**2
        h1e_DSE =     DSE_FACTOR * ( -2*AVEdipole * dip_ao[:,:] + quad_ao[:,:] ) 
        eri_DSE = 2 * DSE_FACTOR  * np.einsum( 'pq,rs->pqrs', dip_ao[:,:], dip_ao[:,:] )

        # Coulomb and Exchange Matrices
        J_a, K_a         = get_JK( D_a, eri )
        J_b, K_b         = get_JK( D_b, eri )
        DSE_J_a, DSE_K_a = get_JK( D_a, eri_DSE )
        DSE_J_b, DSE_K_b = get_JK( D_b, eri_DSE )

        # Fock matrix
        F_a  = h1e     + J_a + J_b - K_a
        F_a += h1e_DSE + DSE_J_a + DSE_J_b - DSE_K_a
        F_b  = h1e     + J_a + J_b - K_b
        F_b += h1e_DSE + DSE_J_a + DSE_J_b - DSE_K_b

        eps_a, C_a = eigh( F_a )
        eps_b, C_b = eigh( F_b )
        D_a        = make_RDM1_ao( C_a, n_elec_alpha )
        D_b        = make_RDM1_ao( C_b, n_elec_beta )

    e_convergence = 1e-8
    d_convergence = 1e-6
    maxiter       = 2000

    old_energy  = np.einsum("ab,ab->", D_a, h1e )
    old_energy += np.einsum("ab,ab->", D_b, h1e )
    old_energy += nuclear_repulsion_energy
    old_C_a = C_a.copy()
    old_C_b = C_b.copy()
    old_D_a = D_a.copy()
    old_D_b = D_b.copy()
    old_F_a = F_a.copy()
    old_F_b = F_a.copy()
    old_dE  = 0.0
    old_dD  = 0.0
    DIIS_flag = False
    MOM_flag  = False

    myDIIS = DIIS( ao_overlap=S, unrestricted=True )

    for iter in range( maxiter ):

        AVEdipole = (do_CS) * np.einsum( 'pq,pq->', D_a + D_b, dip_ao[:,:] )
        DSE_FACTOR = 0.5 * (LAM - f)**2
        h1e_DSE    =     DSE_FACTOR * ( -2*AVEdipole * dip_ao[:,:] + quad_ao[:,:] ) 
        eri_DSE    = 2 * DSE_FACTOR * np.einsum( 'pq,rs->pqrs', dip_ao[:,:], dip_ao[:,:] )

        # Coulomb and Exchange Matrices
        J_a, K_a         = get_JK( D_a, eri )
        J_b, K_b         = get_JK( D_b, eri )
        DSE_J_a, DSE_K_a = get_JK( D_a, eri_DSE )
        DSE_J_b, DSE_K_b = get_JK( D_b, eri_DSE )

        # Fock matrix
        F_a  = h1e     + J_a + J_b - K_a
        F_b  = h1e     + J_a + J_b - K_b
        F_a += h1e_DSE + DSE_J_a + DSE_J_b - DSE_K_a
        F_b += h1e_DSE + DSE_J_a + DSE_J_b - DSE_K_b
        
        if ( iter < 100 ):
            F_a = do_DAMP( F_a, old_F_a )
            F_b = do_DAMP( F_b, old_F_b )

        # Diagonalize Fock matrix
        eps_a, C_a = eigh( F_a )
        eps_b, C_b = eigh( F_b )

        if ( iter == 5 ):
            # Break symmetry by mixing a-HOMO and a-LUMO
            C_b    = C_a.copy()
            angle   = np.pi/5
            HOMO_a = C_a[:,n_elec_alpha-1]
            LUMO_a = C_a[:,n_elec_alpha+0]
            C_a[:,n_elec_alpha-1] = HOMO_a * np.cos(angle) + LUMO_a * np.sin(angle)
            C_b[:,n_elec_beta-1]  = HOMO_a * np.cos(angle) - LUMO_a * np.sin(angle)


        # Get density matrix in AO basis
        occ_inds_a = (np.arange(n_elec_alpha))
        occ_inds_b = (np.arange(n_elec_beta))
        D_a = make_RDM1_ao( C_a, occ_inds_a )
        D_b = make_RDM1_ao( C_b, occ_inds_b )

        # Get current energy for RHF
        energy  = np.einsum("ab,ab->", D_a, h1e + 0.5*(J_a + J_b - K_a) )
        energy += np.einsum("ab,ab->", D_b, h1e + 0.5*(J_b + J_a - K_b) )
        energy += np.einsum("ab,ab->", D_a, h1e_DSE + 0.5*(DSE_J_a + DSE_J_b - DSE_K_a) )
        energy += np.einsum("ab,ab->", D_b, h1e_DSE + 0.5*(DSE_J_b + DSE_J_a - DSE_K_b) )
        energy += nuclear_repulsion_energy
        #energy += DSE_FACTOR*AVEdipole**2
        energy += 0.5 * WC

        dE = energy - old_energy
        dD = np.linalg.norm( D_a - old_D_a ) + np.linalg.norm( D_b - old_D_b )

        old_energy = energy
        old_D_a    = D_a.copy()
        old_D_b    = D_b.copy()
        old_F_a    = F_a.copy()
        old_F_b    = F_b.copy()
        old_dE     = dE*1
        old_dD     = dD*1

        #if ( iter > 50 ):
        #    print("    QED-UHF Iteration %3d: Energy = %1.12f, dE = %1.8f, dD = %1.6f" % (iter, energy, dE, dD))

        if ( iter > 2 and abs(dE) < e_convergence and dD < d_convergence ):
            break
        if ( iter == maxiter-1 ):
            print("FAILURE: VT-QED-UHF DID NOT CONVERGE")
            return float('nan'), float('nan'), float('nan')
        
        if ( iter > 100 ): # Try doing DIIS
            DIIS_flag = True
        if ( iter > 150 ): # Try doing MOM
            MOM_flag = True
            DIIS_flag = False

    # Compute spin operators
    S2, ss1 = get_spin_analysis( C_a[:,:n_elec_alpha], C_b[:,:n_elec_beta] )
    print( "Spin Analsysis of UHF Wavefunction:" )
    print( "\t<S2>                = %1.4f" % (S2) )
    print( "\tMultiplicity s(s+1) = %1.4f" % (ss1) )

    #print('    * VT-QED-RHF Total Energy: %20.12f' % (energy))

    return energy, S2, ss1 

if (__name__ == '__main__' ):
    from matplotlib import pyplot as plt

    mol = gto.Mole()
    mol.basis = "ccpvdz"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'Li 0 0 0; H 0 0 3.0'
    mol.build()

    LAM = 0.5
    WC  = 0.1
    E_list, f_list = do_QED_VT_UHF( mol, LAM, WC )
    E      = np.array( E_list )
    f_list = np.array( f_list )
    print( "Final VT-QED-RHF Energy: %1.8f" % E[-1] )
    plt.plot( np.arange(len(E)), E, "-o" )
    plt.xlabel("Iteration", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$\\lambda$ = %1.2f a.u." % (LAM), fontsize=15)
    plt.tight_layout()
    plt.savefig("E_minimization.jpg", dpi=300)
    plt.clf()

    plt.plot( np.arange(len(f_list)), f_list/LAM, "-o" )
    plt.xlabel("Iteration", fontsize=15)
    plt.ylabel("Normalized Shift Parameter, $\\frac{f}{\\lambda}$", fontsize=15)
    plt.title("$\\lambda$ = %1.2f a.u." % (LAM), fontsize=15)
    plt.tight_layout()
    plt.savefig("f_minimization.jpg", dpi=300)
    plt.clf()

    # LAM    = 0.5
    # f_list = np.linspace(0.0, LAM, 21)
    # E      = np.zeros( (len(f_list)) )
    # for fi,f in enumerate(f_list):
    #     print("f = ", f)
    #     E[fi] = do_QED_VT_RHF( mol, LAM, 0.1, f=f )
    # print( E )
    # 
    # plt.plot( f_list / LAM, E, "-o" )
    # plt.xlabel("Normalized Shift Parameter, $\\frac{f}{\\lambda}$", fontsize=15)
    # plt.ylabel("Energy (a.u.)", fontsize=15)
    # plt.title("$\\lambda$ = %1.2f a.u." % (LAM), fontsize=15)
    # plt.tight_layout()
    # plt.savefig("E_f.jpg", dpi=300)