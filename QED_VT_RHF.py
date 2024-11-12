import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from opt_einsum import contract as opt_einsum # Very fast library for tensor contractions

from pyscf import gto, scf, fci

from ao_ints import get_ao_integrals, get_electric_dipole_ao, get_electric_quadrupole_ao
from tools import get_JK, eigh, make_RDM1_ao, do_DAMP, do_Max_Overlap_Method
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

def get_Gradient( E0, mol, LAM, WC, f, do_CS=True, return_wfn=False, initial_guess=None ):
    df = 1e-3
    EF = __do_QED_VT_RHF_f( mol, LAM, WC, f=f+df )
    EB = __do_QED_VT_RHF_f( mol, LAM, WC, f=f-df )
    GRAD = (EF - EB) / 2 / df
    return GRAD, EF, EB

def get_Hessian( E0, EF, EB, mol, LAM, WC, f, do_CS=True, return_wfn=False, initial_guess=None ):
    df = 1e-3
    #EF = __do_QED_VT_RHF_f( mol, LAM, WC, f=f+df )
    #EB = __do_QED_VT_RHF_f( mol, LAM, WC, f=f-df )
    HESS = (EF - 2*E0 + EB) / df**2
    return HESS

def do_gradient_descent( mol, LAM, WC, do_CS=True, return_wfn=False, initial_guess=None ):
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

def do_Newton_Raphson( mol, LAM, WC, do_CS=True, return_wfn=False, initial_guess=None):
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
        f            = f - GRAD / HESS * ( 1e-3 * (iteration >= 5) + 1 * (iteration < 5) )
        if ( f/LAM > 1.0 or f/LAM < 0.0 ):
            f = f - 0.01 * (-1)*np.sign(GRAD) # Walk the other way this time without using GRAD's value in case it is bogus

        E1 = __do_QED_VT_RHF_f( mol, LAM, WC, f=f )
        print( "f / LAM = %1.6f, E = %1.8f, dE = %1.8f" % (f/LAM, E1, E1-E0) )
        E_list.append( E1 ) # BEFORE EXIT IF STATEMENT
        f_list.append( f ) # BEFORE EXIT IF STATEMENT
        if ( abs(E1 - E0) < 1e-6 ):
            print( "f / LAM = %1.4f, E = %1.12f" % (f / LAM, E1) )
            return E_list, f_list
        E0 = E1
        iteration += 1
    return E_list, f_list

def do_QED_VT_RHF( mol, LAM, WC, f=None, do_CS=True, return_wfn=False, initial_guess=None ):

    if ( f is None ):
        if ( LAM == 0.0 ): 
            E = __do_QED_VT_RHF_f( mol, LAM, WC, f=0.0, do_CS=do_CS, return_wfn=return_wfn, initial_guess=initial_guess )
            return [E], [0.0]
        #return do_gradient_descent( mol, LAM, WC, do_CS=do_CS, return_wfn=return_wfn, initial_guess=initial_guess )
        E_list, f_list = do_Newton_Raphson( mol, LAM, WC, do_CS=do_CS, return_wfn=return_wfn, initial_guess=initial_guess )
        return E_list, f_list
    else:
        print("Doing single point calculation with f = %1.4f" % f)
        E = __do_QED_VT_RHF_f( mol, LAM, WC, f=f, do_CS=do_CS, return_wfn=return_wfn, initial_guess=initial_guess )
        return E

def __do_QED_VT_RHF_f( mol, LAM, WC, f=None, do_CS=True, return_wfn=False, initial_guess=None ):

    h1e, eri, n_elec_alpha, n_elec_beta, nuclear_repulsion_energy, Shalf = get_ao_integrals( mol, return_Shalf=True )
    dip_ao, quad_ao = get_electric_dipole_ao( mol, Shalf=Shalf ), get_electric_quadrupole_ao( mol, Shalf=Shalf )
    EPOL    = np.array([0,0,1])
    dip_ao  = np.einsum("x,xpq->pq", EPOL, dip_ao)
    quad_ao = np.einsum("x,xypq,y->pq", EPOL, quad_ao, EPOL)

    # # Diagonalize the dipole operator
    Emu, Umu = np.linalg.eigh( dip_ao )

    # # Rotate all relevant integrals to the dipole basis
    h1e     = opt_einsum( "ap,ab,bq->pq", Umu, h1e, Umu )
    eri     = opt_einsum( "ap,bq,abcd,cr,ds->pqrs", Umu, Umu, eri, Umu, Umu )
    dip_ao  = opt_einsum( "ap,ab,bq->pq", Umu, dip_ao, Umu )
    quad_ao = opt_einsum( "ap,ab,bq->pq", Umu, quad_ao, Umu )

    # # Apply VT-transformation to h1e and eri
    h1e, eri, G2, G4 = get_h1e_eri_f( f, WC, Emu, h1e, eri )

    # Choose core as guess for Fock matrix
    F = h1e

    # Diagonalize Fock
    eps, C = eigh( F )

    # Get density matrix in AO basis
    D    = make_RDM1_ao( C, n_elec_alpha )

    e_convergence = 1e-8
    d_convergence = 1e-6
    maxiter       = 2000

    old_energy = np.einsum("ab,ab->", D, 2*h1e ) + nuclear_repulsion_energy
    old_D = D.copy()
    old_F = F.copy()
    old_C = C.copy()
    DIIS_flag = False
    MOM_flag  = False

    myDIIS = DIIS()

    for iter in range( maxiter ):

        # DSE
        AVEdipole  = (do_CS) * np.einsum( 'pq,pq->', D, dip_ao[:,:] )
        DSE_FACTOR = 0.5 * ( LAM - f )**2
        h1e_DSE    =     DSE_FACTOR * ( -2*AVEdipole * dip_ao[:,:] + quad_ao[:,:] ) 
        eri_DSE    = 2 * DSE_FACTOR  * np.einsum( 'pq,rs->pqrs', dip_ao[:,:], dip_ao[:,:] )

        # Coulomb matrix
        J     = np.einsum( 'rs,pqrs->pq', D, eri )
        DSE_J = np.einsum( 'rs,pqrs->pq', D, eri_DSE )

        # Exchange matrix
        K     = np.einsum( 'rs,prsq->pq', D, eri )
        DSE_K = np.einsum( 'rs,prsq->pq', D, eri_DSE )

        # Fock matrix
        F  = h1e     + 2 * J     - K
        F += h1e_DSE +     DSE_J - DSE_K
        
        if ( iter < 3 ):
            F = do_DAMP( F, old_F )

        if ( iter > 5 ):
            F = myDIIS.extrapolate( F, D )

        # Diagonalize Fock matrix
        eps, C = eigh( F )

        # Get density matrix in AO basis
        D = make_RDM1_ao( C, n_elec_alpha )

        # Get current energy for RHF
        energy  = np.einsum("ab,ab->", D, 2*h1e + 2*J - K )
        energy += np.einsum("ab,ab->", D, 2*h1e_DSE + 2*DSE_J - DSE_K )
        energy += nuclear_repulsion_energy
        #energy += DSE_FACTOR*AVEdipole**2
        energy += 0.5 * WC

        dE = energy - old_energy
        dD = np.linalg.norm( D - old_D )

        # if ( iter > 5 and dD > 1.0 ):            
        #    inds = do_Max_Overlap_Method( C, old_C, (np.arange(n_elec_alpha)) )
        #    C    = C[:,inds]
        #    D    = make_RDM1_ao( C, (np.arange(n_elec_alpha)) )
        #    dD   = 2 * np.linalg.norm( D - old_D )

        #print("    Iteration %3d: Energy = %1.12f, dE = %1.8f, dD = %1.6f" % (iter, energy, dE, dD))

        old_energy = energy
        old_D      = D.copy()

        if ( iter > 2 and abs(dE) < e_convergence and dD < d_convergence ):
            break
        if ( iter == maxiter-1 ):
            print("FAILURE: VT-QED-RHF DID NOT CONVERGE")
            return float('nan')

    #print('    * VT-QED-RHF Total Energy: %20.12f' % (energy))

    if ( return_wfn == True ):
        return energy, C
    return energy

if (__name__ == '__main__' ):
    from matplotlib import pyplot as plt

    mol = gto.Mole()
    mol.basis = "ccpvdz"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'H 0 0 0; H 0 0 2.0'
    mol.build()
    LAM = 0.5
    WC  = 1.0
    E    = do_QED_VT_RHF( mol, LAM, WC )
    E, C = do_QED_VT_RHF( mol, LAM, WC, return_wfn=True )
    E    = do_QED_VT_RHF( mol, LAM, WC, initial_guess=C )

    mol.atom = 'H 0 0 0; H 0 0 10.0'
    mol.build()
    E    = do_QED_VT_RHF( mol, LAM, WC )
    E    = do_QED_VT_RHF( mol, LAM, WC, initial_guess=C )





    # LAM = 0.5
    # WC  = 0.1
    # E_list, f_list = do_QED_VT_RHF( mol, LAM, WC )
    # E      = np.array( E_list )
    # f_list = np.array( f_list )
    # print( "Final VT-QED-RHF Energy: %1.8f" % E[-1] )
    # plt.plot( np.arange(len(E)), E, "-o" )
    # plt.xlabel("Iteration", fontsize=15)
    # plt.ylabel("Energy (a.u.)", fontsize=15)
    # plt.title("$\\lambda$ = %1.2f a.u." % (LAM), fontsize=15)
    # plt.tight_layout()
    # plt.savefig("E_minimization.jpg", dpi=300)
    # plt.clf()

    # plt.plot( np.arange(len(f_list)), f_list/LAM, "-o" )
    # plt.xlabel("Iteration", fontsize=15)
    # plt.ylabel("Normalized Shift Parameter, $\\frac{f}{\\lambda}$", fontsize=15)
    # plt.title("$\\lambda$ = %1.2f a.u." % (LAM), fontsize=15)
    # plt.tight_layout()
    # plt.savefig("f_minimization.jpg", dpi=300)
    # plt.clf()

    # # LAM    = 0.5
    # # f_list = np.linspace(0.0, LAM, 21)
    # # E      = np.zeros( (len(f_list)) )
    # # for fi,f in enumerate(f_list):
    # #     print("f = ", f)
    # #     E[fi] = do_QED_VT_RHF( mol, LAM, 0.1, f=f )
    # # print( E )
    # # 
    # # plt.plot( f_list / LAM, E, "-o" )
    # # plt.xlabel("Normalized Shift Parameter, $\\frac{f}{\\lambda}$", fontsize=15)
    # # plt.ylabel("Energy (a.u.)", fontsize=15)
    # # plt.title("$\\lambda$ = %1.2f a.u." % (LAM), fontsize=15)
    # # plt.tight_layout()
    # # plt.savefig("E_f.jpg", dpi=300)