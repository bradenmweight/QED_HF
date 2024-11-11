import numpy as np
from pyscf import gto, scf, fci, tdscf

def do_RHF_pyscf( mol ):
    myRHF = scf.RHF( mol )
    e_rhf = myRHF.kernel()
    print('    * RHF Total Energy (PySCF) : %20.12f' % (e_rhf))
    return e_rhf, myRHF

def do_UHF_pyscf( mol ):
    # UHF -- BMW: Need to break symmetry of initial guess to get right solution
    mf1 = scf.UHF(mol)
    dm_alpha, dm_beta = mf1.get_init_guess()
    dm_beta[:2,:2] = 0 # BMW: Set some of the beta coefficients to zero to break alpha/beta symmetry
    dm = (dm_alpha,dm_beta)
    mf1.kernel(dm) # BMW: Pass in modified initial guess
    mf2 = scf.UHF(mol)
    dm_alpha, dm_beta = mf2.get_init_guess()
    dm_beta[:,:] = 0 # BMW: Set some of the beta coefficients to zero to break alpha/beta symmetry
    dm = (dm_alpha,dm_beta)
    mf2.kernel(dm) # BMW: Pass in modified initial guess
    if ( mf1.e_tot < mf2.e_tot ): # BMW: Check which symmetry breaking works... H2 is mf1 but LiH is mf2
        e_uhf = mf1.e_tot
        mf    = mf1
    else:
        e_uhf = mf2.e_tot
        mf    = mf2
    #print('    * UHF Total Energy (PySCF) : %20.12f' % (e_uhf))
    return e_uhf, mf

def get_RCIS_pyscf( mol, nstates=1, symmetry="s" ):
    if ( symmetry[0].lower() == "s" ):
        singlet = True
    elif ( symmetry[0].lower() == "t" ):
        singlet = False
    # CIS calculation using PySCF
    mf = scf.RHF( mol )
    mf.kernel()
    #myci = tdscf.TDHF( mf )
    myci = tdscf.TDA( mf )
    myci.nroots = nstates
    myci.singlet = singlet
    myci.kernel()
    return mf.e_tot, myci.e_tot 

def get_UCIS_pyscf( mol, nstates=1 ):
    # CIS calculation using PySCF
    E, mf = do_UHF_pyscf( mol )
    #myci = tdscf.TDHF( mf )
    myci = tdscf.TDA( mf )
    myci.nroots = nstates
    #myci.singlet = True
    myci.kernel()
    #A,B = myci.get_ab()
    #A   = np.array(A)
    #print( "A\n", np.round( A[0].reshape((len(A),-1)), 3 ) )
    #print( "A\n", np.round( A[1].reshape((len(A),-1)), 3 ) )
    #print( "A\n", np.round( A[2].reshape((len(A),-1)), 3 ) )
    return mf.e_tot, myci.e_tot 

def do_FCI_pyscf( mol ):
    myRHF = scf.RHF( mol )
    e_fci = fci.FCI( myRHF ).kernel()[0]
    print('    * FCI Total Energy (PySCF): %20.12f' % (e_fci))
    return e_fci

if ( __name__ == "__main__" ):
    pass