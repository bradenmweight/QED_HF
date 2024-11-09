import numpy as np
from pyscf import gto, scf, fci

def do_RHF_pyscf( mol ):
    myRHF = scf.RHF( mol )
    e_rhf = myRHF.kernel()
    print('    * RHF Total Energy (PySCF) : %20.12f' % (e_rhf))
    return e_rhf

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
    else:
        e_uhf = mf2.e_tot
    print('    * UHF Total Energy (PySCF) : %20.12f' % (e_uhf))
    return e_uhf

def do_FCI_pyscf( mol ):
    myRHF = scf.RHF( mol )
    e_fci = fci.FCI( myRHF ).kernel()[0]
    print('    * FCI Total Energy (PySCF): %20.12f' % (e_fci))
    return e_fci

if ( __name__ == "__main__" ):
    pass