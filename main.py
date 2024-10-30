import numpy as np
import subprocess as sp
from matplotlib import pyplot as plt

from pyscf import gto

from RHF import do_RHF
from UHF import do_UHF 
from QED_RHF import do_QED_RHF
from QED_UHF import do_QED_UHF

def get_Globals():

    global basis_set, do_coherent_state
    basis_set = 'ccpvdz'
    do_coherent_state = True

    global LAM_LIST, WC
    WC           = 0.1 # a.u.
    dL           = 0.025 # a.u.
    LAM_LIST     = np.arange(0.0, 0.5+dL, dL)

    global DATA_DIR
    DATA_DIR = "PLOTS_DATA"
    sp.call( f"mkdir -p {DATA_DIR}", shell=True )

def do_H2_Dissociation( mol ):

    R_LIST       = np.arange(0.5, 6.1, 0.025)

    QEDRHF_BRADEN = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDUHF_BRADEN = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    RHF_BRADEN    = np.zeros( len(R_LIST) )
    UHF_BRADEN    = np.zeros( len(R_LIST) )
    for Ri,R in enumerate( R_LIST ):
        mol.atom = 'H 0 0 0; H 0 0 %1.8f' % R
        mol.build()
        RHF_BRADEN[Ri]         = do_RHF( mol )
        UHF_BRADEN[Ri]         = do_UHF( mol )
        for LAMi,LAM in enumerate( LAM_LIST ):
            QEDRHF_BRADEN[LAMi,Ri] = do_QED_RHF( mol, LAM, WC, do_coherent_state=do_coherent_state )
            QEDUHF_BRADEN[LAMi,Ri] = do_QED_UHF( mol, LAM, WC, do_coherent_state=do_coherent_state )
    

    plt.plot( R_LIST, RHF_BRADEN[:] + WC/2, "-", lw=8, alpha=0.5, c="black", label="RHF" )
    plt.plot( R_LIST, UHF_BRADEN[:] + WC/2, "-", lw=8, alpha=0.5, c="blue", label="UHF" )
    for LAMi,LAM in enumerate( LAM_LIST ):
        plt.plot( R_LIST, QEDRHF_BRADEN[LAMi,:], "-", ms=3, markerfacecolor='none', c="black", label="QED-RHF" * (LAMi==0) )
        plt.plot( R_LIST, QEDUHF_BRADEN[LAMi,:], "-", ms=3, markerfacecolor='none', c="blue", label="QED-UHF" * (LAMi==0) )
    plt.xlabel("Nuclear Separation, $R$ (a.u.)", fontsize=15)
    plt.ylabel("Energy, $E_0$ (a.u.)", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/H2_PES.jpg", dpi=300)
    plt.clf()

    for LAMi,LAM in enumerate( LAM_LIST ):
        plt.semilogy( R_LIST, QEDRHF_BRADEN[LAMi,:] - QEDUHF_BRADEN[LAMi,:], "-", label="$\\lambda$ = %1.3f" % LAM )
    plt.xlabel("Nuclear Separation, $R$ (a.u.)", fontsize=15)
    plt.ylabel("$E_\\mathrm{QED-RHF}$ - $E_\\mathrm{QED-UHF}$ (a.u.)", fontsize=15)
    plt.ylim(1e-6,0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/H2_PES_RHF-UHF.jpg", dpi=300)
    plt.clf()

    CF_POINTS = np.zeros( len(LAM_LIST) )
    for LAMi,LAM in enumerate( LAM_LIST ):
        for Ri,R in enumerate( R_LIST ):
            if ( QEDRHF_BRADEN[LAMi,Ri] - QEDUHF_BRADEN[LAMi,Ri] > 1e-6 ):        
                CF_POINTS[LAMi] = R
    plt.plot( LAM_LIST, CF_POINTS, "o-" )
    plt.xlabel("Coupling Strength, $\\lambda$ (a.u.)", fontsize=15)
    plt.ylabel("Coulson-Fischer Points, $R_\\mathrm{CF}$ (a.u.)", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/H2_PES_CFPs.jpg", dpi=300)
    plt.clf()


if ( __name__ == "__main__" ):

    get_Globals()

    mol = gto.Mole()
    mol.basis = basis_set
    mol.unit = 'Bohr'
    mol.symmetry = False

    do_H2_Dissociation( mol )







    exit()



    dL           = 0.1
    LAM_LIST     = [0.0]#np.arange(0.0, 1+dL, dL)
    QEDHF_BRADEN = np.zeros_like(LAM_LIST)
    QEDHF_ZHY    = np.zeros_like(LAM_LIST)
    RHH          = 2.8
    WC           = 0.1
    #mol.atom     = 'H 0 0 0; H 0 0 %1.3f' % ( RHH )
    mol.atom     = 'Li 0 0 %1.3f; H 0 0 %1.3f' % ( -3/4*RHH, 1/4*RHH )
    #mol.atom     = 'Li 0 0 %1.3f; Li 0 0 %1.3f' % ( -2, 2 )
    mol.build()

    for LAMi,LAM in enumerate( LAM_LIST ):
        print("Working on LAM = %1.2f" % LAM)
        QEDHF_BRADEN[LAMi], QEDHF_ZHY[LAMi], e_rhf, e_uhf, e_fci = do_QED_RHF( mol, LAM, WC )
    
    plt.plot( LAM_LIST, LAM_LIST*0 + e_rhf, "-", c='black', lw=5, alpha=0.5, label="RHF (PySCF) + $\\frac{\\hbar \\omega}{2}$" )
    # plt.plot( LAM_LIST, LAM_LIST*0 + e_fci, "-", c='blue', lw=5, alpha=0.5, label="FCI (PySCF) + $\\frac{\\hbar \\omega}{2}$" )
    plt.plot( LAM_LIST, QEDHF_BRADEN, "-", c='black', label="QED-UHF (Braden)" )
    plt.plot( LAM_LIST, QEDHF_ZHY, "o", c='black', label="QED-RHF (Yu)" )
    # plt.plot( LAM_LIST, QEDHF_BRADEN - QEDHF_ZHY, "-", c='black', label="QED-RHF Yu" )
    plt.legend()
    plt.xlabel("Coupling Strength, $\\lambda$ (a.u.)", fontsize=15)
    plt.ylabel("Energy, $E_0$ (a.u.)", fontsize=15)
    plt.tight_layout()
    plt.savefig("H2_LAM_SCAN_QEDUHF.jpg", dpi=300)
    plt.clf()