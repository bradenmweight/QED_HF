import numpy as np
from matplotlib import pyplot as plt

from pyscf import gto, scf

from RHF import do_RHF
from UHF import do_UHF
from QED_RHF import do_QED_RHF, do_QED_HF_ZHY
from QED_UHF import do_QED_UHF
from QED_VT_RHF import do_QED_VT_RHF
from QED_VT_UHF import do_QED_VT_UHF

from pyscf_results import do_RHF_pyscf, do_UHF_pyscf


def main():
    mol = gto.Mole()
    mol.basis = "sto3g"
    mol.unit = 'Angstrom'
    mol.symmetry = False
    mol.atom = \
'''C 1.7913090545   -0.0745398644    0.0184596800
N 0.4277379156    0.4416819705    0.0067776063
N -0.4277379156   -0.4416819705    0.0067776063
C -1.7913090545    0.0745398644    0.0184596800
H 2.3109976118    0.3289846823   -0.8541305852
H 1.8171632620   -1.1680305269    0.0140500788
H 2.2884450671    0.3205358657    0.9079354075
H -2.3109976118   -0.3289846823   -0.8541305852
H -1.8171632620    1.1680305269    0.0140500788
H -2.2884450671   -0.3205358657    0.9079354075
'''
    mol.build()

    WC     = 0.5
    LAM    = 0.5
    f_list = np.linspace(0.0, LAM, 21)
    E_VT_QEDRHF       = np.zeros( (len(f_list)) )
    E_VT_QEDUHF       = np.zeros( (len(f_list)) )
    E_VT_S2           = np.zeros( (len(f_list)) )
    E_VT_ss1          = np.zeros( (len(f_list)) )
    E_RHF_pyscf       = scf.RHF( mol ).kernel()
    E_UHF_pyscf, _   = do_UHF_pyscf( mol )
    E_RHF             = do_RHF( mol )
    E_UHF, S2, ss1    = do_UHF( mol )
    E_QEDRHF          = do_QED_RHF( mol, LAM, WC )
    E_QEDRHF_ZHY      = do_QED_HF_ZHY( mol, LAM, WC )
    E_QEDUHF, S2, ss1 = do_QED_UHF( mol, LAM, WC )
    for fi,f in enumerate(f_list):
         E_VT_QEDRHF[fi]                            = do_QED_VT_RHF( mol, LAM, WC, f=f )
         E_VT_QEDUHF[fi], E_VT_S2[fi], E_VT_ss1[fi] = do_QED_VT_UHF( mol, LAM, WC, f=f )
    E_QED_CS_VT_RHF_OPT, f_QED_CS_VT_RHF_OPT        = do_QED_VT_RHF( mol, LAM, WC )
    E_QED_CS_VT_UHF_OPT, f_QED_CS_VT_UHF_OPT, _, _  = do_QED_VT_UHF( mol, LAM, WC )
    E_QED_CS_VT_RHF_OPT = E_QED_CS_VT_RHF_OPT[-1]
    E_QED_CS_VT_UHF_OPT = E_QED_CS_VT_UHF_OPT[-1]
    f_QED_CS_VT_RHF_OPT = f_QED_CS_VT_RHF_OPT[-1]
    f_QED_CS_VT_UHF_OPT = f_QED_CS_VT_UHF_OPT[-1]
    print( E_RHF )
    print( E_UHF, S2, ss1 )
    print( E_RHF_pyscf )
    print( E_UHF_pyscf )
    print( E_QEDRHF )
    print( E_QEDRHF_ZHY )
    print( E_QEDUHF )
    print( E_VT_QEDRHF )
    print( E_VT_QEDUHF )
    print( E_QED_CS_VT_RHF_OPT, f_QED_CS_VT_RHF_OPT / LAM )
    print( E_QED_CS_VT_UHF_OPT, f_QED_CS_VT_UHF_OPT / LAM )
    
    plt.plot( f_list / LAM, E_RHF_pyscf + f_list*0, "-", c="black", label="RHF (PySCF)" )
    plt.plot( f_list / LAM, E_RHF + f_list*0, "o", ms=5, c="black", label="RHF" )
    plt.plot( f_list / LAM, E_UHF_pyscf + f_list*0, "--", c="red", label="UHF (PySCF)" )
    plt.plot( f_list / LAM, E_UHF + f_list*0, "o", ms=3, c="red", label="UHF" )
    plt.plot( f_list / LAM, E_QEDRHF + f_list*0 - 0.5 * WC, "-", c="blue", label="CS-QED-RHF - $\\frac{\\omega_\\mathrm{c}}{2}$" )
    plt.plot( f_list / LAM, E_QEDRHF_ZHY + f_list*0 - 0.5 * WC, "--", c="green", label="CS-QED-RHF (Y.Z.) - $\\frac{\\omega_\\mathrm{c}}{2}$" )
    plt.plot( f_list / LAM, E_QEDUHF + f_list*0 - 0.5 * WC, ".", c="orange", label="CS-QED-UHF - $\\frac{\\omega_\\mathrm{c}}{2}$" )
    
    plt.plot( f_list / LAM, E_VT_QEDRHF - 0.5 * WC, "--", c="cyan", label="CS-VT-QED-RHF - $\\frac{\\omega_\\mathrm{c}}{2}$" )
    plt.plot( f_list / LAM, E_VT_QEDUHF - 0.5 * WC, "o", c="cyan", label="CS-VT-QED-UHF - $\\frac{\\omega_\\mathrm{c}}{2}$" )
    plt.scatter( f_QED_CS_VT_RHF_OPT / LAM, E_QED_CS_VT_RHF_OPT - 0.5 * WC, c="black", marker="x", s=200, label="CS-VT-QED-RHF (OPT) - $\\frac{\\omega_\\mathrm{c}}{2}$" )
    plt.scatter( f_QED_CS_VT_UHF_OPT / LAM, E_QED_CS_VT_UHF_OPT - 0.5 * WC, c="red", marker="x", s=100, label="CS-VT-QED-UHF (OPT) - $\\frac{\\omega_\\mathrm{c}}{2}$" )
    plt.xlabel("Normalized Shift Parameter, $\\frac{f}{\\lambda}$", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$\\lambda$ = %1.2f a.u.  $\\omega$ = %1.2f a.u." % (LAM, WC), fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig("E_f.jpg", dpi=300)

if ( __name__ == "__main__" ):
    main()