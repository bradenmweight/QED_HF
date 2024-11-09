import numpy as np
from matplotlib import pyplot as plt

from pyscf import gto, scf

from RHF import do_RHF
from UHF import do_UHF
from QED_RHF import do_QED_RHF, do_QED_HF_ZHY
from QED_VT_RHF import do_QED_VT_RHF
from QED_VT_UHF import do_QED_VT_UHF


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
    E_VT_QEDRHF = np.zeros( (len(f_list)) )
    E_VT_QEDUHF = np.zeros( (len(f_list)) )
    E_VT_S2     = np.zeros( (len(f_list)) )
    E_VT_ss1    = np.zeros( (len(f_list)) )
    E_RHF_pyscf = scf.RHF( mol ).kernel()
    E_RHF      = do_RHF( mol )
    E_UHF, S2, ss1 = do_UHF( mol )
    E_QEDHF    = do_QED_RHF( mol, LAM, WC )
    E_QEDHF_ZHY = do_QED_HF_ZHY( mol, LAM, WC )
    for fi,f in enumerate(f_list):
        print("f / LAM = ", f / LAM)
        E_VT_QEDRHF[fi] = do_QED_VT_RHF( mol, LAM, WC, f=f )
        E_VT_QEDUHF[fi], E_VT_S2[fi], E_VT_ss1[fi] = do_QED_VT_UHF( mol, LAM, WC, f=f )
        print( "E_VT_QED-RHF = ", E_VT_QEDRHF[fi] )
        print( "E_VT_QED-UHF = ", E_VT_QEDUHF[fi], E_VT_S2[fi], E_VT_ss1[fi] )
    E_QED_CS_VT_RHF_OPT, f_QED_CS_VT_RHF_OPT = do_QED_VT_RHF( mol, LAM, WC )
    E_QED_CS_VT_UHF_OPT, f_QED_CS_VT_UHF_OPT = do_QED_VT_RHF( mol, LAM, WC )
    E_QED_CS_VT_RHF_OPT = E_QED_CS_VT_RHF_OPT[-1]
    E_QED_CS_VT_UHF_OPT = E_QED_CS_VT_UHF_OPT[-1]
    f_QED_CS_VT_RHF_OPT = f_QED_CS_VT_RHF_OPT[-1]
    f_QED_CS_VT_UHF_OPT = f_QED_CS_VT_UHF_OPT[-1]
    print( E_RHF )
    print( E_UHF, S2, ss1 )
    print( E_RHF_pyscf )
    print( E_QEDHF )
    print( E_QEDHF_ZHY )
    print( E_VT_QEDRHF )
    print( E_VT_QEDUHF )
    print( E_QED_CS_VT_RHF_OPT, f_QED_CS_VT_RHF_OPT / LAM )
    print( E_QED_CS_VT_UHF_OPT, f_QED_CS_VT_UHF_OPT / LAM )
    
    plt.plot( f_list / LAM, E_RHF_pyscf + f_list*0 + 0.5 * WC, "-", c="black", label="RHF (PySCF) + $\\frac{\\omega_\\mathrm{c}}{2}$" )
    plt.plot( f_list / LAM, E_RHF + f_list*0 + 0.5 * WC, "o", c="red", label="RHF + $\\frac{\\omega_\\mathrm{c}}{2}$" )
    plt.plot( f_list / LAM, E_UHF + f_list*0 + 0.5 * WC, "--", c="red", label="UHF + $\\frac{\\omega_\\mathrm{c}}{2}$" )
    plt.plot( f_list / LAM, E_VT_QEDRHF, "o", c="cyan", label="CS-VT-QED-RHF" )
    plt.plot( f_list / LAM, E_VT_QEDUHF, "--", c="cyan", label="CS-VT-QED-UHF" )
    plt.plot( f_list / LAM, E_QEDHF_ZHY + f_list*0, "-", c="green", label="CS-QED-RHF (Y.Z.)" )
    plt.plot( f_list / LAM, E_QEDHF + f_list*0, "--", c="blue", label="CS-QED-RHF" )
    plt.scatter( f_QED_CS_VT_RHF_OPT / LAM, E_QED_CS_VT_RHF_OPT, c="cyan", marker="x", s=200, label="CS-VT-QED-RHF (Opt)" )
    plt.scatter( f_QED_CS_VT_UHF_OPT / LAM, E_QED_CS_VT_UHF_OPT, c="red", marker="x", s=100, label="CS-VT-QED-UHF (Opt)" )
    plt.xlabel("Normalized Shift Parameter, $\\frac{f}{\\lambda}$", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$\\lambda$ = %1.2f a.u." % (LAM), fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig("E_f.jpg", dpi=300)

if ( __name__ == "__main__" ):
    main()