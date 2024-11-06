import numpy as np
from matplotlib import pyplot as plt

from pyscf import gto

from QED_VT_RHF import do_QED_VT_RHF



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
    E      = np.zeros( (len(f_list)) )
    for fi,f in enumerate(f_list):
        print("f / LAM = ", f / LAM)
        E[fi] = do_QED_VT_RHF( mol, LAM, WC, f=f )
    print( E )
    
    plt.plot( f_list / LAM, E, "-o" )
    plt.xlabel("Normalized Shift Parameter, $\\frac{f}{\\lambda}$", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$\\lambda$ = %1.2f a.u." % (LAM), fontsize=15)
    plt.tight_layout()
    plt.savefig("E_f.jpg", dpi=300)

if ( __name__ == "__main__" ):
    main()