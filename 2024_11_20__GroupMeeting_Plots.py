import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from pyscf import gto, scf, fci

from QED_RHF import do_QED_RHF, do_QED_HF_ZHY
from QED_VT_RHF import do_QED_VT_RHF

from QED_UHF import do_QED_UHF
from QED_VT_UHF import do_QED_VT_UHF
from QED_RCIS import do_QED_RCIS


def HH_LAM_SCAN_f_resolved():
    mol = gto.Mole()
    mol.basis = "321g"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'H 0 0 0; H 0 0 2.0'
    mol.build()

    WC            = 1.0
    LAM_LIST      = np.arange(0.0, 1+0.25, 0.25 )
    f_LIST        = np.linspace(0.0, 1.0, 31) # Normalized f
    E_LIST_VT     = np.zeros( (len(LAM_LIST),len(f_LIST)) )
    E_LIST_CS     = np.zeros( (len(LAM_LIST)) )
    E_LIST_SC     = np.zeros( (len(LAM_LIST)) )
    E_LIST_ZHY    = np.zeros( (len(LAM_LIST)) )
    f_OPT         = np.zeros( (len(LAM_LIST)) )
    E_LIST_VT_OPT = np.zeros( (len(LAM_LIST)) )

    for li,LAM in enumerate(LAM_LIST):
        E_LIST_CS[li] = do_QED_RHF( mol, LAM, WC )[0]
        E_LIST_ZHY[li] = do_QED_HF_ZHY( mol, LAM, WC )
        E_LIST_SC[li] = do_QED_VT_RHF( mol, LAM, WC, f=LAM )
        E_LIST_VT_OPT[li], f_OPT[li] = do_QED_VT_RHF( mol, LAM, WC )[:,-1]
        for fi,f in enumerate(f_LIST):
            print("LAM = ", LAM, "f = ", f, "f*LAM = ", f*LAM)    
            E_LIST_VT[li,fi] = do_QED_VT_RHF( mol, LAM, WC, f=f*LAM )

    for li,LAM in enumerate(LAM_LIST):
        #plt.scatter( 0.0, E_LIST_ZHY[li], "--", label="CS (ZHY)" * (li==0) )
        plt.plot( f_LIST, E_LIST_VT[li,:], "-", lw=6, ms=5, zorder=0, label="VT-QED-f" * (li==0) )
    for li,LAM in enumerate(LAM_LIST):
        plt.scatter( 0.0, E_LIST_CS[li], marker="*", zorder=1, c="black", s=100, label="CS-QED" * (li==0) )
        plt.scatter( 1.0, E_LIST_SC[li], marker="x", zorder=1, c="black", s=100, label="SC-QED" * (li==0) )
        plt.scatter( f_OPT[li]/LAM, E_LIST_VT_OPT[li], zorder=1, marker="^", c="black", s=100, label="VT-OPT" * (li==0) )
    plt.xlabel("Normalized Shift Parameter, $\\frac{f}{\\lambda}$", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.legend()
    #plt.ylim(-0.5,0.0)
    plt.tight_layout()
    plt.savefig("HH_LAM_SCAN_f_resolved.jpg", dpi=300)
    plt.clf()

def LiH_LAM_SCAN_f_resolved():
    mol = gto.Mole()
    mol.basis = "321g"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'Li 0 0 0; H 0 0 2.0'
    mol.build()

    WC            = 1.0
    LAM_LIST      = np.arange(0.0, 1+0.25, 0.25 )
    f_LIST        = np.linspace(0.0, 1.0, 31) # Normalized f
    E_LIST_VT     = np.zeros( (len(LAM_LIST),len(f_LIST)) )
    E_LIST_CS     = np.zeros( (len(LAM_LIST)) )
    E_LIST_SC     = np.zeros( (len(LAM_LIST)) )
    E_LIST_ZHY    = np.zeros( (len(LAM_LIST)) )
    f_OPT         = np.zeros( (len(LAM_LIST)) )
    E_LIST_VT_OPT = np.zeros( (len(LAM_LIST)) )

    for li,LAM in enumerate(LAM_LIST):
        E_LIST_CS[li] = do_QED_RHF( mol, LAM, WC )[0]
        E_LIST_ZHY[li] = do_QED_HF_ZHY( mol, LAM, WC )
        E_LIST_SC[li] = do_QED_VT_RHF( mol, LAM, WC, f=LAM )
        E_LIST_VT_OPT[li], f_OPT[li] = do_QED_VT_RHF( mol, LAM, WC )[:,-1]
        for fi,f in enumerate(f_LIST):
            print("LAM = ", LAM, "f = ", f, "f*LAM = ", f*LAM)    
            E_LIST_VT[li,fi] = do_QED_VT_RHF( mol, LAM, WC, f=f*LAM )

    for li,LAM in enumerate(LAM_LIST):
        #plt.scatter( 0.0, E_LIST_ZHY[li], "--", label="CS (ZHY)" * (li==0) )
        plt.plot( f_LIST, E_LIST_VT[li,:], "-", lw=6, ms=5, zorder=0, label="VT-QED-f" * (li==0) )
    for li,LAM in enumerate(LAM_LIST):
        plt.scatter( 0.0, E_LIST_CS[li], marker="*", zorder=1, c="black", s=100, label="CS-QED" * (li==0) )
        plt.scatter( 1.0, E_LIST_SC[li], marker="x", zorder=1, c="black", s=100, label="SC-QED" * (li==0) )
        plt.scatter( f_OPT[li]/LAM, E_LIST_VT_OPT[li], zorder=1, marker="^", c="black", s=100, label="VT-OPT" * (li==0) )
    plt.xlabel("Normalized Shift Parameter, $\\frac{f}{\\lambda}$", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.legend()
    #plt.ylim(-0.5,0.0)
    plt.tight_layout()
    plt.savefig("LiH_LAM_SCAN_f_resolved.jpg", dpi=300)
    plt.clf()

def LiH_LAM_SCAN_mins():
    mol = gto.Mole()
    mol.basis = "321g"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'Li 0 0 0; H 0 0 2.0'
    mol.build()

    WC            = 1.0
    LAM_LIST      = np.arange(0.0, 2.25+0.05, 0.05 )
    E_LIST_CS     = np.zeros( (len(LAM_LIST)) )
    E_LIST_SC     = np.zeros( (len(LAM_LIST)) )
    E_LIST_VT_OPT = np.zeros( (len(LAM_LIST)) )
    f_LIST_VT_OPT = np.zeros( (len(LAM_LIST)) )

    for li,LAM in enumerate(LAM_LIST):
        E_LIST_CS[li] = do_QED_RHF( mol, LAM, WC )[0]
        E_LIST_SC[li] = do_QED_VT_RHF( mol, LAM, WC, f=LAM )
        E_LIST_VT_OPT[li], f_LIST_VT_OPT[li] = do_QED_VT_RHF( mol, LAM, WC )[:,-1]

    plt.semilogy( LAM_LIST, E_LIST_CS - E_LIST_VT_OPT, "-", lw=4, label="CS-QED" )
    plt.semilogy( LAM_LIST, E_LIST_SC - E_LIST_VT_OPT, "-", lw=4, label="SC-QED" )
    #plt.plot( LAM_LIST, E_LIST_VT_OPT, "-", lw=6, ms=5, zorder=0, label="VT-QED" )
    plt.xlabel("Coupling Strength, $\\lambda$ (a.u.)", fontsize=15)
    plt.ylabel("$E - E^{\\mathrm{VT}}$ (a.u.)", fontsize=15)
    plt.legend()
    plt.ylim(1e-3)
    plt.tight_layout()
    plt.savefig("LiH_LAM_SCAN_E_mins.jpg", dpi=300)
    plt.clf()

    plt.plot( LAM_LIST[1:], f_LIST_VT_OPT[1:]/LAM_LIST[1:], "-", lw=4 )
    #plt.plot( LAM_LIST, E_LIST_VT_OPT, "-", lw=6, ms=5, zorder=0, label="VT-QED" )
    plt.xlabel("Coupling Strength, $\\lambda$ (a.u.)", fontsize=15)
    plt.ylabel("Optimal Normalized Shift Parameter, $\\frac{f_\\mathrm{opt}}{\\lambda}$", fontsize=15)
    #plt.legend()
    plt.ylim(None,1.0)
    plt.tight_layout()
    plt.savefig("LiH_LAM_SCAN_f_mins.jpg", dpi=300)
    plt.clf()

def H2_LAM_SCAN_mins():
    mol = gto.Mole()
    mol.basis = "ccpvdz"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'H 0 0 0; H 0 0 2.0'
    mol.build()

    WC            = 1.0
    LAM_LIST      = np.arange(0.0, 2.25+0.05, 0.05 )
    E_LIST_CS     = np.zeros( (len(LAM_LIST)) )
    E_LIST_SC     = np.zeros( (len(LAM_LIST)) )
    E_LIST_VT_OPT = np.zeros( (len(LAM_LIST)) )
    f_LIST_VT_OPT = np.zeros( (len(LAM_LIST)) )

    for li,LAM in enumerate(LAM_LIST):
        E_LIST_CS[li] = do_QED_RHF( mol, LAM, WC )[0]
        E_LIST_SC[li] = do_QED_VT_RHF( mol, LAM, WC, f=LAM )
        E_LIST_VT_OPT[li], f_LIST_VT_OPT[li] = do_QED_VT_RHF( mol, LAM, WC )[:,-1]

    plt.semilogy( LAM_LIST, E_LIST_CS - E_LIST_VT_OPT, "-", lw=4, label="CS-QED" )
    plt.semilogy( LAM_LIST, E_LIST_SC - E_LIST_VT_OPT, "-", lw=4, label="SC-QED" )
    #plt.plot( LAM_LIST, E_LIST_VT_OPT, "-", lw=6, ms=5, zorder=0, label="VT-QED" )
    plt.xlabel("Coupling Strength, $\\lambda$ (a.u.)", fontsize=15)
    plt.ylabel("$E - E^{\\mathrm{VT}}$ (a.u.)", fontsize=15)
    plt.legend()
    plt.ylim(1e-3)
    plt.tight_layout()
    plt.savefig("H2_LAM_SCAN_E_mins.jpg", dpi=300)
    plt.clf()

    plt.plot( LAM_LIST[1:], f_LIST_VT_OPT[1:]/LAM_LIST[1:], "-", lw=4 )
    #plt.plot( LAM_LIST, E_LIST_VT_OPT, "-", lw=6, ms=5, zorder=0, label="VT-QED" )
    plt.xlabel("Coupling Strength, $\\lambda$ (a.u.)", fontsize=15)
    plt.ylabel("Optimal Normalized Shift Parameter, $\\frac{f_\\mathrm{opt}}{\\lambda}$", fontsize=15)
    #plt.legend()
    plt.ylim(None,1.0)
    plt.tight_layout()
    plt.savefig("H2_LAM_SCAN_f_mins.jpg", dpi=300)
    plt.clf()

def LiH_f_optimization():
    mol = gto.Mole()
    mol.basis = "321g"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'Li 0 0 0; H 0 0 2.0'
    mol.build()

    WC       = 1.0
    LAM_LIST = np.arange(1e-2, 2.0+0.1, 0.1 )
    E_GD     = []
    f_GD     = []
    E_NR     = []
    f_NR     = []

    for li,LAM in enumerate(LAM_LIST):
        tmp = do_QED_VT_RHF( mol, LAM, WC, opt_method="GD" )
        E_GD.append( tmp[0] )
        f_GD.append( tmp[1] )
        tmp = do_QED_VT_RHF( mol, LAM, WC, opt_method="NR" )
        E_NR.append( tmp[0] )
        f_NR.append( tmp[1] )

    
    norm        = matplotlib.colors.Normalize(vmin=-0.5, vmax=LAM_LIST[-1])
    color_map_1 = plt.get_cmap("Reds")
    color_map_2 = plt.get_cmap("Blues")

    for li,LAM in enumerate(LAM_LIST):
        f = np.array(f_GD[li])/LAM
        plt.semilogx( np.arange(1,len(f)+1), np.abs(f - f[-1]), "-", c=color_map_1(LAM), lw=1, label="Gradient Descent"*(li==len(LAM_LIST)-1) )
        f = np.array(f_NR[li])/LAM
        plt.semilogx( np.arange(1,len(f)+1), np.abs(f - f[-1]), "-", c=color_map_2(LAM), lw=1, label="Newton-Raphson"*(li==len(LAM_LIST)-1) )
    plt.xlabel("Number of Iterations", fontsize=15)
    plt.ylabel("Error in Variational Parameter, $|\\frac{f - f_\\mathrm{opt}}{\\lambda}|$", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig("LiH_LAM_SCAN_f_opt_methods.jpg", dpi=300)
    plt.clf()

def H2_spin_contamination():

    def get_FCI(mol):
        mf = scf.RHF(mol).run()
        myci = fci.FCI(mf)
        myci.kernel()
        return myci.e_tot

    mol = gto.Mole()
    mol.basis = "ccpvdz"
    mol.unit = 'Bohr'
    mol.symmetry = False

    WC = 1.0

    R_LIST    = np.arange(0.9, 6.0, 0.1)
    LAM_LIST  = np.arange(0.0, 0.3+0.1, 0.1 )

    E_FCI = np.zeros( (len(R_LIST)) )

    E_RCS = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    E_RVT = np.zeros( (len(LAM_LIST),len(R_LIST)) )

    E_UCS = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    E_UVT = np.zeros( (len(LAM_LIST),len(R_LIST)) )

    S2_UCS = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    S2_UVT = np.zeros( (len(LAM_LIST),len(R_LIST)) )

    for Li,LAM in enumerate(LAM_LIST):
        for Ri,R in enumerate(R_LIST):
            mol.atom = 'H 0 0 0; H 0 0 %1.8f' % (R)
            mol.build()
            E_FCI[Ri]                         = get_FCI(mol)
            E_RCS[Li,Ri]                      = do_QED_RHF( mol, LAM, WC )[0]
            E_UCS[Li,Ri], S2_UCS[Li,Ri], _    = do_QED_UHF( mol, LAM, WC )[:]
            E_RVT[Li,Ri]                      = do_QED_VT_RHF( mol, LAM, WC )[0,-1]
            E_UVT[Li,Ri], _, S2_UVT[Li,Ri], _ = do_QED_VT_UHF( mol, LAM, WC )[:,-1]

    norm        = matplotlib.colors.Normalize(vmin=0, vmax=np.max(S2_UVT))
    color_map1 = plt.get_cmap("brg")
    color_map2 = plt.get_cmap("brg")

    fig, ax = plt.subplots()
    plt.plot( R_LIST, E_FCI[:], "-", c='black', zorder=0, alpha=0.5, lw=8, label="FCI" )
    plt.plot( R_LIST, E_RCS[0,:] - WC/2, "-", zorder=1, c='black', label="RHF" )
    plt.scatter( R_LIST, E_UCS[0,:] - WC/2, zorder=2, s=50, c=color_map1(S2_UCS[Li,:]), label="UHF" )
    smap = plt.cm.ScalarMappable(norm=norm, cmap=color_map1)
    plt.colorbar(smap, ax=ax, label="Spin Contamination, $\\langle \\hat{S}^2 \\rangle$", pad=0.01)
    plt.xlabel("Interatomic Separation, $R$ (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig("H2_spin_contamination.jpg", dpi=300)
    plt.clf()


    fig, ax = plt.subplots()
    plt.plot( R_LIST, E_FCI[:], "-", c='black', zorder=0, alpha=0.5, lw=8, label="FCI" )
    for Li,LAM in enumerate(LAM_LIST):
        plt.plot( R_LIST, E_RCS[Li,:] - WC/2, "-", zorder=1, c='black', label="QED-CS - $\\frac{\\omega_\\mathrm{c}}{2}$"*(Li==0) )
        plt.scatter( R_LIST, E_UCS[Li,:] - WC/2, marker="s", zorder=2, s=50, facecolors='none', edgecolors=color_map1(S2_UCS[Li,:]), label="QED-CS - $\\frac{\\omega_\\mathrm{c}}{2}$"*(Li==0) )
        plt.scatter( R_LIST, E_UVT[Li,:] - WC/2, marker="o", zorder=3, s=25, facecolors='none', edgecolors=color_map2(S2_UVT[Li,:]), label="QED-VT - $\\frac{\\omega_\\mathrm{c}}{2}$"*(Li==0) )
    smap = plt.cm.ScalarMappable(norm=norm, cmap=color_map1)
    plt.colorbar(smap, ax=ax, label="Spin Contamination, $\\langle \\hat{S}^2 \\rangle$", pad=0.01)
    plt.xlabel("Interatomic Separation, $R$ (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.legend()
    plt.ylim(-1.2,-0.8)
    plt.tight_layout()
    plt.savefig("H2_QED_spin_contamination.jpg", dpi=300)
    plt.clf()


    fig, ax = plt.subplots()
    for Li,LAM in enumerate(LAM_LIST):
        plt.plot( R_LIST, S2_UCS[Li,:], "-", label="$\\lambda_\\mathrm{c}$ = %1.1f" % (LAM) )
    plt.xlabel("Interatomic Separation, $R$ (a.u.)", fontsize=15)
    plt.ylabel("Spin Contamination, $\\langle \\hat{S}^2 \\rangle$", fontsize=15)
    plt.title("QED-CS", fontsize=15)
    plt.legend()
    #plt.ylim(-1.2,-0.8)
    plt.tight_layout()
    plt.savefig("H2_QED_S2_CS.jpg", dpi=300)
    plt.clf()

    fig, ax = plt.subplots()
    for Li,LAM in enumerate(LAM_LIST):
        plt.plot( R_LIST, S2_UVT[Li,:], "-", label="$\\lambda_\\mathrm{c}$ = %1.1f" % (LAM) )
    plt.xlabel("Interatomic Separation, $R$ (a.u.)", fontsize=15)
    plt.ylabel("Spin Contamination, $\\langle \\hat{S}^2 \\rangle$", fontsize=15)
    plt.title("QED-VT", fontsize=15)
    plt.legend()
    #plt.ylim(-1.2,-0.8)
    plt.tight_layout()
    plt.savefig("H2_QED_S2_VT.jpg", dpi=300)
    plt.clf()


    # # Calculate the Coulson-Fischer Point for each coupling strength
    # # Define by value of S2
    CFP_CS = np.zeros( (len(LAM_LIST)) )
    for Li,LAM in enumerate(LAM_LIST):
        for Ri,R in enumerate(R_LIST):
            if ( S2_UCS[Li,Ri] > 1e-3 ):
                CFP_CS[Li] = R
                break # Exits inner loop
    CFP_VT = np.zeros( (len(LAM_LIST)) )
    for Li,LAM in enumerate(LAM_LIST):
        for Ri,R in enumerate(R_LIST):
            if ( S2_UVT[Li,Ri] > 1e-3 ):
                CFP_VT[Li] = R
                break # Exits inner loop

    fig, ax = plt.subplots()
    plt.plot( LAM_LIST, CFP_CS, "-o", label="QED-CS" )
    plt.plot( LAM_LIST, CFP_VT, "--o", label="QED-VT" )
    plt.xlabel("Coupling Strength, $\\lambda_\\mathrm{c}$ (a.u.)", fontsize=15)
    plt.ylabel("Coulson-Fischer Point, $R_\\mathrm{CFP}$ (a.u.)", fontsize=15)
    plt.legend()
    #plt.ylim(-1.2,-0.8)
    plt.tight_layout()
    plt.savefig("H2_QED_CFP.jpg", dpi=300)
    plt.clf()

def LiH_spin_contamination():

    def get_FCI(mol):
        mf = scf.RHF(mol).run()
        myci = fci.FCI(mf)
        myci.kernel()
        return myci.e_tot

    mol = gto.Mole()
    mol.basis = "321g"
    mol.unit = 'Bohr'
    mol.symmetry = False

    WC = 1.0

    R_LIST    = np.arange(4.0, 4.3+0.1, 0.01)
    LAM_LIST  = np.arange(0.0, 0.05+0.01, 0.01 )

    E_FCI = np.zeros( (len(R_LIST)) )

    E_RCS = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    E_RVT = np.zeros( (len(LAM_LIST),len(R_LIST)) )

    E_UCS = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    E_UVT = np.zeros( (len(LAM_LIST),len(R_LIST)) )

    S2_UCS = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    S2_UVT = np.zeros( (len(LAM_LIST),len(R_LIST)) )

    for Li,LAM in enumerate(LAM_LIST):
        for Ri,R in enumerate(R_LIST):
            mol.atom = 'Li 0 0 0; H 0 0 %1.8f' % (R)
            mol.build()
            E_FCI[Ri]                         = get_FCI(mol)
            E_RCS[Li,Ri]                      = do_QED_RHF( mol, LAM, WC )[0]
            E_UCS[Li,Ri], S2_UCS[Li,Ri], _    = do_QED_UHF( mol, LAM, WC )[:]
            E_RVT[Li,Ri]                      = do_QED_VT_RHF( mol, LAM, WC )[0,-1]
            E_UVT[Li,Ri], _, S2_UVT[Li,Ri], _ = do_QED_VT_UHF( mol, LAM, WC )[:,-1]

    norm        = matplotlib.colors.Normalize(vmin=0, vmax=np.max(S2_UVT))
    color_map1 = plt.get_cmap("brg")
    color_map2 = plt.get_cmap("brg")

    fig, ax = plt.subplots()
    plt.plot( R_LIST, E_FCI[:], "-", c='black', zorder=0, alpha=0.5, lw=8, label="FCI" )
    plt.plot( R_LIST, E_RCS[0,:] - WC/2, "-", zorder=1, c='black', label="RHF" )
    plt.scatter( R_LIST, E_UCS[0,:] - WC/2, zorder=2, s=50, c=color_map1(S2_UCS[Li,:]), label="UHF" )
    smap = plt.cm.ScalarMappable(norm=norm, cmap=color_map1)
    plt.colorbar(smap, ax=ax, label="Spin Contamination, $\\langle \\hat{S}^2 \\rangle$", pad=0.01)
    plt.xlabel("Interatomic Separation, $R$ (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig("LiH_spin_contamination.jpg", dpi=300)
    plt.clf()


    fig, ax = plt.subplots()
    plt.plot( R_LIST, E_FCI[:], "-", c='black', zorder=0, alpha=0.5, lw=8, label="FCI" )
    for Li,LAM in enumerate(LAM_LIST):
        plt.plot( R_LIST, E_RCS[Li,:] - WC/2, "-", zorder=1, c='black', label="QED-CS - $\\frac{\\omega_\\mathrm{c}}{2}$"*(Li==0) )
        plt.scatter( R_LIST, E_UCS[Li,:] - WC/2, marker="s", zorder=2, s=50, facecolors='none', edgecolors=color_map1(S2_UCS[Li,:]), label="QED-CS - $\\frac{\\omega_\\mathrm{c}}{2}$"*(Li==0) )
        plt.scatter( R_LIST, E_UVT[Li,:] - WC/2, marker="o", zorder=3, s=25, facecolors='none', edgecolors=color_map2(S2_UVT[Li,:]), label="QED-VT - $\\frac{\\omega_\\mathrm{c}}{2}$"*(Li==0) )
    smap = plt.cm.ScalarMappable(norm=norm, cmap=color_map1)
    plt.colorbar(smap, ax=ax, label="Spin Contamination, $\\langle \\hat{S}^2 \\rangle$", pad=0.01)
    plt.xlabel("Interatomic Separation, $R$ (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.legend()
    plt.ylim(-7.96,-7.8)
    plt.tight_layout()
    plt.savefig("LiH_QED_spin_contamination.jpg", dpi=300)
    plt.clf()


    fig, ax = plt.subplots()
    for Li,LAM in enumerate(LAM_LIST):
        plt.plot( R_LIST, S2_UCS[Li,:], "-", label="$\\lambda_\\mathrm{c}$ = %1.1f" % (LAM) )
    plt.xlabel("Interatomic Separation, $R$ (a.u.)", fontsize=15)
    plt.ylabel("Spin Contamination, $\\langle \\hat{S}^2 \\rangle$", fontsize=15)
    plt.title("QED-CS", fontsize=15)
    plt.legend()
    #plt.ylim(-1.2,-0.8)
    plt.tight_layout()
    plt.savefig("LiH_QED_S2_CS.jpg", dpi=300)
    plt.clf()

    fig, ax = plt.subplots()
    for Li,LAM in enumerate(LAM_LIST):
        plt.plot( R_LIST, S2_UVT[Li,:], "-", label="$\\lambda_\\mathrm{c}$ = %1.1f" % (LAM) )
    plt.xlabel("Interatomic Separation, $R$ (a.u.)", fontsize=15)
    plt.ylabel("Spin Contamination, $\\langle \\hat{S}^2 \\rangle$", fontsize=15)
    plt.title("QED-VT", fontsize=15)
    plt.legend()
    #plt.ylim(-1.2,-0.8)
    plt.tight_layout()
    plt.savefig("LiH_QED_S2_VT.jpg", dpi=300)
    plt.clf()


    # # Calculate the Coulson-Fischer Point for each coupling strength
    # # Define by value of S2
    CFP_CS = np.zeros( (len(LAM_LIST)) )
    for Li,LAM in enumerate(LAM_LIST):
        for Ri,R in enumerate(R_LIST):
            if ( S2_UCS[Li,Ri] > 1e-3 ):
                CFP_CS[Li] = R
                break # Exits inner loop
    CFP_VT = np.zeros( (len(LAM_LIST)) )
    for Li,LAM in enumerate(LAM_LIST):
        for Ri,R in enumerate(R_LIST):
            if ( S2_UVT[Li,Ri] > 1e-3 ):
                CFP_VT[Li] = R
                break # Exits inner loop

    fig, ax = plt.subplots()
    plt.plot( LAM_LIST, CFP_CS, "-o", label="QED-CS" )
    plt.plot( LAM_LIST, CFP_VT, "--o", label="QED-VT" )
    plt.xlabel("Coupling Strength, $\\lambda_\\mathrm{c}$ (a.u.)", fontsize=15)
    plt.ylabel("Coulson-Fischer Point, $R_\\mathrm{CFP}$ (a.u.)", fontsize=15)
    plt.legend()
    #plt.ylim(-1.2,-0.8)
    plt.tight_layout()
    plt.savefig("LiH_QED_CFP.jpg", dpi=300)
    plt.clf()

def H2_LAM_SCAN_ALL_METHODS():
    mol = gto.Mole()
    mol.basis = "ccpvdz"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'H 0 0 0; H 0 0 2.0'
    mol.build()

    WC = 1.0

    LAM_LIST  = np.arange(0.0, 0.5+0.01, 0.01 )

    E_CS     = np.zeros( (len(LAM_LIST)) )
    E_SC     = np.zeros( (len(LAM_LIST)) )
    E_CS_ZHY = np.zeros( (len(LAM_LIST)) )
    E_VT     = np.zeros( (len(LAM_LIST)) )

    for Li,LAM in enumerate(LAM_LIST):
        E_CS[Li]      = do_QED_RHF( mol, LAM, WC )[0]
        E_CS_ZHY[Li]  = do_QED_HF_ZHY( mol, LAM, WC )
        E_VT[Li]      = do_QED_VT_RHF( mol, LAM, WC )[0,-1]
        E_SC[Li]      = do_QED_VT_RHF( mol, LAM, WC, f=LAM )

    plt.plot( LAM_LIST, E_CS, "-", lw=4, label="CS-QED" )
    plt.plot( LAM_LIST, E_CS_ZHY, "--", lw=4, label="CS-QED (Yu Zhang)" )
    plt.plot( LAM_LIST, E_SC, "-", lw=4, label="SC-QED" )
    plt.plot( LAM_LIST, E_VT, "-", lw=4, label="VT-QED" )
    plt.xlabel("Coupling Strength, $\\lambda$ (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15) #  - E^{\\mathrm{VT}}(\\lambda = 0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("H2_LAM_SCAN_ALL_METHODS.jpg", dpi=300)
    plt.clf()

def H2_Bond_Scan_ALL_METHODS():
    mol = gto.Mole()
    mol.basis = "ccpvdz"
    mol.unit = 'Bohr'
    mol.symmetry = False

    LAM_LIST  = np.array([0.0, 0.3]) # np.arange(0.0, 0.5+0.25, 0.25 )
    R_LIST = np.arange(0.75, 3.55, 0.05)

    WC = 1.0

    E_CS     = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    E_SC     = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    E_CS_ZHY = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    E_VT     = np.zeros( (len(LAM_LIST),len(R_LIST)) )

    for Li,LAM in enumerate(LAM_LIST):
        for Ri,R in enumerate(R_LIST):
            mol.atom = 'H 0 0 0; H 0 0 %1.8f' % (R)
            mol.build()
            E_CS[Li,Ri]      = do_QED_RHF( mol, LAM, WC )[0]
            E_CS_ZHY[Li,Ri]  = do_QED_HF_ZHY( mol, LAM, WC )
            E_VT[Li,Ri]      = do_QED_VT_RHF( mol, LAM, WC )[0,-1]
            E_SC[Li,Ri]      = do_QED_VT_RHF( mol, LAM, WC, f=LAM )

    for Li,LAM in enumerate(LAM_LIST):
        plt.plot( R_LIST, E_CS[Li,:],     "-",  c='black',  lw=2, zorder=2, label="CS-QED"*(Li==0) )
        plt.plot( R_LIST, E_CS_ZHY[Li,:], "--", c='red',  lw=2, zorder=3, label="CS-QED (Yu Zhang)"*(Li==0) )
        plt.plot( R_LIST, E_SC[Li,:],     ".",  c='cyan',  lw=2, zorder=1, label="SC-QED"*(Li==0) )
        plt.plot( R_LIST, E_VT[Li,:],     "o",  c='green', zorder=0,  lw=2, label="VT-QED"*(Li==0) )
    plt.xlabel("$R_\\mathrm{HH}$ (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15) #  - E^{\\mathrm{VT}}(\\lambda = 0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("H2_BOND_SCAN_ALL_METHODS.jpg", dpi=300)
    plt.clf()

def H2_WC_SCAN_ALL_METHODS():
    mol = gto.Mole()
    mol.basis = "ccpvdz"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'H 0 0 0; H 0 0 2.0'
    mol.build()

    LAM = 0.1

    WC_LIST  = np.arange(0.1, 5.0+0.05, 0.05 )

    E_CS     = np.zeros( (len(WC_LIST)) )
    E_SC     = np.zeros( (len(WC_LIST)) )
    E_CS_ZHY = np.zeros( (len(WC_LIST)) )
    E_VT     = np.zeros( (len(WC_LIST)) )

    E0 = do_QED_RHF( mol, 0.0, 0.0 )[0]

    for WCi,WC in enumerate(WC_LIST):
        E_CS[WCi]      = do_QED_RHF( mol, LAM, WC )[0]
        E_CS_ZHY[WCi]  = do_QED_HF_ZHY( mol, LAM, WC )
        E_VT[WCi]      = do_QED_VT_RHF( mol, LAM, WC )[0,-1]
        E_SC[WCi]      = do_QED_VT_RHF( mol, LAM, WC, f=LAM )

    plt.plot( WC_LIST, WC_LIST*0 + E0, "--", lw=2, c='black', label="E($\\omega_\\mathrm{c}$ = 0, $\\lambda_\\mathrm{c}$ = 0)" )
    plt.plot( WC_LIST, E_CS - WC_LIST/2, "-", lw=4, label="CS-QED" )
    plt.plot( WC_LIST, E_CS_ZHY - WC_LIST/2, "--", lw=4, label="CS-QED (Yu Zhang)" )
    plt.plot( WC_LIST, E_SC - WC_LIST/2, "-", lw=4, label="SC-QED" )
    plt.plot( WC_LIST, E_VT - WC_LIST/2, "-", lw=4, label="VT-QED" )
    plt.xlabel("Cavity Frequency, $\\omega_\\mathrm{c}$ (a.u.)", fontsize=15)
    plt.ylabel("Energy - $\\frac{\\omega_\\mathrm{c}}{2}$ (a.u.)", fontsize=15) #  - E^{\\mathrm{VT}}(\\lambda = 0)
    plt.title("$\\lambda_\\mathrm{c}$ = %1.2f" % LAM, fontsize=15) #  - E^{\\mathrm{VT}}(\\lambda = 0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("H2_WC_SCAN_ALL_METHODS.jpg", dpi=300)
    plt.clf()

def H2_QED_CIS_WC_SCAN():
    
    mol = gto.Mole()
    mol.basis = "sto3g"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'H 0 0 0; H 0 0 2.0'
    mol.build()

    nstates  = 2
    LAM      = 0.0
    WC_LIST  = np.arange(0.4, 1.0+0.001, 0.001 )

    E_QED_RCIS  = np.zeros( (len(WC_LIST), nstates+1) )
    PHOT_NUM    = np.zeros( (len(WC_LIST), nstates+1) )

    for WCi,WC in enumerate(WC_LIST):
        E_QED_RCIS[WCi,0]  = do_QED_RHF( mol, LAM, WC )[0]
        E_QED_RCIS[WCi,1:], PHOT_NUM[WCi,1:] = E_QED_RCIS[WCi,0] + do_QED_RCIS( mol, LAM, WC, nstates=nstates, calc_photon_number=True )#[0]

    for state in range(nstates+1):
        #plt.plot( WC_LIST, WC_LIST*0 + E_RCIS[state], "-", lw=3, label="State %d" % state )
        plt.plot( WC_LIST, E_QED_RCIS[:,state] - WC_LIST/2, "-", lw=3, label="QED-RCIS %d" % state )
    plt.xlabel("Cavity Frequency, $\\omega_\\mathrm{c}$ (a.u.)", fontsize=15)
    plt.ylabel("Energy - $\\frac{\\omega_\\mathrm{c}}{2}$ (a.u.)", fontsize=15) #  - E^{\\mathrm{VT}}(\\lambda = 0)
    plt.title("$\\lambda_\\mathrm{c}$ = %1.2f" % LAM, fontsize=15) #  - E^{\\mathrm{VT}}(\\lambda = 0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("H2_QED_CIS_WC_SCAN.jpg", dpi=300)
    plt.clf()

    # # Plot the PES colored by the photon number
    norm        = matplotlib.colors.Normalize(vmin=np.min(PHOT_NUM), vmax=np.max(PHOT_NUM))
    color_map   = plt.get_cmap("viridis")
    colors      = color_map(norm(PHOT_NUM))
    fig, ax = plt.subplots()
    for state in range(nstates+1):
        plt.scatter( WC_LIST, E_QED_RCIS[:,state] - WC_LIST/2, c=colors[:,state] )
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color_map), ax=ax, label="Photon Number", pad=0.01)
    plt.xlabel("Cavity Frequency, $\\omega_\\mathrm{c}$ (a.u.)", fontsize=15)
    plt.ylabel("Energy - $\\frac{\\omega_\\mathrm{c}}{2}$ (a.u.)", fontsize=15) #  - E^{\\mathrm{VT}}(\\lambda = 0)
    plt.title("$\\lambda_\\mathrm{c}$ = %1.2f" % LAM, fontsize=15) #  - E^{\\mathrm{VT}}(\\lambda = 0)
    plt.tight_layout()
    plt.savefig("H2_QED_CIS_WC_SCAN_PHOT.jpg", dpi=300)
    plt.clf()

def H2_QED_CIS_LAM_SCAN():
    
    mol = gto.Mole()
    mol.basis = "sto3g"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'H 0 0 0; H 0 0 2.0'
    mol.build()

    nstates  = 2

    E_HF = do_QED_RHF( mol, 0.0, 0.0 )[0]
    E_S1 = do_QED_RCIS( mol, 0.0, 0.0, nstates=2 )[1]
    WC   = E_S1

    LAM_LIST    = np.arange(0.0, 0.95+0.002, 0.002 )
    E_QED_RCIS  = np.zeros( (len(LAM_LIST), nstates+1) )
    PHOT_NUM    = np.zeros( (len(LAM_LIST), nstates+1) )

    for LAMi,LAM in enumerate(LAM_LIST):
        E_QED_RCIS[LAMi,0]  = do_QED_RHF( mol, LAM, WC )[0]
        TMP = do_QED_RCIS( mol, LAM, WC, nstates=nstates, calc_photon_number=True )#[0]
        E_QED_RCIS[LAMi,1:] = E_QED_RCIS[LAMi,0] + TMP[0][:] 
        PHOT_NUM[LAMi,1:]   = TMP[1][:]


    for state in range(nstates+1):
        plt.plot( LAM_LIST, E_QED_RCIS[:,state], "-", lw=3, label="QED-RCIS %d" % state )
    plt.xlabel("Coupling Strength, $\\lambda_\\mathrm{c}$ (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15) #  - E^{\\mathrm{VT}}(\\lambda = 0)
    plt.title("$\\omega_\\mathrm{c}$ = %1.2f" % WC, fontsize=15) #  - E^{\\mathrm{VT}}(\\lambda = 0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("H2_QED_CIS_LAM_SCAN.jpg", dpi=300)
    plt.clf()

    # # Plot the PES colored by the photon number
    norm        = matplotlib.colors.Normalize(vmin=np.min(PHOT_NUM), vmax=np.max(PHOT_NUM))
    color_map   = plt.get_cmap("viridis")
    colors      = color_map(norm(PHOT_NUM))
    fig, ax = plt.subplots()
    for state in range(nstates+1):
        plt.scatter( LAM_LIST, E_QED_RCIS[:,state], c=colors[:,state] )
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color_map), ax=ax, label="Photon Number", pad=0.01)
    plt.xlabel("Coupling Strength, $\\lambda_\\mathrm{c}$ (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$\\omega_\\mathrm{c}$ = %1.2f" % WC, fontsize=15)
    plt.tight_layout()
    plt.savefig("H2_QED_CIS_LAM_SCAN_PHOT.jpg", dpi=300)
    plt.clf()

def H2_DQMC_CCSD_comparison():
    mol = gto.Mole()
    mol.basis = "ccpvdz"
    mol.unit = 'Bohr'
    mol.symmetry = False
    mol.atom = 'H 0 0 -1.4; H 0 0 1.4' # This has to be |2.8| to match the DQMC results
    mol.build()

    WC = 20.0 / 27.2114 # a.u.

    A0_LIST   = np.arange(0.0, 1.0+0.05, 0.05 )
    LAM_LIST  = np.sqrt(2 * WC) * A0_LIST

    E_CS     = np.zeros( (len(LAM_LIST)) )
    E_SC     = np.zeros( (len(LAM_LIST)) )
    E_CS_ZHY = np.zeros( (len(LAM_LIST)) )
    E_VT     = np.zeros( (len(LAM_LIST)) )

    for LAMi,LAM in enumerate(LAM_LIST):
        E_CS[LAMi]      = do_QED_RHF( mol, LAM, WC )[0]
        E_CS_ZHY[LAMi]  = do_QED_HF_ZHY( mol, LAM, WC )
        E_VT[LAMi]      = do_QED_VT_RHF( mol, LAM, WC )[0,-1]
        E_SC[LAMi]      = do_QED_VT_RHF( mol, LAM, WC, f=LAM )

    plt.plot( LAM_LIST, E_CS - E_CS[0], "-", lw=4, label="CS-QED" )
    plt.plot( LAM_LIST, E_CS_ZHY - E_CS_ZHY[0], "--", lw=4, label="CS-QED (Yu Zhang)" )
    plt.plot( LAM_LIST, E_SC - E_SC[0], "-", lw=4, label="SC-QED" )
    plt.plot( LAM_LIST, E_VT - E_VT[0], "-", lw=4, label="VT-QED" )
    plt.xlabel("Couopling Strength, $A_0$ (a.u.)", fontsize=15)
    plt.ylabel("Energy - $\\frac{\\omega_\\mathrm{c}}{2}$ (a.u.)", fontsize=15)
    plt.title("$\\omega_\\mathrm{c}$ = %1.2f" % WC, fontsize=15)
    plt.legend()
    plt.ylim(0,0.5)
    plt.tight_layout()
    plt.savefig("H2_DQMC_CCSD_comparison.jpg", dpi=300)
    plt.clf()
    np.savetxt( "H2_DQMC_CCSD_comparison.dat", np.c_[A0_LIST,LAM_LIST,E_CS - E_CS[0], E_CS_ZHY - E_CS_ZHY[0], E_SC - E_SC[0], E_VT - E_VT[0] ], header="A0 LAM CS CS_ZHY SC VT" )

def H2_QED_RCIS_BOND_SCAN():

    mol = gto.Mole()
    mol.basis = "sto3g"
    mol.unit = 'Bohr'
    mol.symmetry = False

    nstates  = 2
    LAM      = 0.1
    WC       = 0.6 # 0.6 for sto-3g and 0.4 for 321g
    R_LIST   = np.arange(0.75, 6.0+0.01, 0.01 )

    E_QED_RCIS_S = np.zeros( (len(R_LIST), nstates+1) )
    E_QED_RCIS_T = np.zeros( (len(R_LIST), nstates) )
    PHOT_NUM_S   = np.zeros( (len(R_LIST), nstates+1) )
    PHOT_NUM_T   = np.zeros( (len(R_LIST), nstates) )

    for Ri,R in enumerate(R_LIST):
        mol.atom = 'H 0 0 0; H 0 0 %1.8f' % R
        mol.build()
        E_QED_RCIS_S[Ri,0]  = do_QED_RHF( mol, LAM, WC )[0]
        TMP                   = do_QED_RCIS( mol, LAM, WC, symmetry='s', nstates=nstates, calc_photon_number=True )#[0]
        E_QED_RCIS_S[Ri,1:] = E_QED_RCIS_S[Ri,0] + TMP[0][:] 
        PHOT_NUM_S[Ri,1:]   = TMP[1][:]

        TMP                   = do_QED_RCIS( mol, LAM, WC, symmetry='t', nstates=nstates, calc_photon_number=True )#[0]
        E_QED_RCIS_T[Ri,:]  = E_QED_RCIS_S[Ri,0] + TMP[0][:]
        PHOT_NUM_T[Ri,:]    = TMP[1][:]

    # # Plot the PES colored by the photon number
    norm        = matplotlib.colors.Normalize(vmin=np.min(PHOT_NUM_S), vmax=np.max(PHOT_NUM_S))
    color_map   = plt.get_cmap("viridis")
    colors_S    = color_map(norm(PHOT_NUM_S))
    colors_T    = color_map(norm(PHOT_NUM_T))
    fig, ax = plt.subplots()
    for state in range(nstates+1):
        plt.scatter( R_LIST, E_QED_RCIS_S[:,state], c=colors_S[:,state] )
    for state in range(nstates):
        plt.scatter( R_LIST, E_QED_RCIS_T[:,state], c=colors_T[:,state] )
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color_map), ax=ax, label="Photon Number", pad=0.01)
    plt.xlabel("$R_\\mathrm{HH}$ (a.u.)", fontsize=15)
    plt.ylabel("Energy$ (a.u.)", fontsize=15) #  - E^{\\mathrm{VT}}(\\lambda = 0)
    plt.title("$\\lambda_\\mathrm{c}$ = %1.2f   $\\omega_\\mathrm{c}$ = %1.2f" % (LAM,WC), fontsize=15) #  - E^{\\mathrm{VT}}(\\lambda = 0)
    plt.tight_layout()
    plt.savefig("H2_QED_RCIS_BOND_SCAN_PHOT.jpg", dpi=300)
    plt.clf()


if (  __name__ == '__main__' ):
    # HH_LAM_SCAN_f_resolved()
    # LiH_LAM_SCAN_f_resolved()
    # LiH_LAM_SCAN_mins()
    # H2_LAM_SCAN_mins()
    # LiH_f_optimization()

    #H2_spin_contamination()
    #### LiH_spin_contamination() # NOT WORKING...

    #H2_LAM_SCAN_ALL_METHODS()
    #H2_Bond_Scan_ALL_METHODS()
    #H2_WC_SCAN_ALL_METHODS()

    #H2_DQMC_CCSD_comparison()
    
    
    
    #H2_QED_CIS_WC_SCAN()
    #H2_QED_CIS_LAM_SCAN()
    
    H2_QED_RCIS_BOND_SCAN()