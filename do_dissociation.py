import numpy as np
import subprocess as sp
import matplotlib
from matplotlib import pyplot as plt

from pyscf import gto

from RHF import do_RHF
from UHF import do_UHF 
from QED_RHF import do_QED_RHF, do_QED_HF_ZHY
from QED_UHF import do_QED_UHF
from QED_VT_RHF import do_QED_VT_RHF
from QED_VT_UHF import do_QED_VT_UHF

def get_Globals():

    global basis_set, do_CS
    basis_set = 'ccpvdz' # 'aug-ccpvdz'
    do_CS = True

    global LAM_LIST, WC
    WC           = 0.1 # a.u.
    dL           = 0.05 # 0.01 # a.u.
    LAM_LIST     = np.arange(0.0, 0.1+dL, dL)

    global DATA_DIR
    DATA_DIR = "PLOTS_DATA"
    sp.call( f"mkdir -p {DATA_DIR}", shell=True )

def do_plots( R_LIST, QEDRHF, QEDUHF, QEDRHF_ZHY, QEDUHF_S2, QEDUHF_ss1, VTQEDRHF, VTQEDUHF, VTQEDUHF_S2, VTQEDUHF_ss1, RHF, UHF, UHF_S2, UHF_ss1, title="" ):
    title = title.strip("_")

    plt.plot( R_LIST, RHF[:] + WC/2, "-", lw=8, alpha=0.5, c="black", label="RHF" )
    plt.plot( R_LIST, UHF[:] + WC/2, "-", lw=8, alpha=0.5, c="blue", label="UHF" )
    for LAMi,LAM in enumerate( LAM_LIST ):
        plt.plot( R_LIST, QEDRHF[LAMi,:], "-", ms=3, markerfacecolor='none', c="black", label="QED-RHF" * (LAMi==0) )
        plt.plot( R_LIST, QEDUHF[LAMi,:], "-", ms=3, markerfacecolor='none', c="blue", label="QED-UHF" * (LAMi==0) )
        plt.plot( R_LIST, VTQEDRHF[LAMi,:], "--", ms=3, markerfacecolor='none', c="green", label="VT-QED-RHF" * (LAMi==0) )
        plt.plot( R_LIST, VTQEDUHF[LAMi,:], "-.", ms=3, markerfacecolor='none', c="orange", label="VT-QED-UHF" * (LAMi==0) )
    plt.xlabel("Nuclear Separation, $R$ (a.u.)", fontsize=15)
    plt.ylabel("Energy, $E_0$ (a.u.)", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/{title}_PES.jpg", dpi=300)
    plt.clf()
    outlist = [R_LIST]
    for LAMi,LAM in enumerate( LAM_LIST ):
        outlist.append( QEDRHF[LAMi,:] )
    np.savetxt(f"{DATA_DIR}/{title}_PES_QEDRHF.dat", np.array(outlist).T, fmt="%1.8f", header="R_LIST (Bohr), E_RHF (a.u.)")
    for LAMi,LAM in enumerate( LAM_LIST ):
       outlist.append( QEDUHF[LAMi,:] )
    np.savetxt(f"{DATA_DIR}/{title}_PES_QEDUHF.dat", np.array(outlist).T, fmt="%1.8f", header="R_LIST (Bohr), E_UHF (a.u.)")
    for LAMi,LAM in enumerate( LAM_LIST ):
        outlist.append( VTQEDRHF[LAMi,:] )
    np.savetxt(f"{DATA_DIR}/{title}_PES_VTQEDRHF.dat", np.array(outlist).T, fmt="%1.8f", header="R_LIST (Bohr), E_RHF (a.u.)")
    for LAMi,LAM in enumerate( LAM_LIST ):
       outlist.append( VTQEDUHF[LAMi,:] )
    np.savetxt(f"{DATA_DIR}/{title}_PES_VTQEDUHF.dat", np.array(outlist).T, fmt="%1.8f", header="R_LIST (Bohr), E_UHF (a.u.)")


    for LAMi,LAM in enumerate( LAM_LIST ):
        plt.semilogy( R_LIST, np.abs(QEDRHF[LAMi,:] - QEDUHF[LAMi,:]), "-", label="$\\lambda$ = %1.3f" % LAM )
    plt.xlabel("Nuclear Separation, $R$ (a.u.)", fontsize=15)
    plt.ylabel("$E_\\mathrm{QED-RHF}$ - $E_\\mathrm{QED-UHF}$ (a.u.)", fontsize=15)
    plt.ylim(1e-6,0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/{title}_PES_QEDRHF-QEDUHF.jpg", dpi=300)
    plt.clf()
    outlist = [R_LIST]
    for LAMi,LAM in enumerate( LAM_LIST ):
        outlist.append( QEDRHF[LAMi,:] - QEDUHF[LAMi,:] )
    np.savetxt(f"{DATA_DIR}/{title}_PES_QEDRHF-QEDUHF.dat", np.array(outlist).T, fmt="%1.8f", header="R_LIST (Bohr), E_QEDRHF-E_QEDUHF (a.u.) for LAM (a.u.) " + " ".join(["%1.3f" % LAM for LAM in LAM_LIST]))

    for LAMi,LAM in enumerate( LAM_LIST ):
        plt.semilogy( R_LIST, np.abs(VTQEDRHF[LAMi,:] - VTQEDUHF[LAMi,:]), "-", label="$\\lambda$ = %1.3f" % LAM )
    plt.xlabel("Nuclear Separation, $R$ (a.u.)", fontsize=15)
    plt.ylabel("$E_\\mathrm{QED-RHF}$ - $E_\\mathrm{QED-UHF}$ (a.u.)", fontsize=15)
    plt.ylim(1e-6,0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/{title}_PES_VTQEDRHF-VTQEDUHF.jpg", dpi=300)
    plt.clf()
    outlist = [R_LIST]
    for LAMi,LAM in enumerate( LAM_LIST ):
        outlist.append( QEDRHF[LAMi,:] - QEDUHF[LAMi,:] )
    np.savetxt(f"{DATA_DIR}/{title}_PES_VTQEDRHF-VTQEDUHF.dat", np.array(outlist).T, fmt="%1.8f", header="R_LIST (Bohr), E_VTQEDRHF-E_VTQEDUHF (a.u.) for LAM (a.u.) " + " ".join(["%1.3f" % LAM for LAM in LAM_LIST]))




    CF_POINTS = np.zeros( len(LAM_LIST) )
    for LAMi,LAM in enumerate( LAM_LIST ):
        for Ri,R in enumerate( R_LIST ):
            if ( QEDRHF[LAMi,Ri] - QEDUHF[LAMi,Ri] > 1e-6 ):        
                CF_POINTS[LAMi] = R
                break
    plt.plot( LAM_LIST, CF_POINTS, "o-" )
    plt.xlabel("Coupling Strength, $\\lambda$ (a.u.)", fontsize=15)
    plt.ylabel("Coulson-Fischer Points, $R_\\mathrm{CF}$ (a.u.)", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/{title}_PES_CFPs_QED.jpg", dpi=300)
    plt.clf()
    np.savetxt(f"{DATA_DIR}/{title}_PES_CFPs_QED.dat", np.c_[LAM_LIST, CF_POINTS], fmt="%1.8f", header="LAM_LIST (a.u.), CF_POINTS (Bohr)")


    CF_POINTS = np.zeros( len(LAM_LIST) )
    for LAMi,LAM in enumerate( LAM_LIST ):
        for Ri,R in enumerate( R_LIST ):
            if ( VTQEDRHF[LAMi,Ri] - VTQEDUHF[LAMi,Ri] > 1e-6 ):        
                CF_POINTS[LAMi] = R
                break
    plt.plot( LAM_LIST, CF_POINTS, "o-" )
    plt.xlabel("Coupling Strength, $\\lambda$ (a.u.)", fontsize=15)
    plt.ylabel("Coulson-Fischer Points, $R_\\mathrm{CF}$ (a.u.)", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/{title}_PES_CFPs_VTQED.jpg", dpi=300)
    plt.clf()
    np.savetxt(f"{DATA_DIR}/{title}_PES_CFPs_VTQED.dat", np.c_[LAM_LIST, CF_POINTS], fmt="%1.8f", header="LAM_LIST (a.u.), CF_POINTS (Bohr)")




    CF_POINTS = np.zeros( len(LAM_LIST) )
    for LAMi,LAM in enumerate( LAM_LIST ):
        for Ri,R in enumerate( R_LIST ):
            if ( QEDUHF_S2[LAMi,Ri] > 1e-2 ):
                CF_POINTS[LAMi] = R
                break
    plt.plot( LAM_LIST, CF_POINTS, "o-" )
    plt.xlabel("Coupling Strength, $\\lambda$ (a.u.)", fontsize=15)
    plt.ylabel("Coulson-Fischer Points, $R_\\mathrm{CF}$ (a.u.)", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/{title}_PES_CFPs_fromS2_QED.jpg", dpi=300)
    plt.clf()
    np.savetxt(f"{DATA_DIR}/{title}_PES_CFPs_fromS2_QED.dat", np.c_[LAM_LIST, CF_POINTS], fmt="%1.8f", header="LAM_LIST (a.u.), CF_POINTS (Bohr)")

    CF_POINTS = np.zeros( len(LAM_LIST) )
    for LAMi,LAM in enumerate( LAM_LIST ):
        for Ri,R in enumerate( R_LIST ):
            if ( VTQEDUHF_S2[LAMi,Ri] > 1e-2 ):
                CF_POINTS[LAMi] = R
                break
    plt.plot( LAM_LIST, CF_POINTS, "o-" )
    plt.xlabel("Coupling Strength, $\\lambda$ (a.u.)", fontsize=15)
    plt.ylabel("Coulson-Fischer Points, $R_\\mathrm{CF}$ (a.u.)", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/{title}_PES_CFPs_fromS2_VTQED.jpg", dpi=300)
    plt.clf()
    np.savetxt(f"{DATA_DIR}/{title}_PES_CFPs_fromS2_VTQED.dat", np.c_[LAM_LIST, CF_POINTS], fmt="%1.8f", header="LAM_LIST (a.u.), CF_POINTS (Bohr)")



    for LAMi,LAM in enumerate( LAM_LIST ):
        plt.plot( R_LIST, QEDUHF_S2[LAMi,:], "o-", label="$\\lambda$ = %1.3f" % LAM )
        plt.plot( R_LIST, VTQEDUHF_S2[LAMi,:], "--", label="$\\lambda$ = %1.3f" % LAM )
    plt.xlabel("Nuclear Separation, $R$ (a.u.)", fontsize=15)
    plt.ylabel("Spin-Squared Operator, $\\langle \\hat{S}^2 \\rangle$", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/{title}_PES_S2_QED_VTQED.jpg", dpi=300)
    plt.clf()
    outlist = [R_LIST]
    for LAMi,LAM in enumerate( LAM_LIST ):
        outlist.append( QEDUHF_S2[LAMi,:] )
    np.savetxt(f"{DATA_DIR}/{title}_PES_S2_QED.dat", np.array(outlist).T, fmt="%1.8f", header="R (a.u.), S2-Value for LAM (a.u.) = " + " ".join(["%1.3f" % LAM for LAM in LAM_LIST]) )
    outlist = [R_LIST]
    for LAMi,LAM in enumerate( LAM_LIST ):
        outlist.append( QEDUHF_ss1[LAMi,:] )
    np.savetxt(f"{DATA_DIR}/{title}_PES_ss1_QED.dat", np.array(outlist).T, fmt="%1.8f", header="R (a.u.), s(s+1)-Value for LAM (a.u.) = " + " ".join(["%1.3f" % LAM for LAM in LAM_LIST]) )
    for LAMi,LAM in enumerate( LAM_LIST ):
        outlist.append( VTQEDUHF_S2[LAMi,:] )
    np.savetxt(f"{DATA_DIR}/{title}_PES_S2_VTQED.dat", np.array(outlist).T, fmt="%1.8f", header="R (a.u.), S2-Value for LAM (a.u.) = " + " ".join(["%1.3f" % LAM for LAM in LAM_LIST]) )
    outlist = [R_LIST]
    for LAMi,LAM in enumerate( LAM_LIST ):
        outlist.append( VTQEDUHF_ss1[LAMi,:] )
    np.savetxt(f"{DATA_DIR}/{title}_PES_ss1_VTQED.dat", np.array(outlist).T, fmt="%1.8f", header="R (a.u.), s(s+1)-Value for LAM (a.u.) = " + " ".join(["%1.3f" % LAM for LAM in LAM_LIST]) )


    cmap = plt.get_cmap("brg")
    fig1, ax1 = plt.subplots()
    for LAMi,LAM in enumerate( LAM_LIST ):
        ax1.plot( R_LIST, QEDRHF[LAMi,:], "-", lw=2, c="black", label="QED-RHF" * (LAMi==0) )
        ax1.plot( R_LIST, QEDRHF_ZHY[LAMi,:], "--", lw=2, c="red", label="QED-RHF (ZHY)" * (LAMi==0) )
        ax1.plot( R_LIST, VTQEDRHF[LAMi,:], "--", ms=3, markerfacecolor='none', c="green", label="VT-QED-RHF" * (LAMi==0) )
        ax1.scatter( R_LIST, QEDUHF[LAMi,:], c=cmap(QEDUHF_S2[LAMi,:]), label="QED-UHF" * (LAMi==0) )
        ax1.scatter( R_LIST+(R_LIST[1]-R_LIST[0])/2, VTQEDRHF[LAMi,:], c=cmap(QEDUHF_S2[LAMi,:]), label="VT-QED-UHF" * (LAMi==0) )
    plt.xlabel("Nuclear Separation, $R$ (a.u.)", fontsize=15)
    plt.ylabel("Energy, $E_0$ (a.u.)", fontsize=15)
    plt.legend()
    cbar = fig1.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap),ax=ax1,pad=0.01)
    cbar.set_label(label='Spin-Squared Operator, $\\langle \\hat{S}^2 \\rangle$', size=15)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/{title}_PES_S2_Colored_QED.jpg", dpi=300)
    plt.clf()

    cmap = plt.get_cmap("brg")
    fig1, ax1 = plt.subplots()
    for LAMi,LAM in enumerate( LAM_LIST ):
        ax1.plot( R_LIST, QEDRHF[LAMi,:], "-", lw=2, c="black", label="QED-RHF" * (LAMi==0) )
        ax1.plot( R_LIST, QEDRHF_ZHY[LAMi,:], "--", lw=2, c="red", label="QED-RHF (ZHY)" * (LAMi==0) )
        ax1.plot( R_LIST, VTQEDRHF[LAMi,:], "--", ms=3, markerfacecolor='none', c="green", label="VT-QED-RHF" * (LAMi==0) )
        ax1.scatter( R_LIST, VTQEDRHF[LAMi,:], c=cmap(QEDUHF_S2[LAMi,:]), label="VT-QED-UHF" * (LAMi==0) )
    plt.xlabel("Nuclear Separation, $R$ (a.u.)", fontsize=15)
    plt.ylabel("Energy, $E_0$ (a.u.)", fontsize=15)
    plt.legend()
    cbar = fig1.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap),ax=ax1,pad=0.01)
    cbar.set_label(label='Spin-Squared Operator, $\\langle \\hat{S}^2 \\rangle$', size=15)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/{title}_PES_S2_Colored_VTQED.jpg", dpi=300)
    plt.clf()



def do_H2_Dissociation( mol ):

    R_LIST       = np.arange(0.6, 5.0, 0.05)

    VTQEDRHF      = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    VTQEDUHF      = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    VTQEDUHF_S2   = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    VTQEDUHF_ss1  = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDRHF_ZHY    = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDRHF        = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDUHF        = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDUHF_S2     = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDUHF_ss1    = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    RHF           = np.zeros( len(R_LIST) )
    UHF           = np.zeros( len(R_LIST) )
    UHF_S2        = np.zeros( len(R_LIST) )
    UHF_ss1       = np.zeros( len(R_LIST) )
    for Ri,R in enumerate( R_LIST ):
        print( "Working on %d of %d" % (Ri, len(R_LIST)) )
        mol.atom = 'H 0 0 0; H 0 0 %1.8f' % R
        mol.build()
        RHF[Ri]        = do_RHF( mol )
        UHF[Ri], UHF_S2[Ri], UHF_ss1[Ri] = do_UHF( mol )
        for LAMi,LAM in enumerate( LAM_LIST ):
            QEDRHF_ZHY[LAMi,Ri] = do_QED_HF_ZHY( mol, LAM, WC )
            VTQEDRHF[LAMi,Ri]   = np.array(do_QED_VT_RHF( mol, LAM, WC ))[0,-1] # [E_LIST, f_LIST]
            VTQEDUHF[LAMi,Ri], _, VTQEDUHF_S2[LAMi,Ri], VTQEDUHF_ss1[LAMi,Ri] = np.array(do_QED_VT_UHF( mol, LAM, WC ))[:,-1] # [E_LIST, f_LIST, S2_list, ss1_list]
            QEDRHF[LAMi,Ri]     = do_QED_RHF( mol, LAM, WC, do_CS=do_CS )
            QEDUHF[LAMi,Ri], QEDUHF_S2[LAMi,Ri], QEDUHF_ss1[LAMi,Ri] = do_QED_UHF( mol, LAM, WC, do_CS=do_CS )

    do_plots( R_LIST, QEDRHF, QEDUHF, QEDRHF_ZHY, QEDUHF_S2, QEDUHF_ss1, VTQEDRHF, VTQEDUHF, VTQEDUHF_S2, VTQEDUHF_ss1, RHF, UHF, UHF_S2, UHF_ss1, title="H2_" )

def do_LiH_Dissociation( mol ):

    dR          = 0.05 # 0.025
    R_LIST      = np.arange(3.6, 6+dR, dR)

    VTQEDRHF      = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    VTQEDUHF      = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    VTQEDUHF_S2   = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    VTQEDUHF_ss1  = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDRHF_ZHY    = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDRHF        = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDUHF        = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDUHF_S2     = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDUHF_ss1    = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    RHF           = np.zeros( len(R_LIST) )
    UHF           = np.zeros( len(R_LIST) )
    UHF_S2        = np.zeros( len(R_LIST) )
    UHF_ss1       = np.zeros( len(R_LIST) )
    for Ri,R in enumerate( R_LIST ):
        print( "Working on %d of %d" % (Ri, len(R_LIST)) )
        mol.atom = 'Li 0 0 0; H 0 0 %1.8f' % R
        mol.build()
        RHF[Ri]        = do_RHF( mol )
        UHF[Ri], UHF_S2[Ri], UHF_ss1[Ri] = do_UHF( mol )
        for LAMi,LAM in enumerate( LAM_LIST ):
            QEDRHF_ZHY[LAMi,Ri] = do_QED_HF_ZHY( mol, LAM, WC )
            VTQEDRHF[LAMi,Ri]   = np.array(do_QED_VT_RHF( mol, LAM, WC ))[0,-1] # [E_LIST, f_LIST]
            VTQEDUHF[LAMi,Ri], _, VTQEDUHF_S2[LAMi,Ri], VTQEDUHF_ss1[LAMi,Ri] = np.array(do_QED_VT_UHF( mol, LAM, WC ))[:,-1] # [E_LIST, f_LIST, S2_list, ss1_list]
            QEDRHF[LAMi,Ri]     = do_QED_RHF( mol, LAM, WC, do_CS=do_CS )
            QEDUHF[LAMi,Ri], QEDUHF_S2[LAMi,Ri], QEDUHF_ss1[LAMi,Ri] = do_QED_UHF( mol, LAM, WC, do_CS=do_CS )

    do_plots( R_LIST, QEDRHF, QEDUHF, QEDRHF_ZHY, QEDUHF_S2, QEDUHF_ss1, VTQEDRHF, VTQEDUHF, VTQEDUHF_S2, VTQEDUHF_ss1, RHF, UHF, UHF_S2, UHF_ss1, title="H2_" )

def do_N2_Dissociation( mol ):

    dR          = 0.05 # 0.025
    R_LIST      = np.arange(1.5, 6+dR, dR)

    VTQEDRHF      = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    VTQEDUHF      = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    VTQEDUHF_S2   = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    VTQEDUHF_ss1  = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDRHF_ZHY    = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDRHF        = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDUHF        = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDUHF_S2     = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDUHF_ss1    = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    RHF           = np.zeros( len(R_LIST) )
    UHF           = np.zeros( len(R_LIST) )
    UHF_S2        = np.zeros( len(R_LIST) )
    UHF_ss1       = np.zeros( len(R_LIST) )
    for Ri,R in enumerate( R_LIST ):
        print( "Working on %d of %d" % (Ri, len(R_LIST)) )
        mol.atom = 'N 0 0 0; N 0 0 %1.8f' % R
        mol.build()
        RHF[Ri]        = do_RHF( mol )
        UHF[Ri], UHF_S2[Ri], UHF_ss1[Ri] = do_UHF( mol )
        for LAMi,LAM in enumerate( LAM_LIST ):
            QEDRHF_ZHY[LAMi,Ri] = do_QED_HF_ZHY( mol, LAM, WC )
            VTQEDRHF[LAMi,Ri]   = np.array(do_QED_VT_RHF( mol, LAM, WC ))[0,-1] # [E_LIST, f_LIST]
            VTQEDUHF[LAMi,Ri], _, VTQEDUHF_S2[LAMi,Ri], VTQEDUHF_ss1[LAMi,Ri] = np.array(do_QED_VT_UHF( mol, LAM, WC ))[:,-1] # [E_LIST, f_LIST, S2_list, ss1_list]
            QEDRHF[LAMi,Ri]     = do_QED_RHF( mol, LAM, WC, do_CS=do_CS )
            QEDUHF[LAMi,Ri], QEDUHF_S2[LAMi,Ri], QEDUHF_ss1[LAMi,Ri] = do_QED_UHF( mol, LAM, WC, do_CS=do_CS )

    do_plots( R_LIST, QEDRHF, QEDUHF, QEDRHF_ZHY, QEDUHF_S2, QEDUHF_ss1, VTQEDRHF, VTQEDUHF, VTQEDUHF_S2, VTQEDUHF_ss1, RHF, UHF, UHF_S2, UHF_ss1, title="H2_" )

def do_C2_Dissociation( mol ):

    dR          = 0.05 # 0.025
    R_LIST      = np.arange(1.5, 6+dR, dR)

    VTQEDRHF      = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    VTQEDUHF      = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    VTQEDUHF_S2   = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    VTQEDUHF_ss1  = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDRHF_ZHY    = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDRHF        = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDUHF        = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDUHF_S2     = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    QEDUHF_ss1    = np.zeros( (len(LAM_LIST),len(R_LIST)) )
    RHF           = np.zeros( len(R_LIST) )
    UHF           = np.zeros( len(R_LIST) )
    UHF_S2        = np.zeros( len(R_LIST) )
    UHF_ss1       = np.zeros( len(R_LIST) )
    for Ri,R in enumerate( R_LIST ):
        print( "Working on %d of %d" % (Ri, len(R_LIST)) )
        mol.atom = 'C 0 0 0; C 0 0 %1.8f' % R
        mol.build()
        RHF[Ri]        = do_RHF( mol )
        UHF[Ri], UHF_S2[Ri], UHF_ss1[Ri] = do_UHF( mol )
        for LAMi,LAM in enumerate( LAM_LIST ):
            QEDRHF_ZHY[LAMi,Ri] = do_QED_HF_ZHY( mol, LAM, WC )
            VTQEDRHF[LAMi,Ri]   = np.array(do_QED_VT_RHF( mol, LAM, WC ))[0,-1] # [E_LIST, f_LIST]
            VTQEDUHF[LAMi,Ri], _, VTQEDUHF_S2[LAMi,Ri], VTQEDUHF_ss1[LAMi,Ri] = np.array(do_QED_VT_UHF( mol, LAM, WC ))[:,-1] # [E_LIST, f_LIST, S2_list, ss1_list]
            QEDRHF[LAMi,Ri]     = do_QED_RHF( mol, LAM, WC, do_CS=do_CS )
            QEDUHF[LAMi,Ri], QEDUHF_S2[LAMi,Ri], QEDUHF_ss1[LAMi,Ri] = do_QED_UHF( mol, LAM, WC, do_CS=do_CS )

    do_plots( R_LIST, QEDRHF, QEDUHF, QEDRHF_ZHY, QEDUHF_S2, QEDUHF_ss1, VTQEDRHF, VTQEDUHF, VTQEDUHF_S2, VTQEDUHF_ss1, RHF, UHF, UHF_S2, UHF_ss1, title="C2_" )


if ( __name__ == "__main__" ):

    get_Globals()

    mol = gto.Mole()
    mol.basis = basis_set
    mol.unit = 'Bohr'
    mol.symmetry = False

    do_H2_Dissociation( mol )
    do_LiH_Dissociation( mol )
    do_N2_Dissociation( mol )
    do_C2_Dissociation( mol )






