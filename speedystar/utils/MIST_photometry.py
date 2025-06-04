import os
import numpy as np
from scipy import interpolate
from astropy import units as u
from astropy import constants as const
from tqdm import tqdm
import mwdust
import time

#For log10(z/z_sun) = -0.25, 0, +0.25, load in MIST bolometric corrections on 
#T_eff / logg/ A_v grid for different filters.
spectrum_datap00 = os.path.join(os.path.dirname(__file__), \
        'MIST_bologrid_VISTABessellGaiaDECamLSSTSDSS_0.0_reduced.txt')
spectrum_datap025 = os.path.join(os.path.dirname(__file__), \
        'MIST_bologrid_VISTABessellGaiaDECamLSSTSDSS_+0.25_reduced.txt')
spectrum_datam025 = os.path.join(os.path.dirname(__file__), \
        'MIST_bologrid_VISTABessellGaiaDECamLSSTSDSS_-0.25_reduced.txt')

T_eff, Logg, Av,Bessell_Up00, Bessell_Bp00, Bessell_Vp00, Bessell_Rp00, \
Bessell_Ip00, Gaia_G_EDR3p00, Gaia_BP_EDR3p00, Gaia_RP_EDR3p00, \
VISTA_Zp00, VISTA_Yp00, VISTA_Jp00, VISTA_Hp00, VISTA_Ksp00, DECam_up00, \
DECam_gp00, DECam_rp00, DECam_ip00, DECam_zp00, DECam_Yp00, LSST_up00, \
LSST_gp00, LSST_rp00, LSST_ip00, LSST_zp00, LSST_yp00,  \
SDSS_up00, SDSS_gp00, SDSS_rp00, SDSS_ip00, SDSS_zp00 \
    = np.loadtxt(spectrum_datap00, dtype = 'str', unpack=True)

T_eff, Logg, Av, Bessell_Um025, Bessell_Bm025, Bessell_Vm025, Bessell_Rm025, \
Bessell_Im025, Gaia_G_EDR3m025, Gaia_BP_EDR3m025, Gaia_RP_EDR3m025, \
VISTA_Zm025, VISTA_Ym025, VISTA_Jm025, VISTA_Hm025, VISTA_Ksm025, \
DECam_um025, DECam_gm025, DECam_rm025, DECam_im025, DECam_zm025, DECam_Ym025, \
LSST_um025, LSST_gm025, LSST_rm025, LSST_im025, LSST_zm025, LSST_ym025, \
SDSS_um025, SDSS_gm025, SDSS_rm025, SDSS_im025, SDSS_zm025 \
    = np.loadtxt(spectrum_datam025, dtype = 'str', unpack=True)

T_eff, Logg, Av, Bessell_Up025, Bessell_Bp025, Bessell_Vp025, Bessell_Rp025, \
Bessell_Ip025, Gaia_G_EDR3p025, Gaia_BP_EDR3p025, Gaia_RP_EDR3p025, \
VISTA_Zp025, VISTA_Yp025, VISTA_Jp025, VISTA_Hp025, VISTA_Ksp025, \
DECam_up025, DECam_gp025, DECam_rp025, DECam_ip025, DECam_zp025, DECam_Yp025, \
LSST_up025, LSST_gp025, LSST_rp025, LSST_ip025, LSST_zp025, LSST_yp025, \
SDSS_up025, SDSS_gp025, SDSS_rp025, SDSS_ip025, SDSS_zp025 \
    = np.loadtxt(spectrum_datap025, dtype = 'str', unpack=True)

legalBands = ['T_eff', 'Logg', 'Av', 'Bessell_U', 'Bessell_B', 'Bessell_V', 'Bessell_R', \
    'Bessell_I', 'Gaia_G', 'Gaia_BP', 'Gaia_RP', \
    'VISTA_Z', 'VISTA_Y', 'VISTA_J', 'VISTA_H', 'VISTA_Ks', 'DECam_u', \
    'DECam_g', 'DECam_r', 'DECam_i', 'DECam_z', 'DECam_Y', 'LSST_u', \
    'LSST_g', 'LSST_r', 'LSST_i', 'LSST_z', 'LSST_y',  \
    'SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z']

datp00 = np.loadtxt(spectrum_datap00, dtype = 'str', unpack=True)
datm025 = np.loadtxt(spectrum_datam025, dtype = 'str', unpack=True)
datp025 = np.loadtxt(spectrum_datap025, dtype = 'str', unpack=True)

p00dict = {}
m025dict = {}
p025dict = {}

for i, key in enumerate(legalBands):
    p00dict[key] = datp00[i,:]
    m025dict[key] = datm025[i,:]
    p025dict[key] = datp025[i,:]

def G_to_GRVS( G, V_I ):
    # From Gaia G band magnitude to Gaia G_RVS magnitude
    # Jordi+ 2010 , Table 3, second row:

    a = -0.0138
    b = 1.1168
    c = -0.1811
    d = 0.0085

    f = a + b * V_I + c * V_I**2. + d * V_I**3.

    return G - f # G_RVS magnitude

def get_e_vlos_old(V, T):

    #For an older pygaia version. Depreciated.
    #Calculate expected Gaia radial velocity error given stellar effective 
    #temperature and Johnson-Cousins V-band magnitude.

    from pygaia.errors.spectroscopic import vrad_error_sky_avg
   
    startypetemps=np.array([31500, 15700, 9700, 8080, 7220, 5920, 5660, 5280])
    startypes =            ['B0V', 'B5V', 'A0V','A5V','F0V','G0V','G5V','K0V']

    types = np.empty(len(V)).astype(str)

    for i in range(len(V)):
        types[i] = startypes[np.argmin(abs(T[i]-startypetemps))]

    e_vlos = vrad_error_sky_avg(V, types)

    return e_vlos

def get_U(T, logg, av, met):
    global interp_Bessell_U_p00, interp_Bessell_U_m025, interp_Bessell_U_p025
    #Define interpolation functions and compute U-band bolometric 
    #corrections, rounding sample to nearest quarter-dex in metallicity

    if 'interp_Bessell_U_p00' not in globals():
        interp_Bessell_U_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, Av)), Bessell_Up00)
    if 'interp_Bessell_U_m025' not in globals():
        interp_Bessell_U_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, Av)), Bessell_Um025)
    if 'interp_Bessell_U_p025' not in globals():
        interp_Bessell_U_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, Av)), Bessell_Up025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_Bessell_U_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_Bessell_U_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_Bessell_U_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_B(T, logg, av, met):
    global interp_Bessell_B_p00, interp_Bessell_B_m025, interp_Bessell_B_p025
    #Define interpolation functions and compute U-band bolometric 
    #corrections, rounding sample to nearest quarter-dex in metallicity

    if 'interp_Bessell_B_p00' not in globals():
        interp_Bessell_B_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, Av)), Bessell_Bp00)
    if 'interp_Bessell_B_m025' not in globals():
        interp_Bessell_B_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, Av)), Bessell_Bm025)
    if 'interp_Bessell_B_p025' not in globals():
        interp_Bessell_B_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, Av)), Bessell_Bp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_Bessell_B_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_Bessell_B_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_Bessell_B_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_V(T, logg, av, met):
    global interp_Bessell_V_p00, interp_Bessell_V_m025, interp_Bessell_V_p025

    if 'interp_Bessell_V_p00' not in globals():
        interp_Bessell_V_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Vp00)
    if 'interp_Bessell_V_m025' not in globals():
        interp_Bessell_V_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Vm025)
    if 'interp_Bessell_V_p025' not in globals():
        interp_Bessell_V_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Vp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_Bessell_V_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_Bessell_V_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_Bessell_V_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_R(T, logg, av, met):

    interp_Bessell_R_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Rp00)
    interp_Bessell_R_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Rm025)
    interp_Bessell_R_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Rp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_Bessell_R_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_Bessell_R_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_Bessell_R_p025(T[idx], logg[idx], av[idx])
    
    return np.array(BC,dtype='float')

def get_I(T, logg, av, met):
    global interp_Bessell_I_p00, interp_Bessell_I_m025, interp_Bessell_I_p025

    if 'interp_Bessell_I_p00' not in globals():
        interp_Bessell_I_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Ip00)
    if 'interp_Bessell_I_m025' not in globals():
        interp_Bessell_I_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Im025)
    if 'interp_Bessell_I_p025' not in globals():
        interp_Bessell_I_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Ip025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_Bessell_I_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_Bessell_I_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_Bessell_I_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_G(T, logg, av, met):
    global interp_Gaia_G_p00, interp_Gaia_G_m025, interp_Gaia_G_p025

    if 'interp_Gaia_G_p00' not in globals():
        interp_Gaia_G_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Gaia_G_EDR3p00)
    if 'interp_Gaia_G_m025' not in globals():
        interp_Gaia_G_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Gaia_G_EDR3m025)
    if 'interp_Gaia_G_p025' not in globals():
        interp_Gaia_G_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Gaia_G_EDR3p025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_Gaia_G_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_Gaia_G_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_Gaia_G_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_Rp(T, logg, av, met):
    global interp_Gaia_Rp_p00, interp_Gaia_Rp_m025, interp_Gaia_Rp_p025

    if 'interp_Gaia_Rp_p00' not in globals():
        interp_Gaia_Rp_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Gaia_RP_EDR3p00)
    if 'interp_Gaia_Rp_m025' not in globals():
        interp_Gaia_Rp_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Gaia_RP_EDR3m025)
    if 'interp_Gaia_Rp_p025' not in globals():
        interp_Gaia_Rp_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Gaia_RP_EDR3p025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_Gaia_Rp_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_Gaia_Rp_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_Gaia_Rp_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_Bp(T, logg, av, met):

    interp_Gaia_Bp_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Gaia_BP_EDR3p00)
    interp_Gaia_Bp_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Gaia_BP_EDR3m025)
    interp_Gaia_Bp_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Gaia_BP_EDR3p025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_Gaia_Bp_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_Gaia_Bp_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_Gaia_Bp_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_Z(T, logg, av, met):

    interp_VISTA_Z_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Zp00)
    interp_VISTA_Z_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Zm025)
    interp_VISTA_Z_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Zp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_VISTA_Z_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_VISTA_Z_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_VISTA_Z_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_Y(T, logg, av, met):

    interp_VISTA_Y_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Yp00)
    interp_VISTA_Y_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Ym025)
    interp_VISTA_Y_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Yp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_VISTA_Y_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_VISTA_Y_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_VISTA_Y_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_J(T, logg, av, met):

    interp_VISTA_J_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, \
                                Logg, Av)), VISTA_Jp00)
    interp_VISTA_J_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, \
                                Logg, Av)), VISTA_Jm025)
    interp_VISTA_J_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, \
                                Logg, Av)), VISTA_Jp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_VISTA_J_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_VISTA_J_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_VISTA_J_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_H(T, logg, av, met):

    interp_VISTA_H_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Hp00)
    interp_VISTA_H_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Hm025)
    interp_VISTA_H_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Hp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_VISTA_H_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_VISTA_H_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_VISTA_H_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_K(T, logg, av, met):

    interp_VISTA_K_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Ksp00)
    interp_VISTA_K_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Ksm025)
    interp_VISTA_K_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Ksp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_VISTA_K_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_VISTA_K_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_VISTA_K_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_DECam_u(T, logg, av, met):

    interp_DECam_u_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_up00)
    interp_DECam_u_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_um025)
    interp_DECam_u_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_up025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_DECam_u_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_DECam_u_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_DECam_u_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_DECam_g(T, logg, av, met):
    global interp_DECam_g_p00, interp_DECam_g_m025, interp_DECam_g_p025

    if 'interp_DECam_g_p00' not in globals():
        interp_DECam_g_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                        Av)), DECam_gp00)
    if 'interp_DECam_g_m025' not in globals():
        interp_DECam_g_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_gm025)
    if 'interp_DECam_g_p025' not in globals():
        interp_DECam_g_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_gp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_DECam_g_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_DECam_g_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_DECam_g_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_DECam_r(T, logg, av, met):
    global interp_DECam_r_p00, interp_DECam_r_m025, interp_DECam_r_p025

    if 'interp_DECam_r_p00' not in globals():
        interp_DECam_r_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_rp00)
    if 'interp_DECam_r_m025' not in globals():
        interp_DECam_r_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_rm025)
    if 'interp_DECam_r_p025' not in globals():
        interp_DECam_r_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_rp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_DECam_r_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_DECam_r_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_DECam_r_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_DECam_i(T, logg, av, met):

    interp_DECam_i_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_ip00)
    interp_DECam_i_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_im025)
    interp_DECam_i_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_ip025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_DECam_i_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_DECam_i_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_DECam_i_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_DECam_z(T, logg, av, met):

    interp_DECam_z_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_zp00)
    interp_DECam_z_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_zm025)
    interp_DECam_z_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_zp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_DECam_z_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_DECam_z_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_DECam_z_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_DECam_Y(T, logg, av, met):

    interp_DECam_Y_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_Yp00)
    interp_DECam_Y_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_Ym025)
    interp_DECam_Y_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_Yp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_DECam_Y_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_DECam_Y_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_DECam_Y_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_LSST_u(T, logg, av, met):

    interp_LSST_u_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_up00)
    interp_LSST_u_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_um025)
    interp_LSST_u_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_up025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_LSST_u_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_LSST_u_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_LSST_u_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_LSST_g(T, logg, av, met):

    interp_LSST_g_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_gp00)
    interp_LSST_g_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_gm025)
    interp_LSST_g_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_gp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_LSST_g_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_LSST_g_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_LSST_g_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_LSST_r(T, logg, av, met):

    interp_LSST_r_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_rp00)
    interp_LSST_r_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_rm025)
    interp_LSST_r_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_rp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_LSST_r_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_LSST_r_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_LSST_r_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_LSST_i(T, logg, av, met):

    interp_LSST_i_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_ip00)
    interp_LSST_i_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_im025)
    interp_LSST_i_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_ip025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_LSST_i_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_LSST_i_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_LSST_i_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_LSST_z(T, logg, av, met):

    interp_LSST_z_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_zp00)
    interp_LSST_z_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_zm025)
    interp_LSST_z_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_zp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_LSST_z_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_LSST_z_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_LSST_z_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_LSST_y(T, logg, av, met):

    interp_LSST_y_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_yp00)
    interp_LSST_y_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_ym025)
    interp_LSST_y_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_yp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_LSST_y_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_LSST_y_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_LSST_y_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_SDSS_u(T, logg, av, met):

    interp_SDSS_u_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), SDSS_up00)
    interp_SDSS_u_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), SDSS_um025)
    interp_SDSS_u_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), SDSS_up025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_SDSS_u_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_SDSS_u_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_SDSS_u_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_SDSS_g(T, logg, av, met):

    interp_SDSS_g_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), SDSS_gp00)
    interp_SDSS_g_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), SDSS_gm025)
    interp_SDSS_g_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), SDSS_gp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_SDSS_g_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_SDSS_g_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_SDSS_g_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_SDSS_r(T, logg, av, met):

    interp_SDSS_r_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), SDSS_rp00)
    interp_SDSS_r_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), SDSS_rm025)
    interp_SDSS_r_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), SDSS_rp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_SDSS_r_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_SDSS_r_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_SDSS_r_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_SDSS_i(T, logg, av, met):

    interp_SDSS_i_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), SDSS_ip00)
    interp_SDSS_i_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), SDSS_im025)
    interp_SDSS_i_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), SDSS_ip025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_SDSS_i_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_SDSS_i_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_SDSS_i_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_SDSS_z(T, logg, av, met):

    interp_SDSS_z_p00 =  interpolate.LinearNDInterpolator(list(zip(T_eff,
                                                        Logg, Av)), SDSS_zp00)
    interp_SDSS_z_m025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, 
                                                        Logg, Av)), SDSS_zm025)
    interp_SDSS_z_p025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, 
                                                        Logg, Av)), SDSS_zp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = interp_SDSS_z_m025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = interp_SDSS_z_p00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = interp_SDSS_z_p025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_Mag(T, logg, av, met, band):

    if 'interp_'+band+'_p00' not in bandInterps.keys():
        bandInterps['interp_'+band+'_p00'] =  \
            interpolate.LinearNDInterpolator( list(zip(T_eff, Logg, Av)), p00dict[band])
    if 'interp_'+band+'_m025' not in bandInterps.keys():
        bandInterps['interp_'+band+'_m025'] =  \
            interpolate.LinearNDInterpolator( list(zip(T_eff, Logg, Av)), m025dict[band])
    if 'interp_'+band+'_p025' not in bandInterps.keys():
        bandInterps['interp_'+band+'_p025'] =  \
            interpolate.LinearNDInterpolator( list(zip(T_eff, Logg, Av)), p025dict[band])

    BC = np.empty(len(T))
    idx = np.where((met<-0.125))[0]
    BC[idx] = bandInterps['interp_'+band+'_m025'](T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = bandInterps['interp_'+band+'_p00'](T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = bandInterps['interp_'+band+'_p025'](T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_Mags(av, r, l, b, M, Met, Teff, R, Lum, dust, bands):
    '''
    Computes apparent magnitudes given the input

    Parameters
    ----------
    av : real
        visual extinction [Mags]
        will be None-type when this function is called for the first time.
        If multiple calls are required, subsequent calls go faster if av
        is already calculated.
    r : Quantity
        distance from the Earth [kpc]
    b : Quantity
        Galactic latitude [deg]
    l : Quantity
        Galactic longitude [deg]
    M : Quantity
        Stellar mass [M_sun]
    Met : Quantity
        Metallicity xi \def log10(Z / 0.0142) 
    R : Quantity
        Stellar radius [R_sun]
    Lum : Quantity
        Stellar luminosity [L_sun]
    dust : DustMap
        DustMap to be used
    bands : list of strings
        Photometric bands in which to calculate mock magnitudes. 
        See sample.photometry()
    errors : list of strings
        Astometric/spectroscopic errors to calculate. See sample.photometry()    
    Returns
    -------
    Av : real
        Visual extinction at each star's position and distance [mags]
    Mags : dictionary
        Apparent magnitudes in the chosen bands. Keys are the elements 
        of bands, entries are dim-1 numpy arrays of size self.size
    errs: dictionary
        Chosen errors. Keys are the elements of errors, entries 
        are dim-1 numpy arrays of size self.size        
    '''
    global bandInterps

    if 'bandInterps' not in globals():
        bandInterps = {}
    
    Mags = {}
    MagFuncs = {'Bessell_U': get_U, 'Bessell_B': get_B, 'Bessell_V': get_V,   
                'Bessell_R': get_R, 'Bessell_I': get_I, 'Gaia_G': get_G,
                'Gaia_RP': get_Rp, 'Gaia_BP': get_Bp, 'VISTA_Z': get_Z,
                'VISTA_Y': get_Y, 'VISTA_J': get_J, 'VISTA_H': get_H,
                'VISTA_Ks': get_K, 'DECam_u': get_DECam_u, 
                'DECam_g': get_DECam_g, 'DECam_r': get_DECam_r, 
                'DECam_i': get_DECam_i, 'DECam_z': get_DECam_z,
                'DECam_Y': get_DECam_Y, 'LSST_u': get_LSST_u, 
                'LSST_g': get_LSST_g, 'LSST_r': get_LSST_r, 
                'LSST_i': get_LSST_i, 'LSST_z': get_LSST_z,
                'LSST_y': get_LSST_y, 'SDSS_u': get_SDSS_u, 
                'SDSS_g': get_SDSS_g, 'SDSS_r': get_SDSS_r, 
                'SDSS_i': get_SDSS_i, 'SDSS_z': get_SDSS_z}

    #errs = {}

    r, l, b, M, Lum = r*u.kpc, l*u.deg, b*u.deg, M*u.Msun, Lum*u.Lsun

    T = Teff * u.K                   # Temperature of the star at t = tage [K]
    R = (R * u.solRad).to(u.m)    # Radius of the star at t = tage [m]

    # Log of surface gravity in cgs
    logg = np.log10((const.G * M / R**2.).to(u.cm / u.s**2).value) 

    av = dust(l.to(u.deg).value, b.to(u.deg).value, r.to(u.kpc).value) * 2.682

    #if av is None:
    #    mu = 5.*np.log10(r.to(u.pc).value) - 5. # Distance modulus

    #    av = np.empty(len(l))

    #    for i in tqdm(range(len(l)),desc='Calculating dust extinction'):
            #For each star, calculate visual extinction by querying dust map
            #av[i] = dust.query_dust(l.to(u.deg)[i].value, 
            #                        b.to(u.deg)[i].value, mu[i]) * 2.682

            #av[i] = mwdust.Combined15()(l.to(u.deg)[i].value, b.to(u.deg)[i].value,r.to(u.kpc)[i].value))
    #        av[i] = dust(l.to(u.deg)[i].value, b.to(u.deg)[i].value,r.to(u.kpc)[i].value)           

    # Solar bolometric magnitude
    MbolSun = 4.74

    # Distance correction for computing the unreddened flux at Earth, [mag]
    dist_correction_Mag = (- 2.5 * np.log10(((10*u.pc/r)**2.).to(1))).value

    pbar = tqdm(range(len(bands)),desc='Calculating magnitudes')

    for band in bands:
        if band not in legalBands and band!='Gaia_GRVS':
            print('Band {} not recognized. Continuing.'.format(band))
            continue
        elif band=='Gaia_GRVS':
            continue

        pbar.update(1)
        #BC = MagFuncs[band](T.value, logg, av, Met)
        #print(np.nanmax(BC),np.nanmin(BC))
        BC = get_Mag(T, logg, av, Met, band)
        Mags[band] = MbolSun - 2.5*np.log10(Lum.value) - BC + \
                        dist_correction_Mag

    '''
    if 'Bessell_U' in bands:
        pbar.update(1)
        BC = get_U(T.value, logg, av, Met)
        UMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        UMag = UMag0 + dist_correction_Mag # Johnson-Cousins U magnitude [mag]
        Mags['Bessell_U'] = UMag

    if 'Bessell_B' in bands:
        pbar.update(1)
        BC = get_B(T.value, logg, av, Met)
        BMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        BMag = BMag0 + dist_correction_Mag # Johnson-Cousins B magnitude [mag]
        Mags['Bessell_B'] = BMag

    if 'Bessell_V' in bands:
        pbar.update(1)
        BC = get_V(T.value, logg, av, Met)
        VMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        VMag = VMag0 + dist_correction_Mag # Johnson-Cousins V magnitude [mag]
        Mags['Bessell_V'] = VMag
    if 'Bessell_R' in bands:
        pbar.update(1)
        BC = get_R(T.value, logg, av, Met)
        RMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        RMag = RMag0 + dist_correction_Mag # Johnson-Cousins R magnitude [mag]
        Mags['Bessell_R'] = RMag
    if 'Bessell_I' in bands:
        pbar.update(1)
        pass
        BC = get_I(T.value, logg, av, Met)
        IMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        IcMag = IMag0 + dist_correction_Mag # Johnson-Cousins I magnitude [mag]
        Mags['Bessell_I'] = IcMag
    if 'Gaia_G' in bands:
        pbar.update(1)
        pass
        BC = get_G(T.value, logg, av, Met)
        GMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        GMag = GMag0 + dist_correction_Mag # Gaia G magnitude [mag]
        Mags['Gaia_G'] = GMag
    if 'Gaia_RP' in bands:
        pbar.update(1)       
        pass
        BC = get_Rp(T.value, logg, av, Met)
        RPMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        RPMag = RPMag0 + dist_correction_Mag # Gaia G_RP magnitude [mag]
        Mags['Gaia_RP'] = RPMag
    if 'Gaia_BP' in bands:
        pbar.update(1)
        pass
        BC = get_Bp(T.value, logg, av, Met)
        BPMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        BPMag = BPMag0 + dist_correction_Mag # Gaia G_BP magnitude [mag]  
        Mags['Gaia_BP'] = BPMag
    if 'VISTA_Z' in bands:
        pbar.update(1)
        BC = get_Z(T.value, logg, av, Met)
        ZMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        ZMag = ZMag0 + dist_correction_Mag # VISTA Z magnitude [mag]
        Mags['VISTA_Z'] = ZMag
    if 'VISTA_Y' in bands:
        pbar.update(1)
        BC = get_Y(T.value, logg, av, Met)
        YMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        YMag = YMag0 + dist_correction_Mag # VISTA Y magnitude [mag]
        Mags['VISTA_Y'] = YMag
    if 'VISTA_J' in bands:
        pbar.update(1)
        pass
        BC = get_J(T.value, logg, av, Met)
        JMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        JMag = JMag0 + dist_correction_Mag # VISTA J magnitude [mag]  
        Mags['VISTA_J'] = JMag
    if 'VISTA_H' in bands:
        pbar.update(1)
        pass
        BC = get_H(T.value, logg, av, Met)
        HMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        HMag = HMag0 + dist_correction_Mag # VISTA H magnitude [mag]  
        Mags['VISTA_H'] = HMag
    if 'VISTA_K' in bands:
        pbar.update(1)
        pass
        BC = get_K(T.value, logg, av, Met)
        KMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        KMag = KMag0 + dist_correction_Mag # VISTA K magnitude [mag]  
        Mags['VISTA_K'] = KMag
    if 'DECam_u' in bands:
        pbar.update(1)
        pass
        BC = get_DECam_u(T.value, logg, av, Met)
        uMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        uMag = uMag0 + dist_correction_Mag # DECam u magnitude [mag]  
        Mags['DECam_u'] = uMag
    if 'DECam_g' in bands:
        pbar.update(1)
        pass
        BC = get_DECam_g(T.value, logg, av, Met)
        gMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        gMag = gMag0 + dist_correction_Mag # DECam g magnitude [mag]  
        Mags['DECam_g'] = gMag
    if 'DECam_r' in bands:
        pbar.update(1)
        pass
        BC = get_DECam_r(T.value, logg, av, Met)
        rMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        rMag = rMag0 + dist_correction_Mag # DECam r magnitude [mag]  
        Mags['DECam_r'] = rMag
    if 'DECam_i' in bands:
        pbar.update(1)
        pass
        BC = get_DECam_i(T.value, logg, av, Met)
        iMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        iMag = iMag0 + dist_correction_Mag # DECam i magnitude [mag]  
        Mags['DECam_i'] = iMag
    if 'DECam_z' in bands:
        pbar.update(1)
        pass
        BC = get_DECam_z(T.value, logg, av, Met)
        zMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        zMag = zMag0 + dist_correction_Mag # DECam z magnitude [mag]  
        Mags['DECam_z'] = zMag
    if 'DECam_Y' in bands:
        pbar.update(1)
        pass
        BC = get_DECam_Y(T.value, logg, av, Met)
        YMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        YMag = YMag0 + dist_correction_Mag # DECam Y magnitude [mag]  
        Mags['DECam_Y'] = YMag
    if 'LSST_u' in bands:
        pbar.update(1)
        pass
        BC = get_LSST_u(T.value, logg, av, Met)
        uMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        uMag = uMag0 + dist_correction_Mag # LSST u magnitude [mag]  
        Mags['LSST_u'] = uMag
    if 'LSST_g' in bands:
        pbar.update(1)
        pass
        BC = get_LSST_g(T.value, logg, av, Met)
        gMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        gMag = gMag0 + dist_correction_Mag # LSST g magnitude [mag]  
        Mags['LSST_g'] = gMag
    if 'LSST_r' in bands:
        pbar.update(1)
        pass
        BC = get_LSST_r(T.value, logg, av, Met)
        rMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        rMag = rMag0 + dist_correction_Mag # LSST r magnitude [mag]  
        Mags['LSST_r'] = rMag
    if 'LSST_i' in bands:
        pbar.update(1)
        pass
        BC = get_LSST_i(T.value, logg, av, Met)
        iMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        iMag = iMag0 + dist_correction_Mag # LSST i magnitude [mag]  
        Mags['LSST_i'] = iMag
    if 'LSST_z' in bands:
        pbar.update(1)
        pass
        BC = get_LSST_z(T.value, logg, av, Met)
        zMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        zMag = zMag0 + dist_correction_Mag # LSST z magnitude [mag]  
        Mags['LSST_z'] = zMag
    if 'LSST_y' in bands:
        pbar.update(1)
        pass
        BC = get_LSST_y(T.value, logg, av, Met)
        YMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        YMag = YMag0 + dist_correction_Mag # LSST y magnitude [mag]  
        Mags['LSST_y'] = YMag
    if 'SDSS_u' in bands:
        pbar.update(1)
        pass
        BC = get_SDSS_u(T.value, logg, av, Met)
        uMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        uMag = uMag0 + dist_correction_Mag # SDSS u magnitude [mag]  
        Mags['SDSS_u'] = uMag
    if 'SDSS_g' in bands:
        pbar.update(1)
        pass
        BC = get_SDSS_g(T.value, logg, av, Met)
        gMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        gMag = gMag0 + dist_correction_Mag # SDSS g magnitude [mag]  
        Mags['SDSS_g'] = gMag
    if 'SDSS_r' in bands:
        pbar.update(1)
        pass
        BC = get_SDSS_r(T.value, logg, av, Met)
        rMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        rMag = rMag0 + dist_correction_Mag # SDSS r magnitude [mag]  
        Mags['SDSS_r'] = rMag
    if 'SDSS_i' in bands:
        pbar.update(1)
        pass
        BC = get_SDSS_i(T.value, logg, av, Met)
        iMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        iMag = iMag0 + dist_correction_Mag # SDSS i magnitude [mag]  
        Mags['SDSS_i'] = iMag
    if 'SDSS_z' in bands:
        pbar.update(1)
        pass
        BC = get_SDSS_z(T.value, logg, av, Met)
        zMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        zMag = zMag0 + dist_correction_Mag # SDSS z magnitude [mag]  
        Mags['SDSS_z'] = zMag
    '''
    if 'Gaia_GRVS' in bands:

        if not 'Gaia_G' in bands:
            #BC = MagFuncs['Gaia_G'](T.value, logg, av, Met)
            BC = get_Mag(T, logg, av, Met, 'Gaia_G')
            Mags['Gaia_G'] = MbolSun - 2.5*np.log10(Lum.value) - BC + \
                        dist_correction_Mag
        if not 'Bessell_V' in bands:
            #BC = MagFuncs['Bessell_V'](T.value, logg, av, Met)
            BC = get_Mag(T, logg, av, Met, 'Bessell_V')
            Mags['Bessell_V'] = MbolSun - 2.5*np.log10(Lum.value) - BC + \
                        dist_correction_Mag
        if not 'Bessell_I' in bands:
            #BC = MagFuncs['Bessell_I'](T.value, logg, av, Met)
            BC = get_Mag(T, logg, av, Met, 'Bessell_I')
            Mags['Bessell_I'] = MbolSun - 2.5*np.log10(Lum.value) - BC + \
                        dist_correction_Mag
        V_I = Mags['Bessell_V'] - Mags['Bessell_I'] # V - Ic colour, [mag]

        pbar.update(1)
        #V_I = VMag - IcMag # V - Ic colour, [mag]
        GRVSMag = G_to_GRVS( Mags['Gaia_G'], V_I )

        Mags['Gaia_GRVS'] = GRVSMag
    '''

    # ============== Errors! ================== #
    from pygaia.errors.astrometric import proper_motion_uncertainty
    from pygaia.errors.astrometric import parallax_uncertainty
    from pygaia.errors.spectroscopic import radial_velocity_uncertainty

    #Calculate astrometric and radial velocity errors. May require 
    #calculation of apparent magnitudes in other bands if they aren't 
    #already available

    # Parallax error (PyGaia) [mas]
    if 'e_par' in errors:        
        if 'GMag' not in locals():
            #BC = get_G(T.value, logg, av, Met)
            BC = get_Mag(T, logg, av, Met, 'Gaia_G')
            GMag = MbolSun - 2.5*np.log10(Lum.value) - BC \
                + dist_correction_Mag
        errs['e_par'] = parallax_uncertainty(GMag)/1000

     # ICRS proper motion errors (PyGaia) [mas/yr]
    if 'e_pmra' in errors:
        if 'GMag' not in locals():
            #BC = get_G(T.value, logg, av, Met)
            BC = get_Mag(T, logg, av, Met, 'Gaia_G')
            GMag = MbolSun - 2.5*np.log10(Lum.value) - BC \
                    + dist_correction_Mag

        pmerrs = proper_motion_uncertainty(GMag)
        errs['e_pmra'] = pmerrs[0]/1000

    if 'e_pmdec' in errors:
        if 'pmerrs' not in locals():
            if 'GMag' not in locals():
                BC = get_Mag(T, logg, av, Met, 'Gaia_G')
                #BC = get_G(T.value, logg, av, Met)
                GMag = MbolSun - 2.5*np.log10(Lum.value) - BC \
                        + dist_correction_Mag           
            pmerrs = proper_motion_uncertainty(GMag)/1000
        errs['e_pmdec'] = pmerrs[1]/1000

        
    # heliocentric radial velocity error [km/s]
    if 'e_vlos' in errors:    
        if 'VMag' not in locals():
            #BC = get_V(T.value, logg, av, Met)
            BC = get_Mag(T, logg, av, Met, 'Bessell_V')
            VMag = MbolSun - 2.5*np.log10(Lum.value) - BC \
                    + dist_correction_Mag           
        if 'GRVSMag' not in locals():
            #BC = get_V(T.value, logg, av, Met)
            BC = get_Mag(T, logg, av, Met, 'Bessell_V')
            VMag = MbolSun - 2.5*np.log10(Lum.value) - BC \
                    + dist_correction_Mag           

            if not 'Gaia_G' in locals():
                #BC = get_G(T.value, logg, av, Met)
                BC = get_Mag(T, logg, av, Met, 'Gaia_G')
                GMag = MbolSun - 2.5*np.log10(Lum.value) - BC \
                    + dist_correction_Mag
            if not 'Bessell_V' in locals():
                #BC = get_V(T.value, logg, av, Met)
                BC = get_Mag(T, logg, av, Met, 'Bessell_V')
                VMag = MbolSun - 2.5*np.log10(Lum.value) - BC \
                    + dist_correction_Mag
            if not 'Bessell_I' in locals():
                #BC = get_I(T.value, logg, av, Met)
                BC = get_Mag(T, logg, av, Met, 'Bessell_I')
                IcMag = MbolSun - 2.5*np.log10(Lum.value) - BC \
                    + dist_correction_Mag

            V_I = VMag - IcMag # V - Ic colour, [mag]
            GRVSMag = G_to_GRVS( GMag, V_I )        

        #errs['e_vlos'] = get_e_vlos(VMag, T.value) 
        errs['e_vlos'] = radial_velocity_uncertainty2(GRVSMag, T, logg)
    '''
    return av, Mags

'''
import numpy as np
import ujson
import os

__all__ = ["radial_velocity_uncertainty"]

_default_release = "dr4"

_ROOT = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(_ROOT, "rv_uncertainty_model_coeffs.json")) as fp:
    _rv_unc_model = ujson.load(fp)
    fp.close()

_rv_nb_transits_dr4 = 32
_rv_nb_transits_dr5 = 64
_grvs_zeropoint = 21.317
_exposure_time = 4.4167032
_n_spectrum_per_transit = 3
_pixel_width_al = 0.02453
_band_width = 24.0
_median_background = 4.7
_n_ac_samples_bright = 10
_n_ac_samples_faint = 1
_ron = 3.2


def _in_interval(a: np.array([], float), left, right, closed="both"):
    """
    Check whether the input number or array elements lie within the given interval.

    Parameters
    ----------
    a : ndarray, float
        Input number(s) to check.
    left : float
        Left bound of interval, where left<=right.
    right : float
        Right bound of interval, where right>=left.
    closed : str
        Can be 'both' for a closed interval, 'neither' for an open interval, 'left' for a left-closed interval, or 'right' for a right-closed interval.

    Returns
    -------
    result : ndarray, boolean
        Result of the check, True or False
    """
    if left > right:
        raise ValueError("Left must be less than or equal to right")
    if closed == "both":
        return (a >= left) & (a <= right)
    elif closed == "left":
        return (a >= left) & (a < right)
    elif closed == "right":
        return (a > left) & (a <= right)
    elif closed == "neither":
        return (a > left) & (a < right)
    else:
        raise ValueError("The closed parameter must be one of both|left|right|neither")


def radial_velocity_uncertainty2(
    grvs: np.array([], float),
    teff: np.array([], float),
    logg: np.array([], float),
    release=_default_release,
):
    r"""
    Simulate the Gaia DR3 radial velocity uncertainty for the input list of
    :math:`G_\mathrm{RVS}`, :math:`T_\mathrm{eff}`, and :math:`\log(g)` values.

    Parameters
    ----------
    grvs : ndarray, float
        Value(s) of :math:`G_\mathrm{RVS}` for which the calculate the radial velocity uncertainty.
    teff : ndarray, float
        Value(s) of :math:`T_\mathrm{eff}` (in K) for which to calculate the radial velocity uncertainty.
    logg : ndarray, float
        Value(s) of :math:`\log(g)` for which to calculate the radial velocity uncertainty.

    Returns
    -------
    sigma_rv : ndarray, float
        Value(s) of the radial velocity uncertainty. NaNs are returned for input
        magnitudes, temperatures, and surface gravities outside the grids as defined in
        `Katz et al (2022)
        <https://ui.adsabs.harvard.edu/abs/2022arXiv220605902K/abstracti>`_ (their
        figures E.1 and F.1), or outside the model validity ranges quoted on the `Gaia
        Science Performance pages
        <https://www.cosmos.esa.int/web/gaia/science-performance#spectroscopic%20performance>`_.
    """
    ggrvs = np.array(grvs)
    tteff = np.array(teff)
    llogg = np.array(logg)
    if not (
        (tteff.size == llogg.size)
        and (tteff.size == ggrvs.size)
        and (llogg.size == ggrvs.size)
    ):
        raise ValueError("Arrays grvs, teff. and logg must be of same size")
    rv_uncs = np.full(tteff.shape, np.nan)
    if release.upper() == "DR3":
        coeffs = "dr3coeffs"
    else:
        coeffs = "dr45coeffs"
        if release.upper() == "DR4":
            rv_nb_transits = _rv_nb_transits_dr4
        else:
            rv_nb_transits = _rv_nb_transits_dr5

    for group in _rv_unc_model.keys():
        slots = _in_interval(
            tteff,
            _rv_unc_model[group]["teff"][0],
            _rv_unc_model[group]["teff"][1],
            _rv_unc_model[group]["teff"][2],
        ) & _in_interval(
            llogg,
            _rv_unc_model[group]["logg"][0],
            _rv_unc_model[group]["logg"][1],
            _rv_unc_model[group]["logg"][2],
        )
        if np.any(slots):
            if release.upper() == "DR3":
                rv_uncs[slots] = _rv_unc_model[group][coeffs]["sfloor"] + _rv_unc_model[
                    group
                ][coeffs]["b"] * np.exp(
                    _rv_unc_model[group][coeffs]["a"]
                    * (ggrvs[slots] - _rv_unc_model[group][coeffs]["grvs0"])
                )
            else:
                rv_expected_sig_to_noise = np.zeros_like(ggrvs)
                collected_signal = (
                    np.power(10.0, 0.4 * (_grvs_zeropoint - ggrvs))
                    * _exposure_time
                    * _n_spectrum_per_transit
                    * rv_nb_transits
                    * (_pixel_width_al / _band_width)
                )
                bck_per_sample = (
                    _median_background * _n_spectrum_per_transit * rv_nb_transits
                )
                rn_per_sample = _ron * _ron * _n_spectrum_per_transit * rv_nb_transits
                bright = ggrvs <= 7
                faint = np.logical_not(bright)
                rv_expected_sig_to_noise[bright] = collected_signal[bright] / np.sqrt(
                    collected_signal[bright]
                    + bck_per_sample * _n_ac_samples_bright
                    + rn_per_sample * _n_ac_samples_bright
                )
                rv_expected_sig_to_noise[faint] = collected_signal[faint] / np.sqrt(
                    collected_signal[faint]
                    + bck_per_sample * _n_ac_samples_faint
                    + rn_per_sample * _n_ac_samples_faint
                )
                slowsnr = _rv_unc_model[group][coeffs]["sbreak"] * np.power(
                    rv_expected_sig_to_noise[slots]
                    / _rv_unc_model[group][coeffs]["snrbreak"],
                    _rv_unc_model[group][coeffs]["f"],
                )
                shighsnr = _rv_unc_model[group][coeffs]["sfloor"] + (
                    _rv_unc_model[group][coeffs]["sbreak"]
                    - _rv_unc_model[group][coeffs]["sfloor"]
                ) * np.exp(
                    _rv_unc_model[group][coeffs]["g"]
                    * (
                        np.log10(rv_expected_sig_to_noise[slots])
                        - np.log10(_rv_unc_model[group][coeffs]["snrbreak"])
                    )
                )
                h = (
                    1
                    + np.tanh(
                        _rv_unc_model[group][coeffs]["k"]
                        * (
                            np.log10(rv_expected_sig_to_noise[slots])
                            - np.log10(_rv_unc_model[group][coeffs]["snrbreak"])
                        )
                    )
                ) / 2
                rv_uncs[slots] = h * shighsnr + (1 - h) * slowsnr

    #if release.upper() == "DR3":
    #    rv_uncs[(ggrvs > 14) | (rv_uncs > 20.0)] = np.nan
    #else:
    #    rv_uncs[(ggrvs > 16)] = np.nan
    #    rv_uncs[(ggrvs > 12) & (tteff > 7000)] = np.nan

    return rv_uncs
    '''