__ImportError__ = "One or more required external packages have not been installed. Please see requirements.txt"

import os

try:
    import numpy as np
    from scipy import interpolate
    from astropy import units as u
    from astropy import constants as const
    from tqdm import tqdm
except ImportError:
    raise ImportError(__ImportError__)

#For log10(z/z_sun) = -0.25, 0, +0.25, load in MIST bolometric corrections on 
#T_eff / logg/ A_v grid for different filters.
spectrum_datap00 = os.path.join(os.path.dirname(__file__), 'MIST_bologrid_VISTABessellGaiaDECamLSST_0.0_reduced.txt')
spectrum_datap025 = os.path.join(os.path.dirname(__file__), 'MIST_bologrid_VISTABessellGaiaDECamLSST_0.25_reduced.txt')
spectrum_datam025 = os.path.join(os.path.dirname(__file__), 'MIST_bologrid_VISTABessellGaiaDECamLSST_-0.25_reduced.txt')

T_eff, Logg, Av,Bessell_Up00, Bessell_Bp00, Bessell_Vp00, Bessell_Rp00, \
Bessell_Ip00, Gaia_G_EDR3p00, Gaia_BP_EDR3p00, Gaia_RP_EDR3p00, \
VISTA_Zp00, VISTA_Yp00, VISTA_Jp00, VISTA_Hp00, VISTA_Ksp00, DECam_up00, \
DECam_gp00, DECam_rp00, DECam_ip00, DECam_zp00, DECam_Yp00, LSST_up00, \
LSST_gp00, LSST_rp00, LSST_ip00, LSST_zp00, LSST_yp00 \
    = np.loadtxt(spectrum_datap00, dtype = 'str', unpack=True)

T_eff, Logg, Av, Bessell_Um025, Bessell_Bm025, Bessell_Vm025, Bessell_Rm025, \
Bessell_Im025, Gaia_G_EDR3m025, Gaia_BP_EDR3m025, Gaia_RP_EDR3m025, \
VISTA_Zm025, VISTA_Ym025, VISTA_Jm025, VISTA_Hm025, VISTA_Ksm025, \
DECam_um025, DECam_gm025, DECam_rm025, DECam_im025, DECam_zm025, DECam_Ym025, \
LSST_um025, LSST_gm025, LSST_rm025, LSST_im025, LSST_zm025, LSST_ym025 \
    = np.loadtxt(spectrum_datam025, dtype = 'str', unpack=True)

T_eff, Logg, Av, Bessell_Up025, Bessell_Bp025, Bessell_Vp025, Bessell_Rp025, \
Bessell_Ip025, Gaia_G_EDR3p025, Gaia_BP_EDR3p025, Gaia_RP_EDR3p025, \
VISTA_Zp025, VISTA_Yp025, VISTA_Jp025, VISTA_Hp025, VISTA_Ksp025, \
DECam_up025, DECam_gp025, DECam_rp025, DECam_ip025, DECam_zp025, DECam_Yp025, \
LSST_up025, LSST_gp025, LSST_rp025, LSST_ip025, LSST_zp025, LSST_yp025 \
    = np.loadtxt(spectrum_datap025, dtype = 'str', unpack=True)

def G_to_GRVS( G, V_I ):
    # From Gaia G band magnitude to Gaia G_RVS magnitude
    # Jordi+ 2010 , Table 3, second row:

    a = -0.0138
    b = 1.1168
    c = -0.1811
    d = 0.0085

    f = a + b * V_I + c * V_I**2. + d * V_I**3.

    return G - f # G_RVS magnitude

def get_e_vlos(V, T):

    #Calculate expected Gaia radial velocity error given stellar effective 
    #temperature and Johnson-Cousins V-band magnitude.
    try:
        from pygaia.errors.spectroscopic import vrad_error_sky_avg
    except ImportError:
        raise ImportError(__ImportError__)

    startypetemps=np.array([31500, 15700, 9700, 8080, 7220, 5920, 5660, 5280])
    startypes =            ['B0V', 'B5V', 'A0V','A5V','F0V','G0V','G5V','K0V']

    types = np.empty(len(V)).astype(str)

    for i in range(len(V)):
        types[i] = startypes[np.argmin(abs(T[i]-startypetemps))]

    e_vlos = vrad_error_sky_avg(V, types)

    return e_vlos

def get_U(T, logg, av, met):

    #Define interpolation functions and compute U-band bolometric 
    #corrections, rounding sample to nearest quarter-dex in metallicity

    rbf_2_Up00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Up00)
    rbf_2_Um025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Um025)
    rbf_2_Up025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Up025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_Um025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_Up00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_Up025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_B(T, logg, av, met):

    rbf_2_Bp00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Bp00)
    rbf_2_Bm025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Bm025)
    rbf_2_Bp025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Bp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_Bm025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_Bp00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_Bp025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_V(T, logg, av, met):

    rbf_2_Vp00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Vp00)
    rbf_2_Vm025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Vm025)
    rbf_2_Vp025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Vp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_Vm025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_Vp00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_Vp025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_R(T, logg, av, met):

    rbf_2_Rp00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Rp00)
    rbf_2_Rm025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Rm025)
    rbf_2_Rp025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Rp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_Rm025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_Rp00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_Rp025(T[idx], logg[idx], av[idx])
    
    return np.array(BC,dtype='float')

def get_I(T, logg, av, met):

    rbf_2_Ip00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Ip00)
    rbf_2_Im025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Im025)
    rbf_2_Ip025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Bessell_Ip025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_Im025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_Ip00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_Ip025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_G(T, logg, av, met):

    rbf_2_Gp00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Gaia_G_EDR3p00)
    rbf_2_Gm025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Gaia_G_EDR3m025)
    rbf_2_Gp025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Gaia_G_EDR3p025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_Gm025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_Gp00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_Gp025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_Rp(T, logg, av, met):

    rbf_2_Rpp00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Gaia_RP_EDR3p00)
    rbf_2_Rpm025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Gaia_RP_EDR3m025)
    rbf_2_Rpp025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Gaia_RP_EDR3p025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_Rpm025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_Rpp00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_Rpp025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_Bp(T, logg, av, met):

    rbf_2_Bpp00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Gaia_BP_EDR3p00)
    rbf_2_Bpm025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Gaia_BP_EDR3m025)
    rbf_2_Bpp025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), Gaia_BP_EDR3p025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_Bpm025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_Bpp00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_Bpp025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_Z(T, logg, av, met):

    rbf_2_Zp00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Zp00)
    rbf_2_Zm025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Zm025)
    rbf_2_Zp025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Zp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_Zm025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_Zp00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_Zp025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_Y(T, logg, av, met):

    rbf_2_Yp00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Yp00)
    rbf_2_Ym025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Ym025)
    rbf_2_Yp025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Yp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_Ym025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_Yp00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_Yp025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_J(T, logg, av, met):

    rbf_2_Jp00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, Av)), VISTA_Jp00)
    rbf_2_Jm025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, Av)), VISTA_Jm025)
    rbf_2_Jp025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, Av)), VISTA_Jp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_Jm025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_Jp00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_Jp025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_H(T, logg, av, met):

    rbf_2_Hp00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Hp00)
    rbf_2_Hm025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Hm025)
    rbf_2_Hp025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Hp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_Hm025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_Hp00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_Hp025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_K(T, logg, av, met):

    rbf_2_Kp00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Ksp00)
    rbf_2_Km025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Ksm025)
    rbf_2_Kp025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), VISTA_Ksp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_Km025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_Kp00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_Kp025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_DECam_u(T, logg, av, met):

    rbf_2_up00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_up00)
    rbf_2_um025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_um025)
    rbf_2_up025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_up025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_um025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_up00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_up025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_DECam_g(T, logg, av, met):

    rbf_2_gp00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_gp00)
    rbf_2_gm025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_gm025)
    rbf_2_gp025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_gp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_gm025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_gp00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_gp025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_DECam_r(T, logg, av, met):

    rbf_2_rp00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_rp00)
    rbf_2_rm025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_rm025)
    rbf_2_rp025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_rp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_rm025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_rp00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_rp025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_DECam_i(T, logg, av, met):

    rbf_2_ip00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_ip00)
    rbf_2_im025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_im025)
    rbf_2_ip025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_ip025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_im025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_ip00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_ip025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_DECam_z(T, logg, av, met):

    rbf_2_zp00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_zp00)
    rbf_2_zm025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_zm025)
    rbf_2_zp025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_zp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_zm025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_zp00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_zp025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_DECam_Y(T, logg, av, met):

    rbf_2_Yp00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_Yp00)
    rbf_2_Ym025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_Ym025)
    rbf_2_Yp025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), DECam_Yp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_Ym025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_Yp00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_Yp025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_LSST_u(T, logg, av, met):

    rbf_2_up00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_up00)
    rbf_2_um025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_um025)
    rbf_2_up025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_up025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_um025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_up00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_up025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_LSST_g(T, logg, av, met):

    rbf_2_gp00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_gp00)
    rbf_2_gm025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_gm025)
    rbf_2_gp025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_gp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_gm025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_gp00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_gp025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_LSST_r(T, logg, av, met):

    rbf_2_rp00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_rp00)
    rbf_2_rm025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_rm025)
    rbf_2_rp025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_rp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_rm025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_rp00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_rp025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_LSST_i(T, logg, av, met):

    rbf_2_ip00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_ip00)
    rbf_2_im025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_im025)
    rbf_2_ip025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_ip025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_im025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_ip00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_ip025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_LSST_z(T, logg, av, met):

    rbf_2_zp00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_zp00)
    rbf_2_zm025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_zm025)
    rbf_2_zp025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_zp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_zm025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_zp00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_zp025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_LSST_y(T, logg, av, met):

    rbf_2_yp00 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_yp00)
    rbf_2_ym025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_ym025)
    rbf_2_yp025 =  interpolate.LinearNDInterpolator(list(zip(T_eff, Logg, \
                                                    Av)), LSST_yp025)

    BC = np.empty(len(T))
    idx = np.where((met>=-0.25) & (met<-0.125))[0]
    BC[idx] = rbf_2_ym025(T[idx], logg[idx], av[idx])
    idx = np.where((met>=-0.125) & (met<0.125))[0]
    BC[idx] = rbf_2_yp00(T[idx], logg[idx], av[idx])
    idx = np.where(met>=0.125)[0]
    BC[idx] = rbf_2_yp025(T[idx], logg[idx], av[idx])

    return np.array(BC,dtype='float')

def get_Mags(av, r, l, b, M, Met, Teff, R, Lum, dust, bands, errors):
    #pbar.update(1)
    #Met = 0
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

    r, l, b, M, Lum = r*u.kpc, l*u.deg, b*u.deg, M*u.Msun, Lum*u.Lsun

    Mags = {}
    errs = {}

    T = Teff * u.K                   # Temperature of the star at t = tage [K]
    R = (R * u.solRad).to(u.m)    # Radius of the star at t = tage [m]

    # Log of surface gravity in cgs
    logg = np.log10((const.G * M / R**2.).to(u.cm / u.s**2).value) 

    #Calculate dust extinction if not already done
    if av is None:
        print('Photometry: calculating dust extinction...')
        mu = 5.*np.log10(r.to(u.pc).value) - 5. # Distance modulus

        av = np.empty(len(l))

        for i in tqdm(range(len(l))):
            #For each star, calculate visual extinction by querying dust map
            av[i] = dust.query_dust(l.to(u.deg)[i].value, 
                                    b.to(u.deg)[i].value, mu[i]) * 2.682

    # Solar bolometric magnitude
    MbolSun = 4.74

     # Distance correction for computing the unreddened flux at Earth, [mag]
    dist_correction_Mag = (- 2.5 * np.log10(((10*u.pc/r)**2.).to(1))).value

    pbar = tqdm(range(len(bands)))

    #For each possible band, interpolate MIST bolometric correction 
    #using temperature, surface gravity, extinction and metallicity. 
    #Use bolometric correction to calculate apparent magnitude in each band.
    if 'Bessell_U' in bands:
        print('Photometry: calculating U...')
        pbar.update(1)
        BC = get_U(T.value, logg, av, Met)
        #BC = get_U(T.value, logg, av, np.zeros(len(av)))
        UMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        UMag = UMag0 + dist_correction_Mag # Johnson-Cousins U magnitude [mag]
        Mags['Bessell_U'] = UMag

    if 'Bessell_B' in bands:
        print('Photometry: calculating B...')
        pbar.update(1)
        BC = get_B(T.value, logg, av, Met)
        #BC = get_B(T.value, logg, av, np.zeros(len(av)))
        BMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        BMag = BMag0 + dist_correction_Mag # Johnson-Cousins B magnitude [mag]
        Mags['Bessell_B'] = BMag

    if 'Bessell_V' in bands:
        print('Photometry: calculating V...')
        pbar.update(1)
        #BC = get_V(T.value, logg, av, np.zeros(len(av)))
        BC = get_V(T.value, logg, av, Met)
        VMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        VMag = VMag0 + dist_correction_Mag # Johnson-Cousins V magnitude [mag]
        Mags['Bessell_V'] = VMag
    if 'Bessell_R' in bands:
        print('Photometry: calculating R...')
        pbar.update(1)
        #BC = get_R(T.value, logg, av, np.zeros(len(av)))
        BC = get_R(T.value, logg, av, Met)
        RMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        RMag = RMag0 + dist_correction_Mag # Johnson-Cousins R magnitude [mag]
        Mags['Bessell_R'] = RMag
    if 'Bessell_I' in bands:
        print('Photometry: calculating I...')
        pbar.update(1)
        pass
        BC = get_I(T.value, logg, av, Met)
        #BC = get_I(T.value, logg, av, np.zeros(len(av)))
        IMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        IcMag = IMag0 + dist_correction_Mag # Johnson-Cousins I magnitude [mag]
        Mags['Bessell_I'] = IcMag
    if 'Gaia_G' in bands:
        print('Photometry: calculating Gaia G...')
        pbar.update(1)
        pass
        BC = get_G(T.value, logg, av, Met)
        #BC = get_G(T.value, logg, av, np.zeros(len(av)))
        GMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        GMag = GMag0 + dist_correction_Mag # Gaia G magnitude [mag]
        Mags['Gaia_G'] = GMag
    if 'Gaia_RP' in bands:
        print('Photometry: calculating Gaia G_RP...')
        pbar.update(1)       
        pass
        BC = get_Rp(T.value, logg, av, Met)
        #BC = get_Rp(T.value, logg, av, np.zeros(len(av)))
        RPMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        RPMag = RPMag0 + dist_correction_Mag # Gaia G_RP magnitude [mag]
        Mags['Gaia_RP'] = RPMag
    if 'Gaia_BP' in bands:
        print('Photometry: calculating Gaia G_BP...')
        pbar.update(1)
        pass
        BC = get_Bp(T.value, logg, av, Met)
        #BC = get_Bp(T.value, logg, av, np.zeros(len(av)))
        BPMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        BPMag = BPMag0 + dist_correction_Mag # Gaia G_BP magnitude [mag]  
        Mags['Gaia_BP'] = BPMag
    if 'VISTA_Z' in bands:
        print('Photometry: calculating VISTA Z...')
        pbar.update(1)
        BC = get_Z(T.value, logg, av, Met)
        #BC = get_Z(T.value, logg, av, np.zeros(len(av)))
        ZMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        ZMag = ZMag0 + dist_correction_Mag # VISTA Z magnitude [mag]
        Mags['VISTA_Z'] = ZMag
    if 'VISTA_Y' in bands:
        print('Photometry: calculating VISTA Y...')
        pbar.update(1)
        BC = get_Y(T.value, logg, av, Met)
        #BC = get_Y(T.value, logg, av, np.zeros(len(av)))
        YMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        YMag = YMag0 + dist_correction_Mag # VISTA Y magnitude [mag]
        Mags['VISTA_Y'] = YMag
    if 'VISTA_J' in bands:
        print('Photometry: calculating VISTA J...')
        pbar.update(1)
        pass
        BC = get_J(T.value, logg, av, Met)
        #BC = get_J(T.value, logg, av, np.zeros(len(av)))
        JMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        JMag = JMag0 + dist_correction_Mag # VISTA J magnitude [mag]  
        Mags['VISTA_J'] = JMag
    if 'VISTA_H' in bands:
        print('Photometry: calculating VISTA H...')
        pbar.update(1)
        pass
        BC = get_H(T.value, logg, av, Met)
        #BC = get_H(T.value, logg, av, np.zeros(len(av)))
        HMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        HMag = HMag0 + dist_correction_Mag # VISTA H magnitude [mag]  
        Mags['VISTA_H'] = HMag
    if 'VISTA_K' in bands:
        print('Photometry: calculating VISTA Ks...')
        pbar.update(1)
        pass
        BC = get_K(T.value, logg, av, Met)
        #BC = get_K(T.value, logg, av, np.zeros(len(av)))
        KMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        KMag = KMag0 + dist_correction_Mag # VISTA K magnitude [mag]  
        Mags['VISTA_K'] = KMag
    if 'DECam_u' in bands:
        print('Photometry: calculating DECam u...')
        pbar.update(1)
        pass
        BC = get_DECam_u(T.value, logg, av, Met)
        #BC = get_DECam_u(T.value, logg, av, np.zeros(len(av)))
        uMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        uMag = uMag0 + dist_correction_Mag # DECam u magnitude [mag]  
        Mags['DECam_u'] = uMag
    if 'DECam_g' in bands:
        print('Photometry: calculating DECam g...')
        pbar.update(1)
        pass
        BC = get_DECam_g(T.value, logg, av, Met)
        #BC = get_DECam_g(T.value, logg, av, np.zeros(len(av)))
        gMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        gMag = gMag0 + dist_correction_Mag # DECam g magnitude [mag]  
        Mags['DECam_g'] = gMag
    if 'DECam_r' in bands:
        print('Photometry: calculating DECam r...')
        pbar.update(1)
        pass
        BC = get_DECam_r(T.value, logg, av, Met)
        #BC = get_DECam_r(T.value, logg, av, np.zeros(len(av)))
        rMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        rMag = rMag0 + dist_correction_Mag # DECam r magnitude [mag]  
        Mags['DECam_r'] = rMag
    if 'DECam_i' in bands:
        print('Photometry: calculating DECam i...')
        pbar.update(1)
        pass
        BC = get_DECam_i(T.value, logg, av, Met)
        #BC = get_DECam_i(T.value, logg, av, np.zeros(len(av)))
        iMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        iMag = iMag0 + dist_correction_Mag # DECam i magnitude [mag]  
        Mags['DECam_i'] = iMag
    if 'DECam_z' in bands:
        print('Photometry: calculating DECam z...')
        pbar.update(1)
        pass
        BC = get_DECam_z(T.value, logg, av, Met)
        #BC = get_DECam_z(T.value, logg, av, np.zeros(len(av)))
        zMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        zMag = zMag0 + dist_correction_Mag # DECam z magnitude [mag]  
        Mags['DECam_z'] = zMag
    if 'DECam_Y' in bands:
        print('Photometry: calculating DECam Y...')
        pbar.update(1)
        pass
        BC = get_DECam_Y(T.value, logg, av, Met)
        #BC = get_DECam_Y(T.value, logg, av, np.zeros(len(av)))
        YMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        YMag = YMag0 + dist_correction_Mag # DECam Y magnitude [mag]  
        Mags['DECam_Y'] = YMag
    if 'LSST_u' in bands:
        print('Photometry: calculating LSST u...')
        pbar.update(1)
        pass
        BC = get_LSST_u(T.value, logg, av, Met)
        #BC = get_LSST_u(T.value, logg, av, np.zeros(len(av)))
        uMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        uMag = uMag0 + dist_correction_Mag # LSST u magnitude [mag]  
        Mags['LSST_u'] = uMag
    if 'LSST_g' in bands:
        print('Photometry: calculating LSST g...')
        pbar.update(1)
        pass
        BC = get_LSST_g(T.value, logg, av, Met)
        #BC = get_LSST_g(T.value, logg, av, np.zeros(len(av)))
        gMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        gMag = gMag0 + dist_correction_Mag # LSST g magnitude [mag]  
        Mags['LSST_g'] = gMag
    if 'LSST_r' in bands:
        print('Photometry: calculating LSST r...')
        pbar.update(1)
        pass
        BC = get_LSST_r(T.value, logg, av, Met)
        #BC = get_LSST_r(T.value, logg, av, np.zeros(len(av)))
        rMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        rMag = rMag0 + dist_correction_Mag # LSST r magnitude [mag]  
        Mags['LSST_r'] = rMag
    if 'LSST_i' in bands:
        print('Photometry: calculating LSST i...')
        pbar.update(1)
        pass
        BC = get_LSST_i(T.value, logg, av, Met)
        #BC = get_LSST_i(T.value, logg, av, np.zeros(len(av)))
        iMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        iMag = iMag0 + dist_correction_Mag # LSST i magnitude [mag]  
        Mags['LSST_i'] = iMag
    if 'LSST_z' in bands:
        print('Photometry: calculating LSST z...')
        pbar.update(1)
        pass
        BC = get_LSST_z(T.value, logg, av, Met)
        #BC = get_LSST_z(T.value, logg, av, np.zeros(len(av)))
        zMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        zMag = zMag0 + dist_correction_Mag # LSST z magnitude [mag]  
        Mags['LSST_z'] = zMag
    if 'LSST_y' in bands:
        print('Photometry: calculating LSST y...')
        pbar.update(1)
        pass
        BC = get_LSST_y(T.value, logg, av, Met)
        #BC = get_LSST_y(T.value, logg, av, np.zeros(len(av)))
        YMag0 = MbolSun - 2.5*np.log10(Lum.value) - BC
        YMag = YMag0 + dist_correction_Mag # LSST y magnitude [mag]  
        Mags['LSST_y'] = YMag

    if 'Gaia_GRVS' in bands:
        if not 'Gaia_G' in bands:
            BC = get_G(T.value, logg, av, Met)
            GMag = MbolSun - 2.5*np.log10(Lum.value) - BC \
                    + dist_correction_Mag
        if not 'Bessell_V' in bands:
            BC = get_V(T.value, logg, av, Met)
            VMag = MbolSun - 2.5*np.log10(Lum.value) - BC \
                    + dist_correction_Mag
        if not 'Bessell_I' in bands:
            BC = get_I(T.value, logg, av, Met)
            IcMag = MbolSun - 2.5*np.log10(Lum.value) - BC \
                    + dist_correction_Mag

        print('Photometry: calculating Gaia G_RVS...')
        pbar.update(1)
        V_I = VMag - IcMag # V - Ic colour, [mag]
        GRVS = G_to_GRVS( GMag, V_I )
        Mags['Gaia_GRVS'] = GRVS

    # ============== Errors! ================== #
    try:
        from pygaia.errors.astrometric import proper_motion_uncertainty
        from pygaia.errors.astrometric import parallax_uncertainty
    except ImportError:
        raise ImportError(__ImportError__)

    #Calculate astrometric and radial velocity errors. May require 
    #calculation of apparent magnitudes in other bands if they aren't 
    #already available

    # Parallax error (PyGaia) [mas]
    if 'e_par' in errors:        
        if 'Gaia_G' not in locals():
            BC = get_G(T.value, logg, av, Met)
            GMag = MbolSun - 2.5*np.log10(Lum.value) - BC \
                + dist_correction_Mag
        errs['e_par'] = parallax_uncertainty(GMag)/1000

     # ICRS proper motion errors (PyGaia) [mas/yr]
    if 'e_pmra' in errors:
        if 'GMag' not in locals():
            BC = get_G(T.value, logg, av, Met)
            GMag = MbolSun - 2.5*np.log10(Lum.value) - BC \
                    + dist_correction_Mag

        pmerrs = proper_motion_uncertainty(GMag)
        errs['e_pmra'] = pmerrs[0]/1000

    if 'e_pmdec' in errors:
        if 'pmerrs' not in locals():
            if 'GMag' not in locals():
                BC = get_G(T.value, logg, av, Met)
                GMag = MbolSun - 2.5*np.log10(Lum.value) - BC \
                        + dist_correction_Mag           
            pmerrs = proper_motion_uncertainty(GMag)/1000
        errs['e_pmdec'] = pmerrs[1]/1000

        
    # heliocentric radial velocity error [km/s]
    if 'e_vlos' in errors:    
        if 'VMag' not in locals():
            BC = get_V(T.value, logg, av, Met)
            VMag = MbolSun - 2.5*np.log10(Lum.value) - BC \
                    + dist_correction_Mag           
            
        errs['e_vlos'] = get_e_vlos(VMag, T.value) 

    return av, Mags, errs
