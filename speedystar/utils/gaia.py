#print('safe2.3')
import numpy as np
from scipy import interpolate
import time
from astropy import units as u
import utils.hurley_stellar_evolution as hse
from astropy import constants as const
import os

Met = 0.0 #Assumption: Considering only Solar Metallicities!

#interp_data = os.path.join(os.path.dirname(__file__), 'interp_data_'+np.str(Met)+'.txt')
interp_data = os.path.join(os.path.dirname(__file__), 'Id_A_avg_grid_MH_'+np.str(Met)+'_wbprp.txt')
interp_data2 = os.path.join(os.path.dirname(__file__), 'Id_A_avg_grid_MH_'+np.str(Met)+'_LSSTugr_Vega.txt')

spectrum_data = os.path.join(os.path.dirname(__file__), 'spectrum_data_'+np.str(Met)+'.txt')

Id, A_v, GMag_0, VMag_0, IcMag_0, BPMag_0, RPMag_0 = np.loadtxt(interp_data, unpack = True)
Id, A_v, uMag_0, gMag_0, rMag_0 = np.loadtxt(interp_data2,unpack=True)
 
rbf_2_G =  interpolate.Rbf(Id, A_v, GMag_0, function = 'linear')
rbf_2_V =  interpolate.Rbf(Id, A_v, VMag_0, function = 'linear')
rbf_2_Ic = interpolate.Rbf(Id, A_v, IcMag_0, function = 'linear')
rbf_2_BP =  interpolate.Rbf(Id, A_v, BPMag_0, function = 'linear')
rbf_2_RP = interpolate.Rbf(Id, A_v, RPMag_0, function = 'linear')
rbf_2_u = interpolate.Rbf(Id, A_v, uMag_0, function = 'linear')
rbf_2_g = interpolate.Rbf(Id, A_v, gMag_0, function = 'linear')
rbf_2_r = interpolate.Rbf(Id, A_v, rMag_0, function = 'linear')

files, Id, T, logg, met, Vt, Xh = np.loadtxt(spectrum_data, dtype = 'str', unpack=True)

def closest_spectrum(Teff,Logg, Met):
    '''
        Finds the spectrum from the BaSel library which matches the given
        Teff, Logg
    '''

    #interp_data = os.path.join(os.path.dirname(__file__), 'interp_data_'+np.str(Met)+'.txt')
    #spectrum_data = os.path.join(os.path.dirname(__file__), 'spectrum_data_'+np.str(Met)+'.txt')

    #Id, A_v, GMag_0, VMag_0, IcMag_0 = np.loadtxt(interp_data, unpack = True)
    #rbf_2_G =  interpolate.Rbf(Id, A_v, GMag_0, function = 'linear')
    #rbf_2_V =  interpolate.Rbf(Id, A_v, VMag_0, function = 'linear')
    #rbf_2_Ic = interpolate.Rbf(Id, A_v, IcMag_0, function = 'linear')
    #files, Id, T, logg, met, Vt, Xh = np.loadtxt(spectrum_data, dtype = 'str', unpack=True)

    Vturb = 2.00 # Atmospheric micro-turbulence velocity [km/s]
    XH = 0.00 # Mixing length

    #spectrum_data = os.path.join(os.path.dirname(__file__), 'spectrum_data.txt')
    files, Id, T, logg, met, Vt, Xh = np.loadtxt(spectrum_data, dtype = 'str', unpack=True)

    Id = np.array(Id,dtype='float')
    T = np.array(T,dtype='float')
    logg = np.array(logg,dtype='float')
    met = np.array(met,dtype='float')
    Vt = np.array(Vt, dtype = 'float')
    Xh = np.array(Xh, dtype='float')

    ds = np.sqrt( (T - Teff)**2. + (logg - Logg)**2. + (Met - met)**2. + (Vturb - Vt)**2. + (Xh - XH)**2. )
    #print([Teff,Logg])
    #print(np.min(ds))
    #print(np.argmin(ds))
    indexm = np.where(ds == np.min(ds)) # Chi-square minimization
    identification = Id[indexm]
    return identification

def G_to_GRVS( G, V_I ):
    # From Gaia G band magnitude to Gaia G_RVS magnitude
    # Jordi+ 2010 , Table 3, second row:

    a = -0.0138
    b = 1.1168
    c = -0.1811
    d = 0.0085

    f = a + b * V_I + c * V_I**2. + d * V_I**3.

    return G - f # G_RVS magnitude

def V_I_to_BP_RP( V_I ):
    # From V - Ic to BP - RR colour
    # Jordi+ 2010 , Table 3, eight row:

    a = -0.066
    b = 1.2061
    c = -0.0614
    d = 0.0041

    f = a + b * V_I + c * V_I**2. + d * V_I**3.

    return f # BP-RP colour

def get_e_vlos(V, age, M, Met):

    from pygaia.errors.spectroscopic import vradErrorSkyAvg

    T, R = hse.get_TempRad( M.to(u.solMass).value, Met, age.to(u.Myr).value) # Temperature [K], radius [solRad]

    startypetemps = np.array([31500, 15700, 9700, 8080, 7220, 5920, 5660, 5280])
    startypes =              ['B0V', 'B5V', 'A0V','A5V','F0V','G0V','G5V','K0V']

    #print(V)
    #print(T)
    #print(startypes[np.argmin(abs(T-startypetemps))])

    types = startypes[np.argmin(abs(T-startypetemps))]

    e_vlos = vradErrorSkyAvg(V, types)

    return e_vlos

def get_Mags(r, l, b, M, Met, age, dust):
    '''
        Computes Gaia Grvs magnitudes given the input.
        Written by TM (see author list)

        Parameters
        ----------
            r : Quantity
                distance form the Earth
            l : Quantity
                Galactic latitude
            b : Quantity
                Galactic longitude
            age : Quantity
                Stellar age
            dust : DustMap
                DustMap to be used

        Returns
        -------
            e_par, e_pmra, e_pmdec : Quantity
                errors in parallax, pmra* and pmdec.
    '''
    r, l, b, M, age = r*u.kpc, l*u.deg, b*u.deg, M*u.Msun, age*u.Myr

    t0 = time.time()
    #print('interp_data')
    #interp_data = os.path.join(os.path.dirname(__file__), 'interp_data_'+np.str(Met)+'.txt')
    #print('spectrum_data')
    #spectrum_data = os.path.join(os.path.dirname(__file__), 'spectrum_data_'+np.str(Met)+'.txt')
    #print('Id')
    #Id, A_v, GMag_0, VMag_0, IcMag_0 = np.loadtxt(interp_data, unpack = True)
    #print('rbf_2_G')
    #rbf_2_G =  interpolate.Rbf(Id, A_v, GMag_0, function = 'linear')
    #print('rbf_2_V')
    #rbf_2_V =  interpolate.Rbf(Id, A_v, VMag_0, function = 'linear')
    #print('rbf_2_Ic')
    #rbf_2_Ic = interpolate.Rbf(Id, A_v, IcMag_0, function = 'linear')
    #print('files')
    #files, Id, T, logg, met, Vt, Xh = np.loadtxt(spectrum_data, dtype = 'str', unpack=True)

    beta = np.arcsin(abs(0.497125407*np.sin(b) + 0.867678701*np.cos(b)*np.sin(l - 6.38 * u.deg)))
    #beta = np.arcsin(np.min(1.,abs(0.497125407*np.sin(b) + 0.867678701*np.cos(b)*np.sin(l - 6.38 * u.deg))))

    #print(abs(0.497125407*np.sin(b) + 0.867678701*np.cos(b)*np.sin(l - 6.38 * u.deg)),b,l,beta)
    #T, R = hse.get_TempRad( M.to(u.solMass).value, 0, age.to(u.Myr).value) # Temperature [K], radius [solRad]
    #print('zero')
    #print(R)
    T, R = hse.get_TempRad( M.to(u.solMass).value, Met, age.to(u.Myr).value) # Temperature [K], radius [solRad]
    #print('low met')
    #print(R)

    T = T * u.K                   # Temperature of the star at t = tage [K]
    R = (R * u.solRad).to(u.m)    # Radius of the star at t = tage [m]

    logg = np.log10((const.G * M / R**2.).to(u.cm / u.s**2).value) # Log of surface gravity in cgs

    tsetup = time.time()
    #print('get spec')
    Id = closest_spectrum(T.value, logg, Met) # ID of the best-matching spectrum (chi-squared minimization)
    Id = Id.squeeze() # Removes single-dimensional axes, essential for interpolating magnitudes
    tspec = time.time()

    #print('get attenuation')
    mu = 5.*np.log10(r.to(u.pc).value) - 5. # Distance modulus
    #print('l, b' + str(l) + ' '+ str(b))
    Av = dust.query_dust(l.to(u.deg).value, b.to(u.deg).value, mu) * 2.682
    tatten = time.time()

    #Interpolation: from Id, Av to magnitudes (not corrected for the distance!)
    #print('interpolating')
    GMag0 = rbf_2_G(Id, Av) # Gaia G magnitude, [mag]
    BPMag0 = rbf_2_BP(Id, Av) # Gaia BP magnitude, [mag]
    RPMag0 = rbf_2_RP(Id, Av) # Gaia_RP magnitude, [mag]
    VMag0 = rbf_2_V(Id, Av) # Johnson-Cousins V magnitude, [mag]
    IcMag0 = rbf_2_Ic(Id, Av) # Johnson-Cousins Ic magnitude, [mag]
    uMag0 = rbf_2_u(Id, Av)
    gMag0 = rbf_2_g(Id, Av)
    rMag0 = rbf_2_r(Id, Av)
    #print('done interpolating')
    tinterp = time.time()

    dist_correction_Mag = (- 2.5 * np.log10(((R/r)**2.).to(1))).value # Distance correction for computing the unreddened flux at Earth, [mag]

    #Magnitudes corrected for distance:
    GMag = GMag0 + dist_correction_Mag # Gaia G magnitude, [mag]
    VMag = VMag0 + dist_correction_Mag # Johnson-Cousins V magnitude, [mag]
    IcMag = IcMag0 + dist_correction_Mag # Johnson-Cousins Ic magnitude, [mag]
    BPMag = BPMag0 + dist_correction_Mag # Gaia BP magnitude, [mag]
    RPMag = RPMag0 + dist_correction_Mag # Gaia RP magnitude, [mag]
    uMag = uMag0 + dist_correction_Mag + 0.91 # LSST u magnitude [mag] (AB)
    gMag = gMag0 + dist_correction_Mag - 0.08 # LSST g magnitude [mag] (AB)
    rMag = rMag0 + dist_correction_Mag +0.16 # LSST r magnitude [mag] (AB)

    V_I = VMag - IcMag # V - Ic colour, [mag]

    # ============== Errors! ================== #
    from pygaia.errors.astrometric import properMotionError
    from pygaia.errors.astrometric import parallaxError

    e_par = parallaxError(GMag, V_I, beta) # Parallax error (PyGaia) [uas]
    e_pmra, e_pmdec = properMotionError(GMag, V_I, beta) # ICRS proper motions error (PyGaia) [uas/yr]

    e_vlos = get_e_vlos(VMag, age, M, Met)

    GRVS = G_to_GRVS( GMag, V_I )

    #BP_RP = V_I_to_BP_RP(V_I)
    BP_RP = BPMag - RPMag
    terr = time.time()
    
    ttotal = terr - t0
    #print(['setup time',(tsetup-t0)/ttotal])
    #print(['spec time',(tspec-tsetup)/ttotal])
    #print(['atten time',(tatten-tspec)/ttotal])
    #print(['interp time',(tinterp - tatten)/ttotal])
    #print(['err time',(terr-tinterp)/ttotal])

    #return GRVS, VMag, GMag, rMag, BP_RP, e_par, e_pmra, e_pmdec, e_vlos
    return GRVS, VMag, GMag, uMag, gMag, rMag, BP_RP, e_par, e_pmra, e_pmdec, e_vlos, T.value


def get_errors(r, l, b, M, age, dust):
    '''
        Computes Gaia Grvs magnitudes and errorbars given the input.
        Written by TM (see author list)

        Parameters
        ----------
            r : Quantity
                distance form the Earth
            l : Quantity
                Galactic latitude
            b : Quantity
                Galactic longitude
            age : Quantity
                Stellar age
            dust : DustMap
                DustMap to be used

        Returns
        -------
            e_par, e_pmra, e_pmdec : Quantity
                errors in parallax, pmra* and pmdec.
    '''
    r, l, b, M, age = r*u.kpc, l*u.deg, b*u.deg, M*u.Msun, age*u.Myr

    beta = np.arcsin(abs(0.497125407*np.sin(b) + 0.867678701*np.cos(b)*np.sin(l - 6.38 * u.deg)))
    #beta = np.arcsin(np.min(1.,abs(0.497125407*np.sin(b) + 0.867678701*np.cos(b)*np.sin(l - 6.38 * u.deg))))

    #print(abs(0.497125407*np.sin(b) + 0.867678701*np.cos(b)*np.sin(l - 6.38 * u.deg)),b,l,beta)
    T, R = hse.get_TempRad( M.to(u.solMass).value, Met, age.to(u.Myr).value) # Temperature [K], radius [solRad]
    print(T)
    T, R = hse.get_TempRad( M.to(u.solMass).value, 0, age.to(u.Myr).value) # Temperature [K], radius [solRad]
    print(T)

    T = T * u.K                   # Temperature of the star at t = tage [K]
    R = (R * u.solRad).to(u.m)    # Radius of the star at t = tage [m]

    logg = np.log10((const.G * M / R**2.).to(u.cm / u.s**2).value) # Log of surface gravity in cgs

    Id = closest_spectrum(T.value, logg) # ID of the best-matching spectrum (chi-squared minimization)
    Id = Id.squeeze() # Removes single-dimensional axes, essential for interpolating magnitudes

    #print('start attenuation')
    mu = 5.*np.log10(r.to(u.pc).value) - 5. # Distance modulus
    Av = dust.query_dust(l.to(u.deg).value, b.to(u.deg).value, mu) * 2.682

    #Interpolation: from Id, Av to magnitudes (not corrected for the distance!)

    GMag0 = rbf_2_G(Id, Av) # Gaia G magnitude, [mag]
    VMag0 = rbf_2_V(Id, Av) # Johnson-Cousins V magnitude, [mag]
    IcMag0 = rbf_2_Ic(Id, Av) # Johnson-Cousins Ic magnitude, [mag]

    dist_correction_Mag = (- 2.5 * np.log10(((R/r)**2.).to(1))).value # Distance correction for computing the unreddened flux at Earth, [mag]

    #Magnitudes corrected for distance:
    GMag = GMag0 + dist_correction_Mag # Gaia G magnitude, [mag]
    VMag = VMag0 + dist_correction_Mag # Johnson-Cousins V magnitude, [mag]
    IcMag = IcMag0 + dist_correction_Mag # Johnson-Cousins Ic magnitude, [mag]

    V_I = VMag - IcMag # V - Ic colour, [mag]

    # ============== Errors! ================== #
    from pygaia.errors.astrometric import properMotionError
    from pygaia.errors.astrometric import parallaxError

    e_par = parallaxError(GMag, V_I, beta) # Parallax error (PyGaia) [uas]
    e_pmra, e_pmdec = properMotionError(GMag, V_I, beta) # ICRS proper motions error (PyGaia) [uas/yr]

    #GRVS = G_to_GRVS( GMag, V_I )

    e_vlos = get_e_vlos(Vmag, age, M)

    return e_par, e_pmra, e_pmdec, e_vlos

get_GRVS = np.vectorize(get_Mags)
#get_errs = np.vectorize(get_errors)
