import numpy as np
import matplotlib.pyplot as plt
import math

import astropy
import astropy.units as u
import astropy.constants as const

from scipy.integrate import simps
from scipy.interpolate import interp1d

Myr = u.def_unit('Myr',1.e6 * u.yr) # [Myr] definition

# ------- V_extinction ----------
# - Taking in input the Galactic latitude b (in radians) and the Galacto3.240-centric radius r (in kpc),
# - this function gives back the visual extinction Av at 550 nm

def V_extinction(b,r):
    b = b.to(u.deg) # conversion to degrees
    if np.abs(b) > 10 * u.deg:
        return 1. * u.mag
    else:
        return 0.7 * (r / u.kpc) * u.mag
        

# ------ extinction ---------
# - Taking in input the visual extinction Av at 550 nm and the wavelength lam (in um), this function gives back
# - the extinction A(lam) at the given wavelength, assuming Rv = 3.1 and the expressions in (Cardelli et al. 1989).

def extinction(Av,lam):

	x = (1./lam).to(1/u.um).value # [um^-1]

	Rv = 3.1

	a = np.empty(np.size(x))
	b = np.empty(np.size(x))

	#  ======= IR ============= #

	index_IR = np.where( (x >= 0.3) & (x < 1.1) ) # IR

	a[index_IR] = 0.574*x[index_IR]**1.61	
	b[index_IR] = -0.527*x[index_IR]**1.61

	# ========== Optical/NIR =============== #

	index_OPT = np.where( (x >= 1.1) & (x < 3.3) ) # Optical/NIR

	y = x[index_OPT] - 1.82

	a[index_OPT] = 1. + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
	b[index_OPT] = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7

	# ============ UV ==================== #

	index_UV = np.where( (x >= 3.3) & (x < 8.) ) # UV

	Fa = np.empty(np.size(x))
	Fb = np.empty(np.size(x))

	index_UV1 = np.where( (x >= 3.3) & (x < 5.9) ) # UV - 1
	Fa[index_UV1] = 0.
	Fb[index_UV1] = 0.

	index_UV2 = np.where( (x >= 5.9) & (x < 8) ) # UV - 2
	Fa[index_UV2] = -0.04473*(x[index_UV2] - 5.9)**2. - 0.009779*(x[index_UV2] - 5.9)**3.
	Fb[index_UV2] = 0.2130*(x[index_UV2] - 5.9)**2. + 0.1207*(x[index_UV2] - 5.9)**3.

	a[index_UV] = 1.752 - 0.316*x[index_UV] - 0.104 / ( (x[index_UV] - 4.67)**2. + 0.341) + Fa[index_UV]
	b[index_UV] = -3.090 + 1.825*x[index_UV] + 1.206 / ( (x[index_UV] - 4.62)**2. + 0.263) + Fb[index_UV]
	
	# =========== F-UV ====================== #

	index_FUV = np.where( (x >=8.) & (x <= 10.) ) # Far UV
	
	a[index_FUV] = -1.073 - 0.628*(x[index_FUV]-8) + 0.137*(x[index_FUV]-8)**2. - 0.070*(x[index_FUV]-8)**3.
	b[index_FUV] = 13.670 + 4.257*(x[index_FUV]-8) - 0.420*(x[index_FUV]-8)**2. + 0.374*(x[index_FUV]-8)**3.

	return Av*(a + b/Rv)

# -------- A_l_avg----------------
# - Given the absorption A_l as a function of the wavelength wlen, this function computes the average
# - absorption on the filter defined by the transmission T.

def A_l_avg(A_l,wlen,T):

	return simps( y = A_l*T, x = wlen)/simps(y = T, x = wlen) 


# ------- Xi ---------
# - Taking in input the metallicity abundance, this function gives back a logarithmic expression for the metallicity, useful to compute the MS lifetime

def Xi(MH):
	
	ZSun = 0.02 # Solar metallicity
	Z = ZSun * 10.**MH

	return np.log10(Z/ZSun) 

      
# ------- r_GC --------------          
# - Taking in input the Galactic longitude l (in radians), the Galactic latitude b (in radians), the helio-centric radius r (in kpc),
# - and the distance from the Sun to the GC dLSR [kpc], this function gives back the radial distance of the object from the Galactic Center (in kpc).

def r_GC(l,b,r,dLSR):
    return np.sqrt(r**2 + dLSR**2 - 2.*r*dLSR*np.cos(l)*np.cos(b))
    

# ----- Potentials ------
# Bulge (Hernquist spheroid), disk (Miyamoto-Nagai), halo (NFW profile)
# Total potential = bulge + disk + halo

def phi_b(r,Mb,rb):
    return -const.G*Mb/(r+rb)

def phi_d(r,b,Md,ad,bd):
    R = r*np.cos(b) 
    z = r*np.sin(b) 
    return -const.G*Md/np.sqrt(R**2 + (ad + np.sqrt(z**2 + bd**2))**2 )

def phi_h(r,Mh,rh):
    return -const.G*Mh*np.log(1. + r/rh)/r

def phi_tot(r,b,Mb,rb,Md,ad,bd,Mh,rh):
    return phi_b(r,Mb,rb) + phi_d(r,b,Md,ad,bd) + phi_h(r,Mh,rh)

# ------- v_esc -----------
# - Taking in input the GC distance r (in kpc), and the Galactic latitude b (in radians), this function gives back the escape velocity
# - of the star (in km/s), assuming a three-component Galactic potential.

def v_esc(r,b,Mb,rb,Md,ad,bd,Mh,rh):
    return np.sqrt(-2. * phi_tot(r,b,Mb,rb,Md,ad,bd,Mh,rh) ) #escape velocity, [km/s]


# ------- v_ejection -----------
# - Taking in input the mass of the ejected hypervelocity star (m_HVS), the mass of the bound star (m_b) and the semi-major axis
# - of the distrupted binary (a), this function gives back the ejection velocity according to the formula in Rossi et al. 2014

def v_ejection(m_HVS,m_b,a):
	
	M_BH = 4.e6 * u.solMass # MBH mass, [MSun]

	return np.sqrt(2. * const.G * m_b / a) * ( M_BH / (m_HVS+m_b) )**(1./6.) 

# ------- v_decelerated ---------
# - Taking in input the ejection velociy v_ej and the parameters of the potential, this function commputes the resulting decelerated
#- velocity at the poisition (b,r).

def v_decelerated(v_ej,r,b,r0,Mb,rb,Md,ad,bd,Mh,rh):

	v_d_2 = v_ej**2. + 2.*( phi_tot(r0,b,Mb,rb,Md,ad,bd,Mh,rh) - phi_tot(r,b,Mb,rb,Md,ad,bd,Mh,rh) ) # v^2 [km^2/s^2]
	
	if v_d_2 < 0:

		return 0. * u.km / u.s
	
	else:
		return np.sqrt(v_d_2)

# ------ prop_motions ------
# - Taking in input the Galactic coordinates (l,b,r) [rad, rad, kpc], the velocity vGC (in km/s),
# - this function gives back the Galactic proper motions mul* and mub (in muas/yr) and the radial velocity vrad (in km/s).

def prop_motions(l,b,r,vGC,dLSR):
    
	mul = vGC*dLSR/r* np.sin(l) / r_GC(l,b,r,dLSR)	
	mub = vGC*dLSR/r* np.cos(l)*np.sin(b) / r_GC(l,b,r,dLSR)  
	vrad = vGC* (r - dLSR*np.cos(l)*np.cos(b)) / r_GC(l,b,r,dLSR)
	
	return mul, mub, vrad

# ------ Sun_motion_correction ------

def Sun_motion_correction(l,b,r):

	# From Schonrich 2012
	v_sun = np.array([14.0, 12.24, 7.25]) * u.km / u.s # Sun's orbital velocity vector (U,V,W), [km/s]
	v_LSR = 220. * u.km/ u.s # Circular velocity at Sun's position [km/s]

	# From Josh Thesis:
	mul_app =  ( v_sun[0]*np.sin(l) - (v_sun[1] + v_LSR) * np.cos(l) ) / r
	mub_app =  ( v_sun[0]*np.sin(b)*np.cos(l) + (v_sun[1] + v_LSR)*np.sin(b)*np.sin(l) - v_sun[2]*np.cos(b) ) / r
	vrad_app =  -v_sun[0]*np.cos(b)*np.cos(l) - (v_sun[1] + v_LSR)*np.cos(b)*np.sin(l) - v_sun[2]*np.sin(b)

	return mul_app, mub_app, vrad_app

# ------ Mass_to_Radius -----
# - Taking in input the given star mass, this function gives back its radius according to a mass-radius scaling relation (Demircan & Kahraman 1991).

def Mass_to_Radius(M):

	if M < 1.66 * u.solMass:

		return 1.06 * (M  / u.solMass)**0.945 * u.solRad

	else:

		return 1.33 * (M  / u.solMass)**0.555 * u.solRad

# ------- Distance_Correction -------
# - Taking in input the distance r of the HVS and its radius R, this function computes the
# - distance correction for computing the unreddened flux at Earth [mag]

def Distance_Correction(r,R):

	return (- 2.5 * np.log10(  ((R/r)**2.).to(u.Unit('')) )).value


# ------ log_Surface_Gravity -------
# - Taking in input the Mass and the Radius of the star, this function computes the logarithm of
# - the surface gravity

def log_Surface_Gravity(M,R):

	g = (const.G * M / R**2.).to(u.cm / u.s**2) # surface gravity, cgs units [cm/s^2]

	return np.log10(g.value)

# ------ norm_Power_Law_IMF -----
# - Taking in input the Mass, this function gives back the value of the normalised power-law IMF
# - ~ M^alpha in [M_min, M_max]

def norm_Power_Law_IMF(M, alpha, M_min, M_max):

	A = (1. + alpha) / (M_max**(1.+alpha) - M_min**(1.+alpha))

	return A * M**alpha

# ------ norm_Kroupa_IMF -----
# - Taking in input the Mass, this function gives back the value of the normalised power-law IMF
# - ~ M^alpha in [M_min, M_max]

def norm_Kroupa_IMF(M, M_min, M_max):

	M_t = 0.5

	A = 1. / ( (M_min**(-0.3) - M_t**(-0.3))/0.3 + M_t*(M_t**(-1.3) - M_max**(-1.3))/1.3 )

	if M > M_t:
		return A * M_t * M**(-2.3)
	else:
		return A * M**(-1.3)

# ------ inv_sampling_Power_Law ----
# - Taking in input a uniformly-distributed random number x in [0,1], this function gives
# - back a power-law distributed number n (~ n^alpha) in the range [n_min, n_max]

def inv_sampling_Power_Law(x,alpha,n_min,n_max):

	if alpha == -1:
		return n_min * (n_max/n_min) ** x

	else:
		return ( (n_max**(1.+alpha) - n_min**(1.+alpha))*x + n_min**(1.+alpha) )**(1./(1.+alpha))


# ------ inv_sampling_IMF ----
# - Taking in input a uniformly-distributed random number x in [0,1], this function gives
# - back a mass [in solar mass] according to a power-law IMF with index alpha.

def inv_sampling_IMF(x,alpha):
    mcmin = 0.5
    mcmax = 9.
    return ( (mcmax**(1.+alpha) - mcmin**(1.+alpha))*x + mcmin**(1.+alpha) )**(1./(1.+alpha))


# ------ inv_sampling_Kroupa ----
# - Taking in input a uniformly-distributed random number x in [0,1], this function gives
# - back a mass [in solar mass] according to a kroupa IMF 
# - (Mmin = 0.1, Mmax = 50), alpha = -1.3 for Mmin < M < 0.5, alpha = -2.3 for 0.5 < M < Mmax

def inv_sampling_Kroupa(x):
    Mmin = 0.1
    Mmax = 50.
    Ak = 1. / ( -(0.5**(-0.3) - Mmin**(-0.3))/0.3 - 0.5* (Mmax**(-1.3) - 0.5**(-1.3))/1.3 )
    if x > Ak*(Mmin**(-0.3) - 0.5**(-0.3))/0.3: #equivalent to the condition M < 0.5 Msun
        return (-0.3*x/Ak + Mmin**(-0.3))**(-1./0.3)
    else:
        return ( 0.5**(-1.3) - 1.3*( x/Ak + (0.5**(-0.3) - Mmin**(-0.3))/0.3 ))**(-1./1.3)


# -------- ecliptic ----------
# - Taking in input the Galactic longitude l (in radians) and the Galactic latitude b (in radians),
# - this function returns the ecliptic latitude (in radians), according to Eq. (7) in Jordi et al. 2010.

def ecliptic(l,b):
	return np.arcsin(abs(0.4971*np.sin(b) + 0.8677*np.cos(b)*np.sin(l - 6.38 * u.deg)))


# -------- closest_spectrum --------
# - Taking in input the radius (in RSun), the mass (in MSun) and the effective temperature (in K),
# - this function gives back the spectrum that matches better the given stellar parameters.

def closest_spectrum(Teff,Logg,path):
	
	Met = 0. # Assumption: Considering only Solar Metallicities!
	Vturb = 2.00 # Atmospheric micro-turbulence velocity [km/s]
	XH = 0.00 # Mixing length	

	files, Id, T, logg, met, Vt, Xh = np.loadtxt(path+'BaSeL3.1/WLBC99/BaSeL_library_solar_met.txt', dtype = 'str', unpack=True)
	Id = np.array(Id,dtype='float')
	T = np.array(T,dtype='float')
	logg = np.array(logg,dtype='float')
	met = np.array(met,dtype='float')
	Vt = np.array(Vt, dtype = 'float')
	Xh = np.array(Xh, dtype='float')	

	ds = np.sqrt( (T - Teff)**2. + (logg - Logg)**2. + (Met - met)**2. + (Vturb - Vt)**2. + (Xh - XH)**2. ) 
	indexm = np.where(ds == np.min(ds)) # Chi-square minimization

	identification = Id[indexm]
	
	return identification
	
# ------- get_spectrum -------
# - Taking in input the effective temperature Teff (in K), the logarithm of the surface gravity,
# - the metallicity of the star, its radius (in Rsun) and its distance (in kpc),
# - it returns the flux (in photons/s/m^2/nm) and the correspondent wavelengths (in nm),from the given Basel Spectrum.


#def get_spectrum(Teff, logg, MH, Rstar, dist, Id, path):
def get_spectrum0(Id,Rstar,MH,dist,path):
	
	#INPUT PARAMETERS:
	
	Vturb = 2.00 # Atmospheric micro-turbulence velocity [km/s]
	XH = 0.00 # Mixing length

	#string = "9500  4.00 -0.50  2.00  0.00"    #Teff [K], log g, [M/H], Vturb, XH

        #string = "%i  %.2f %.2f  %.2f  %.2f" % (Teff.value, logg, MH.value, Vturb, XH)
	string = "%.2f" %(MH)

	if float(string) == -2.00:
        	pat = "m20"
	if float(string) == -1.50:
	        pat = "m15"
	if float(string) == -1.00:
	        pat = "m10"
	if float(string) == -0.50:
	        pat = "m05"
	if float(string) == 0.:
	        pat = "p00"
	if float(string) == 0.50:
	        pat = "p05"

	f = open(path+"BaSeL3.1/WLBC99/wlbc99_"+pat+".cor", "r")
	data = f.read().split()
	
	Nlines = 1221 # Number of lines per wavelength and each SED
	Nparameters = 6 # Number of stellar parameters
	
	SEDS = ( len(data) - Nlines ) / ( Nlines + Nparameters) # Number of SEDS per file

	for i in range(0, int(SEDS), 1):
		lower_index = int(Nlines + i*(Nparameters + Nlines) )
		if ( float(data[lower_index]) == int(Id) ):
			flux = np.array( data[lower_index + Nparameters:lower_index + Nparameters + Nlines], dtype='double') # Flux moment, [erg/s/cm/cm/Hz]
	f.close()	

	flux = flux * u.erg / u.s / u.cm**2 / u.Hz

	wlength = np.array(data[0: Nlines], dtype = 'double') * u.nm # Wavelengths, [nm]
	
	Ep = const.h*const.c/wlength          #energy per photon
	
	corr = (Rstar/dist)**2. # Distance correction for computing the unreddened flux at Earth
	
	F_lambda = (4*flux*const.c/wlength/wlength/Ep*corr).to( 1/u.s/u.m**2/u.nm)  #[photon/s/m/m/nm]

	return (wlength,F_lambda)


def get_spectrum(Id,MH,path):
	
	#INPUT PARAMETERS:
	
	Vturb = 2.00 # Atmospheric micro-turbulence velocity [km/s]
	XH = 0.00 # Mixing length

	#string = "9500  4.00 -0.50  2.00  0.00"    #Teff [K], log g, [M/H], Vturb, XH

        #string = "%i  %.2f %.2f  %.2f  %.2f" % (Teff.value, logg, MH.value, Vturb, XH)
	string = "%.2f" %(MH)

	if float(string) == -2.00:
        	pat = "m20"
	if float(string) == -1.50:
	        pat = "m15"
	if float(string) == -1.00:
	        pat = "m10"
	if float(string) == -0.50:
	        pat = "m05"
	if float(string) == 0.:
	        pat = "p00"
	if float(string) == 0.50:
	        pat = "p05"

	f = open(path+"BaSeL3.1/WLBC99/wlbc99_"+pat+".cor", "r")
	data = f.read().split()
	
	Nlines = 1221 # Number of lines per wavelength and each SED
	Nparameters = 6 # Number of stellar parameters
	
	SEDS = ( len(data) - Nlines ) / ( Nlines + Nparameters)# Number of SEDS per file
	

	for i in range(0, int(SEDS), 1):
		lower_index = int(Nlines + i*(Nparameters + Nlines) )
		if ( float(data[lower_index]) == int(Id) ):
			flux = np.array( data[lower_index + Nparameters:lower_index + Nparameters + Nlines], dtype='double') # Flux moment, [erg/s/cm/cm/Hz]
	f.close()	

	flux = flux * u.erg / u.s / u.cm**2 / u.Hz

	wlength = np.array(data[0: Nlines], dtype = 'double') * u.nm # Wavelengths, [nm]
	
	Ep = const.h*const.c/wlength          #energy per photon

	F_lambda = (4*flux*const.c/wlength/wlength/Ep).to( 1/u.s/u.m**2/u.nm)  #[photon/s/m/m/nm]

	return (wlength,F_lambda)

# -------- Magnitudes ----------
# - Taking in input the Flux (in photons/s/m/m/nm), the wavelengths (in nm) and the extinction, this function
# - computes the G, V and IC magnitudes, using the Gaia G passband and the Johnson-Cousins passbands

def Magnitudes(F, wlenf, wlen, A_l, A_l_med_G, A_l_med_V, A_l_med_Ic, TG, TV, TIc, FVega, A_l_med = False):
	
	F = interp1d(wlenf, F) # Interpolating the flux on the passband wavelengths
	F = F(wlen)
	
	GxVega = 0.03 # Zero magntidue for a Vega-like star in the G band
	VVega = 0.030 # Zero magntidue for a Vega-like star in the V band
	IVega = 0.033 # Zero magntidue for a Vega-like star in the Ic band
	
	if A_l_med:
		GMag = -2.5 * np.log10( 10**(-0.4*A_l_med_G) * simps(y = F * TG, x = wlen) / simps(y = FVega * TG, x = wlen) ) + GxVega 
		VMag = -2.5 * np.log10( 10**(-0.4*A_l_med_V) * simps(y = F * TV, x = wlen) / simps(y = FVega * TV, x = wlen) ) + VVega
		IcMag = -2.5 * np.log10( 10**(-0.4*A_l_med_Ic) * simps(y = F * TIc, x = wlen) / simps(y = FVega * TIc, x = wlen)) + IVega
	else:
		GMag = -2.5 * np.log10( simps(y = F * 10.**(-0.4*A_l.value) * TG, x = wlen) / simps(y = FVega * TG, x = wlen) ) + GxVega 
		VMag = -2.5 * np.log10( simps(y = F * 10.**(-0.4*A_l.value) * TV, x = wlen) / simps(y = FVega * TV, x = wlen)) + VVega
		IcMag = -2.5 * np.log10( simps(y = F * 10.**(-0.4*A_l.value) * TIc, x = wlen) / simps(y = FVega * TIc, x = wlen)) + IVega
	
	return GMag, VMag, IcMag

def get_Magnitude(F, wlenf, wlen, A_l, FVega, MVega, T):

        #Interpolates BaSeL spectrum
	F = interp1d(wlenf,F)
        #Gets interpolated spectrum at the same wavelengths as the passband
	F = F(wlen)
	'''	
	T = interp1d(wlen,T)
	T = T(wlen)
	'''

        #Integrates spectrum through passband response (See Jordi+2010, Eq. 1)
	Mag = -2.5 * np.log10( 	simps(y = F * 10**(-0.4*A_l) * T, x = wlen) / simps(y = FVega * T, x = wlen) ) + MVega
	
	return Mag

def get_Magnitude_AB(F, wlenf, wlen, A_l, T):

        #Interpolates BaSeL spectrum
	F = interp1d(wlenf,F)
        #Gets interpolated spectrum at the same wavelengths as the passband
	F = F(wlen)
	'''	
	T = interp1d(wlen,T)
	T = T(wlen)
	'''

        #Integrates spectrum through passband response (See Jordi+2010, Eq. 1)
	Mag = -2.5 * np.log10( 	simps(y = F * 10**(-0.4*A_l) * T, x = wlen) )
	
	return Mag


def G_to_GRVS( G, V_I ): # From Gaia G band magnitude to Gaia G_RVS magnitude

	# C. Jordi et al. , Table 3, second row:
	a = -0.0138
	b = 1.1168
	c = -0.1811
	d = 0.0085

	f = a + b * V_I + c * V_I**2. + d * V_I**3.

	return G - f # G_RVS magnitude
G_to_GRVS = np.vectorize(G_to_GRVS)


# ===== #

def heaviside(x): # Heaviside step function
	return 0.5 * (np.sign(x) + 1)
heaviside = np.vectorize(heaviside)

def Gal_V_element(r,b,dl,db,dr): # Volume element in Galactic coordinates
	return r**2. * np.cos(b) * dl * db * dr # [kpc^3]


def Number_HVS(tMS, R_vec, vF, phi_M, Ndot, ri, b, dl, db, dr, dM):
	
	rho = heaviside( (tMS - 2.*R_vec/vF).value ) * heaviside( (tMS - R_vec/vF).value ) * phi_M * ( Ndot/(4.*np.pi*R_vec**2.*vF) - Ndot/(2.*np.pi*R_vec*tMS*vF**2.) )

	dN = rho * Gal_V_element(ri,b,dl,db,dr) * dM
	N = ( (dN / u.rad**2.).to(u.Unit(''))).value

	return N

# -------- t_MS ------------------
# - Taking in input the mass of the star, this function gives back the MS lifetyme
# - of the star.

def t_MS_easy(M):

	return 10.**(-2.78 * np.log10( (M / u.solMass).value ) + 4.) * u.Myr












