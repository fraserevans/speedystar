
__ImportError__ = "One or more required external packages have not been installed. Please see requirements.txt"

__PunboundAstrometryError__ = 'Computing unbound probabilities requires full equatorial positions and velocities (ra, dec, parallax/distance, pmra, pmdec, vlos). Please make sure your sample includes these attributes.'

__PunboundUncertaintyWarning__ = 'Computing unbound probabilities requires uncertainties on positions and velocities in the equatorial frame. Calculating...'

try:
    from astropy import units as u
    import numpy as np
    from tqdm import tqdm
except ImportError:
    raise ImportError(__ImportError__)

#@photometry
def photometry(self, bands=['Bessell_V', 'Bessell_I', 
                   'Gaia_GRVS', 'Gaia_G', 'Gaia_BP', 'Gaia_RP'],
                   errors = ['e_par', 'e_pmra', 'e_pmdec', 'e_vlos']):

    '''
    Computes mock apparent magnitudes in the Gaia bands (and also others).
    Also calculates mock DR4 astrometric errors using pygaia. 
    These may or may not be overwritten later (see subsample()).

    Parameters
    ----------
    dustmap : DustMap
        Dustmap object to be used
    bands: List of strings
        The photometric bands in which apparent magnitudes are calculated. 
        Names are more or less self-explanatory. Options for now include:
        - Bessell_U, Bessell_B, Bessell_V, Bessell_R, Bessell_I 
          Johnson-Cousins UBVRI filters (Bessell 1990)
        - Gaia_G, Gaia_BP, Gaia_RP, Gaia_GRVS bands
            - NOTE: Only EDR3 bands are currently implemented in MIST. DR3 
              bands are available from Gaia and this code will be updated 
              when DR3 bands are implemented in MIST.
            - NOTE as well: This subroutine calculates G_RVS magnitudes not 
              using the G_RVS transmission curve directly but by a power-law 
              fit using the Bessell_V, Bessell_I and Gaia_G filters 
              (Jordi et al. 2010). Transmission curve was not available prior 
              to Gaia DR3 and is not yet implemented in MIST.
        - VISTA Z, Y, J, H, K_s filters 
        - DECam u, g, r, i, z, Y filters 
        - LSST u, g, r, i, z, y filters
    errors: List of strings
        The Gaia errors to calculate. 
        Fairly inexpensive if you are already calculating Bessell_I, 
        Bessell_V, Gaia_G.
        - Options include:
            - e_par -- DR4 predicted parallax error (mas)
            - e_pmra, e_pmdec -- DR4 predicted proper motion in the 
                    ra (cosdec corrected) and dec directions (mas/yr)
            - e_vlos -- DR4 predicted radial velocity error (km/s)
            - NOTE: errors are computed agnostic of whether or not Gaia 
                could actually detect a given source. Recall that the 
                faint-end magnitude limit of the Gaia astrometric catalogue 
                is G ~ 21 and G_RVS ~ 16.2 for the radial velocity catalogue.
            - NOTE: These error calculations are inexpensive but not the 
                most accurate, particularly for bright sources. 
                Get_Gaia_errors() is slow but more robustly simulates the 
                Gaia astrometric performance 

    '''

    try:
        from galpy.util.coords import radec_to_lb
    except ImportError:
        raise ImportError(__ImportError__)

    from .utils.dustmap import DustMap
    from .utils.MIST_photometry import get_Mags

    if not hasattr(self,'dust'):
        raise ValueError('You must provide a dust map. ' \
            'Please call config_dust() first')

    dustmap = self.dust

    #Needs galactic lat/lon to get G_RVS 
    # converts to it if only equatorial are available
    if(hasattr(self,'ll')):
            l = self.ll
            b = self.bb
    elif(hasattr(self,'ra')):
        data = radec_to_lb(self.ra.to('deg').value, 
                                self.dec.to('deg').value, degree=True)
        l, b = data[:, 0], data[:, 1]
    else:
        raise ValueError('RA/Dec or Galactic lat/lon are required to perform'\
                                'mock photometry. Please check your sample.')           
    if not hasattr(self,'Av'):
        self.Av = None

    if(self.size==0):
        self.Av, self.e_par, self.e_pmra, self.e_pmdec, self.e_vlos \
            = ([] for i in range(5))
        for band in bands:
            setattr(self,band,[])
    else:
        # Calculates visual extinction, apparent magnitudes in 
        # appropriate bands, and errors 
        self.Av, Mags, errs = get_Mags(self.Av, self.dist.to('kpc').value, 
                                      l, b, self.m.to('Msun').value, self.met, 
                                      self.T_eff.value, self.Rad.value, 
                                      self.Lum.value, dustmap, bands, errors)

        #Sets attributes
        for band in bands:
            setattr(self, band, Mags[band])
        for err in errors:
            setattr(self, err, errs[err]) 
        if(hasattr(self,'e_par')):         
            self.e_par = self.e_par * u.mas
        if(hasattr(self,'e_pmra')):         
            self.e_pmra = self.e_pmra * u.mas / u.yr
        if(hasattr(self,'e_pmdec')):         
            self.e_pmdec = self.e_pmdec * u.mas /u.yr
        if(hasattr(self,'e_vlos')):         
            self.e_vlos = self.e_vlos * u.km/u.s

#@get_Punbound
def get_Punbound(self, numsamp = int(5e1), 
                    par_cut_flag=True, par_cut_val = 0.2):

    '''
    Sampling over provided observations w/ errors, returns probability 
    that star is unbound in the provided Galactic potential.

    Parameters:
    ---------------

    covmat : 5x5xself.size array
        Gaia covariance matrix, likely generated by scanninglaw.asf unless 
        you're generating them yourself (would not recommend). 
        Construction of covariance matrix is as follows:
            - [0,0,:] -- RA variances (mas^2)
            - [1,1,:] -- Dec variances (mas^2)
            - [2,2,:] -- parallax variances (mas^2)
            - [3,3,:] -- pm_ra_cosdec variances (mas^2 yr^-2)
            - [4,4,:] -- pm_dec variances (mas^2 yr^-2)
            - off-diagonals -- correlations among errors

    numsamp : integer
        Number of times observations should be sampled to 
        compute unbound probabilities

    par_cut_flag : Boolean
        If True, computes only unbound probabilities for sources with 
        (relative) parallax uncertaintainties less than par_cut_val. 
        Recommended to keep as true -- unbound probabilities are not 
        particularly meaningful for sources with large distance 
        uncertainties and the computation of these probabilities can take a
        long time for populations for whom this cut is not performed.

    par_cut_val : real
        The if par_cut_flag is True, the relative parallax error cut 
        to impose. Default is 20% and it is recommended to keep it here. 
        20% is where estimating distances by inverting parallaxes starts 
        to become problematic -- see Bailer-Jones 2015 
        (https://ui.adsabs.harvard.edu/abs/2015PASP..127..994B)

    '''

    try:
        import astropy.coordinates as coord
        from astropy.table import Table
        from galpy.util.coords import radec_to_lb, pmrapmdec_to_pmllpmbb
        from galpy.potential import evaluatePotentials
        from galpy.potential.mwpotentials import McMillan17, MWPotential2014
    except ImportError:
        raise ImportError(__ImportError__)

    from .utils.dustmap import DustMap

    #Check to make sure astrometry exists
    if not ( hasattr(self,'ra') and hasattr(self,'dec') \
            and hasattr(self,'pmra') and hasattr(self,'pmdec') \
            and hasattr(self,'dist') and hasattr(self,'vlos')):
        raise ValueError(__PunboundAstrometryError__)

    #Check to see if errors exist. Compute them if not
    if not hasattr(self,'cov'):
        if not ( hasattr(self,'e_par') and hasattr(self,'e_pmra') and \
                hasattr(self,'e_pmdec') and hasattr(self,'e_vlos') ):
            print(__PunboundUncertaintyWarning__)
            self.photometry(self.dust)

    #Cut stars with parallax errors above par_cut_val
    if par_cut_flag:
        idx = (self.e_par.to(u.mas)/self.par<=par_cut_val)
        self.subsample(np.where(idx)[0])

    if(self.size==0):
        self.Pub = np.zeros(self.size)
        return

    #Solar position and velocity in McMillan17
    vo               = 233.1 
    ro               = 8.21*u.kpc
    Usun, Vsun, Wsun = 8.6, 13.9, 7.1

    #Initialize...
    self.Pub = np.empty(self.size)
    self.e_GCv = np.empty(self.size)*u.km/u.s

    vSun = [-Usun, Vsun, Wsun] * u.km / u.s # (U, V, W)
        
    v_sun = coord.CartesianDifferential(vSun + [0, vo, 0]*u.km/u.s)

    GCCS = coord.Galactocentric(galcen_distance=ro, z_sun=0*u.kpc, 
                                    galcen_v_sun=v_sun)

    print('Computing P_unbound...')
    for i in tqdm(range(self.size)):

        #Don't even calculate if star is very fast
        if(self.v0[i]>1500*u.km/u.s):
            self.Pub[i] = 1.
            self.e_GCv[i] = 0.
            
        else:
            #Sample a radial velocity
            vlos = np.random.normal(self.vlos[i].value, 
                                    self.e_vlos[i].value,numsamp)*u.km/u.s

            #Get the 'true' astrometry
            means = [self.ra[i].to('mas').value,self.dec[i].to('mas').value, 
                    self.par[i].value,self.pmra[i].to(u.mas/u.yr).value, 
                    self.pmdec[i].to(u.mas/u.yr).value
                    ]

            if hasattr(self, 'cov'):

                # Sample astromerry n times based on covariance matrix
                ratmp, dectmp, partmp, pmra, pmdec = \
                    np.random.multivariate_normal(means,self.cov[:,:,i],
                                                        numsamp).T

                ra = ratmp*u.mas.to(u.deg)*u.deg
                dec = dectmp*u.mas.to(u.deg)*u.deg
                dist = u.kpc/np.abs(partmp)
                pmra, pmdec = pmra*u.mas/u.yr, pmdec*u.mas/u.yr

            else:
                # Sample astrometry based only on errors.
                # Errors are assumed to be uncorrelated
                ra = self.ra[i]*np.ones(numsamp)
                dec = self.dec[i]*np.ones(numsamp)

                dist = u.kpc / abs(np.random.normal(self.par[i].value, 
                                self.e_par[i].to(u.mas).value,numsamp))

                pmra = np.random.normal(self.pmra[i].to(u.mas/u.yr).value, 
                       self.e_pmra[i].to(u.mas/u.yr).value,numsamp)*u.mas/u.yr
                pmdec=np.random.normal(self.pmdec[i].to(u.mas/u.yr).value, 
                      self.e_pmdec[i].to(u.mas/u.yr).value,numsamp)*u.mas/u.yr

            #Get galactocrentric position and velocity 
            data   = radec_to_lb(ra.value, dec.value, degree=True)
            ll, bb = data[:, 0], data[:, 1]

            data       = pmrapmdec_to_pmllpmbb(pmra,pmdec, ra.value, 
                                                    dec.value, degree=True)
            pmll, pmbb = data[:, 0], data[:, 1]

            galactic_coords = coord.Galactic(l=ll*u.deg, b=bb*u.deg, 
                                    distance=dist, pm_l_cosb=pmll*u.mas/u.yr, 
                                    pm_b=pmbb*u.mas/u.yr, radial_velocity=vlos)

            galactocentric_coords = galactic_coords.transform_to(GCCS)

            GCv = np.sqrt(galactocentric_coords.v_x**2. 
                            + galactocentric_coords.v_y**2. 
                            + galactocentric_coords.v_z**2.).to(u.km/u.s)

            R = np.sqrt(galactocentric_coords.x**2 
                            + galactocentric_coords.y**2).to(u.kpc)

            z = galactocentric_coords.z.to(u.kpc)

            #For each sampled entry, get escape velocity
            Vesc = np.zeros(numsamp)*u.km/u.s
            for j in range(numsamp):
                Vesc[j] = np.sqrt(2*(- evaluatePotentials(McMillan17,R[j],z[j])))#*u.km/u.s
                #Vesc[j] = np.sqrt(2*(- evaluatePotentials(MWPotential2014,R[j],z[j])))*u.km/u.s
                #Vesc[j] = np.sqrt(2*(evaluatePotentials(MWPotential2014,1e10*R[j],z[j]) - evaluatePotentials(MWPotential2014,R[j],z[j])))*220*u.km/u.s

            #Calculate fraction of iterations above escape velocity
            inds = (GCv > Vesc)
            self.Pub[i] = len(GCv[inds])/len(Vesc)

            #Calculate spread of sampled galactocentric velocity
            self.e_GCv[i] = np.std(GCv)
