_Gaia_releases = ['DR2', 'EDR3', 'DR3', 'DR4', 'DR5']
_Gaia_errors = ['e_ra', 'e_dec', 'e_par', 'e_pmra', 'e_pmdec', 'e_vlos']

from astropy import units as u
import numpy as np
from tqdm import tqdm
from astropy import constants as const
import os
from galpy.util.conversion import get_physical

def _check_rovozoso(self, ro, vo, zo, solarmotion):
    '''
    Checks to see if the provided ro, vo, zo and solarmotion values are sensible

    Parameters
    ----------
    ro: float, astropy distance quantity
        The distance from the Sun to the Galactic center
    vo: float, astropy velocity quantity
        The circular velocity of the Sun
    zo: float, astropy distance quantity
        The height of the Sun above the Galactic plane
    solarmotion: list of floats, astropy velocity quantities
        The solar motion in the U, V, W directions
    '''

    #Assign vo, ro, zo, solarmotion
    if vo is None:
        if (self.vo is None) or not hasattr(self,'vo'):
            print('Warning: vo not provided. Defaulting to value in galpy config file')
            self.vo = get_physical(MWPotential2014)['vo']*u.km/u.s
    else:
        self.vo = vo

    if ro is None:
        if (self.ro is None) or not hasattr(self,'ro'):
            print('Warning: ro not provided. Defaulting to value in galpy config file.')
            self.ro = get_physical(MWPotential2014)['ro']*u.kpc
    else:
        self.ro = ro

    if zo is None:
        if (self.zo is None) or not hasattr(self,'zo'):
            print('Warning: zo not provided. Defaulting to 20.8 pc (Bovy & Bennett 2019)')
            self.zo = 0.0208*u.kpc
    else:
        selfzo = zo

    if solarmotion is None:
        if not hasattr(self,'solarmotion') or (self.solarmotion is None):
            print('Warning: UVW Solar motion not provided. Defaulting to [-11.1, 12.24, 7.25]*u.km/u.s (Schonrich+2010)')
            self.solarmotion = [-11.1, 12.24, 7.25]*u.km/u.s
    else:
        self.solarmotion = solarmotion

def _sample_errors(self, index=0, numsamp=100):
    '''
    Given real positions and velocities, sample the astrometric and radial velocity errors numsamp times of the i'th star in the sample

    Parameters
    ----------

    index: int
        The index of the star in the sample to sample from. Default is 0.
    numsamp: int
        The number of samples to take. Default is 100.

    Returns
    -------
    o: galpy Orbit object
        The sampled positions and velocities in a length-numsamp array
    '''

    from galpy.orbit import Orbit

    #Sample a radial velocity
    vlos = np.random.normal(self.vlos[index].value, 
                            self.e_vlos[index].value,numsamp)*u.km/u.s

    #Get the 'true' astrometry
    means = [self.ra[index].to('mas').value,self.dec[index].to('mas').value, 
            self.par[index].value,self.pmra[index].to(u.mas/u.yr).value, 
            self.pmdec[index].to(u.mas/u.yr).value
            ]

    if hasattr(self, 'cov'):

        # Sample astrometry n times based on covariance matrix
        ratmp, dectmp, partmp, pmra, pmdec = \
            np.random.multivariate_normal(means,self.cov[:,:,index],
                                                numsamp).T

        ra = ratmp*u.mas.to(u.deg)*u.deg
        dec = dectmp*u.mas.to(u.deg)*u.deg
        dist = u.kpc/np.abs(partmp)
        pmra, pmdec = pmra*u.mas/u.yr, pmdec*u.mas/u.yr

    else:

        # Sample astrometry based only on errors.
        # Errors are assumed to be uncorrelated
        ra = self.ra[index]*np.ones(numsamp)
        dec = self.dec[index]*np.ones(numsamp)

        dist = u.kpc / abs(np.random.normal(self.par[index].value, 
                        self.e_par[index].to(u.mas).value,numsamp))

        pmra = np.random.normal(self.pmra[index].to(u.mas/u.yr).value, 
                self.e_pmra[index].to(u.mas/u.yr).value,numsamp)*u.mas/u.yr
        pmdec=np.random.normal(self.pmdec[index].to(u.mas/u.yr).value, 
                self.e_pmdec[index].to(u.mas/u.yr).value,numsamp)*u.mas/u.yr

    o = Orbit([ra, dec, dist, pmra, pmdec, vlos],radec=True, ro=self.ro, 
            solarmotion=self.solarmotion, vo=self.vo, zo=self.zo)

    return o

def get_Gaia_errors_old(self, errors = ['e_par', 'e_pmra', 
                                    'e_pmdec', 'e_vlos']):

    global dr3sf, dr3astsf, dr3rvssf, dr3rvssfvar
    global dr2sf, dr2astsf, dr2rvssf, dr2rvssfvar, dr2rvssf2

    from .utils.selectionfunctions.source import Source as rSource
    from .utils.MIST_photometry import get_e_vlos_old

    '''
    DEPRECIATED. Calculates mock Gaia astrometric and radial velocity errors depending on the user-specified data release.

    Parameters
    ----------
    errors: List of strings
        The Gaia errors to calculate. 
        - Options include:
            - e_ra -- Error in right ascension (uas)
            - e_dec -- Error in declination (uas)
            - e_par -- parallax error (mas)
            - e_pmra, e_pmdec -- predicted proper motion in the 
                    ra (cosdec corrected) and dec directions (mas/yr)
            - e_vlos -- Radial velocity error (km/s)
            - NOTE: errors are computed agnostic of whether or not Gaia 
                could actually detect a given source. Recall that the 
                faint-end magnitude limit of the Gaia astrometric catalogue 
                is G ~ 21 and G_RVS ~ 16.2 for the radial velocity catalogue.

    '''

    #Check to make sure the requested errors are valid options
    if len([i for i in errors if i not in _Gaia_errors]) > 0:
        print(ErrorWarning)

    errors = [err for err in errors if err in _Gaia_errors]

    #Check to make sure 

    _position_scale = {'DR5': 0.7, 'DR4': 1.0, 'DR3': 1.335, 
                                'EDR3': 1.335, 'DR2': 1.7}
    _propermotion_scale = {'DR5': 0.7*0.5, 'DR4': 1.0, 'DR3': 1.335*1.78, 
                                'EDR3': 1.335*1.78, 'DR2': 4.5}
    _vrad_scale = {'DR5': 0.707, 'DR4': 1.0, 'DR3': 1.33 , 
                                'EDR3': 1.65, 'DR2': 1.65}

    if self.size>0:
        if self.use_ast_sf:
            from scanninglaw.source import Source as aSource
            import scanninglaw.asf as astrospreadfunc
            if 'dr2astsf' not in globals():
                #Load in DR2 astrometric spread function
                dr2astsf = astrospreadfunc.asf() 

            if 'dr3astsf' not in globals():
                #Load in DR3 astrometric spread function
                dr3astsf = astrospreadfunc.asf(version='dr3_nominal') 

            if not ( hasattr(self,'Gaia_G') ):
                print(asfWarning)
                self.photometry(bands=['Gaia_G'],errors=errors)

            if not ( hasattr(self,'ra') and hasattr(self,'dec') ):
                raise ValueError(asfError)

            _which_ast_sf = {'DR2': dr2astsf, 'EDR3': dr3astsf, 
                            'DR3':dr3astsf, 'DR4': dr3astsf, 'DR5': dr3astsf}

            #Position and brightness of each star
            asource = aSource(self.ra,self.dec,frame='icrs',
                                photometry={'gaia_g':self.Gaia_G})

            #Calling the astrometric spread function. 
            # Gives the 5x5 covariance matrix, whose diagonal
            # elements are the variances of ra/dec/par/pmra/pmdec
            #self.cov = dr2astsf(asource)
            self.cov = _which_ast_sf[self.gaia_release](asource)

            if self.gaia_release == 'DR4':
                self.cov[0,0,:] /= _position_scale['DR3']
                self.cov[1,1,:] /= _position_scale['DR3']
                self.cov[2,2,:] /= _position_scale['DR3']
                self.cov[3,3,:] /= _propermotion_scale['DR3']
                self.cov[4,4,:] /= _propermotion_scale['DR3']
            elif self.gaia_release == 'DR5':
                self.cov[0,0,:] /= ( _position_scale['DR3']
                                    / _position_scale['DR5'])
                self.cov[1,1,:] /= ( _position_scale['DR3']
                                    / _position_scale['DR5'])
                self.cov[2,2,:] /= ( _position_scale['DR3']
                                    / _position_scale['DR5'])
                self.cov[3,3,:] /= ( _propermotion_scale['DR3']
                                    / _propermotion_scale['DR5'])
                self.cov[4,4,:] /= ( _propermotion_scale['DR3'] 
                                    / _propermotion_scale['DR5'] )

            #print(self.cov[:,:,0])
            #assign astrometric error
            self.e_ra = np.sqrt(self.cov[0,0])*u.uas
            #print(self.e_ra[0])
            self.e_dec = np.sqrt(self.cov[1,1])*u.uas
            self.e_par   = np.sqrt(self.cov[2,2])*u.mas
            self.e_pmra  = np.sqrt(self.cov[3,3])*u.mas/u.yr
            self.e_pmdec = np.sqrt(self.cov[4,4])*u.mas/u.yr 

            if hasattr(self,'e_vlos'):
                
                self.e_vlos *= _vrad_scale[self.gaia_release]
            elif 'e_vlos' in errors:
                
                if not hasattr(self,'T_eff'):
                    raise ValueError(vlosError)
                if hasattr(self, 'Bessell_V'):
                    self.e_vlos = get_e_vlos_old(self.Bessell_V, \
                                                    self.T_eff.value)
                else:
                    self.photometry(bands=['Bessell_V'], errors=['e_vlos'])
        else:
            
            _which_error = {'e_ra': _position_scale, 'e_dec': _position_scale, 
                        'e_par': _position_scale, 
                        'e_pmra': _propermotion_scale, 
                        'e_pmdec': _propermotion_scale, 
                        'e_vlos': _vrad_scale}
 
            #for err in errors:
            #    print(err)
            #    print(hasattr(self,err))

            if all(hasattr(self,err) for err in errors):       
                
                for err in errors:
                    setattr(self, err, getattr(self,err) 
                                    * _which_error[err][self.gaia_release])
            else:
                self.photometry(errors=[err for err in errors 
                                    if err not in ['e_ra', 'e_dec']]) 

def get_Gaia_errors(self, release=None, errors = ['e_par', 'e_pmra',
                        'e_pmdec', 'e_vlos'], use_ast_sf=False, 
                        get_correlations=False):
    '''
    Calculates mock Gaia astrometric and radial velocity errors depending on
    the user-specified data release. 

    Parameters
    ----------
    release: string
        The Gaia data release to use. Options are DR2, EDR3, DR3, DR4, DR5.
    errors: List of strings
        The Gaia errors to calculate. 
        - Options include:
            - e_ra -- Error in right ascension (uas)
            - e_dec -- Error in declination (uas)
            - e_par -- parallax error (mas)
            - e_pmra, e_pmdec -- predicted proper motion in the 
                    ra (cosdec corrected) and dec directions (mas/yr)
            - e_vlos -- Radial velocity error (km/s)
            - NOTE: errors are computed agnostic of whether or not Gaia 
                could actually detect a given source. Recall that the 
                faint-end magnitude limit of the Gaia astrometric catalogue 
                is G ~ 21 and G_RVS ~ 16.2 for the radial velocity catalogue.
    use_ast_sf: bool
        Whether to use the astrometric spread function (ASF) to calculate 
        astrometric errors. If False, uses the PyGaia package to calculate 
        astrometric errors. Default is False. See my_gaia_selection.ipynb.
    get_correlations: bool
        Whether to calculate the correlations between astrometric errors if
        use_ast_sf is True.
    '''

    global dr3sf, dr3astsf, dr3rvssf, dr3rvssfvar
    global dr2sf, dr2astsf, dr2rvssf, dr2rvssfvar, dr2rvssf2

    from .utils.selectionfunctions.source import Source as rSource
    from .utils.vrad_errs import radial_velocity_uncertainty

    if release is None:
        if hasattr(self,'gaia_release'):
            release = self.gaia_release
        else:
            raise ValueError('Error: invalid Gaia data release. Options are DR2, EDR3, DR3, DR4, DR5. See speedystar.observation.get_Gaia_errors() docstring')

    release = release.upper() 

    #Check to make sure supplied data release is valid option
    if release not in _Gaia_releases:
        raise ValueError('Error: invalid Gaia data release. Options are DR2, EDR3, DR3, DR4, DR5. See speedystar.observation.get_Gaia_errors() docstring')

    #Check to make sure the requested errors are valid options
    if len([i for i in errors if i not in _Gaia_errors]) > 0:
        print('Warning: One or more requested error component not yet implemented. Only available error options are e_ra, e_dec, e_par, e_pmra, e_pmdec, e_vlos. See speedystar.observation.get_Gaia_errors() docstring')

    errors = [err for err in errors if err in _Gaia_errors]

    _position_scale = {'DR5': 0.7, 'DR4': 1.0, 'DR3': 1.335, 
                                'EDR3': 1.335, 'DR2': 1.7}
    _propermotion_scale = {'DR5': 0.7*0.5, 'DR4': 1.0, 'DR3': 1.335*1.78, 
                                'EDR3': 1.335*1.78, 'DR2': 4.5}
    _vrad_scale = {'DR5': 0.707, 'DR4': 1.0, 'DR3': 1.33 , 
                                'EDR3': 1.65, 'DR2': 1.65}

    _which_error = {'e_ra': _position_scale, 'e_dec': _position_scale, 
                    'e_par': _position_scale, 
                    'e_pmra': _propermotion_scale, 
                    'e_pmdec': _propermotion_scale, 
                    'e_vlos': _vrad_scale}

    if self.size == 0:
        print("Warning: Sample already consists of zero stars. Adding error attributes anyway.")
        return

    if any(err in ['e_ra', 'e_dec', 'e_par', 'e_pmra', 'e_pmdec'] 
                for err in errors):
        if not hasattr(self,'Gaia_G'):
            print('Warning: Gaia G band apparent magnitude must be known to compute Gaia astrometric errors. Calculating...')
            self.photometry(bands=['Gaia_G'])

    if 'e_vlos' in errors:
        if not hasattr(self,'T_eff'):
                    raise ValueError('Error: Effective temperature attribute is required to compute radial velocity error. Please see speedystar.utils.evolve.evolve for more information.')
        if not hasattr(self, 'Gaia_GRVS'):
            print('Warning: Gaia G_RVS band apparent magnitude must be known to compute radial velocity errors. Calculating...')
            self.photometry(bands=['Gaia_GRVS'])

    if use_ast_sf:

        #record in the metadata that the ast spread function is being used
        self.use_ast_sf = True

        from scanninglaw.source import Source as aSource
        import scanninglaw.asf as astrospreadfunc
        if 'dr2astsf' not in globals():
            #Load in DR2 astrometric spread function
            dr2astsf = astrospreadfunc.asf() 

        if 'dr3astsf' not in globals():
            #Load in DR3 astrometric spread function
            dr3astsf = astrospreadfunc.asf(version='dr3_nominal') 

        if not ( hasattr(self,'ra') and hasattr(self,'dec') ):
            raise ValueError(asfError)

        _which_ast_sf = {'DR2': dr2astsf, 'EDR3': dr3astsf, 
                        'DR3':dr3astsf, 'DR4': dr3astsf, 'DR5': dr3astsf}

        #Position and brightness of each star
        asource = aSource(self.ra,self.dec,frame='icrs',
                            photometry={'gaia_g':self.Gaia_G})

        #Calling the astrometric spread function. 
        # Gives the 5x5 covariance matrix, whose diagonal
        # elements are the variances of ra/dec/par/pmra/pmdec
        #print(asource)
        self.cov = _which_ast_sf[release](asource)

        if release == 'DR4':
            self.cov[0,0,:] /= _position_scale['DR3']
            self.cov[1,1,:] /= _position_scale['DR3']
            self.cov[2,2,:] /= _position_scale['DR3']
            self.cov[3,3,:] /= _propermotion_scale['DR3']
            self.cov[4,4,:] /= _propermotion_scale['DR3']
        elif release == 'DR5':
            self.cov[0,0,:] /= ( _position_scale['DR3']
                                / _position_scale['DR5'])
            self.cov[1,1,:] /= ( _position_scale['DR3']
                                / _position_scale['DR5'])
            self.cov[2,2,:] /= ( _position_scale['DR3']
                                / _position_scale['DR5'])
            self.cov[3,3,:] /= ( _propermotion_scale['DR3']
                                / _propermotion_scale['DR5'])
            self.cov[4,4,:] /= ( _propermotion_scale['DR3'] 
                                / _propermotion_scale['DR5'] )

        #assign astrometric error
        self.e_ra = np.sqrt(self.cov[0,0])*u.uas
        self.e_dec = np.sqrt(self.cov[1,1])*u.uas
        self.e_par   = np.sqrt(self.cov[2,2])*u.mas
        self.e_pmra  = np.sqrt(self.cov[3,3])*u.mas/u.yr
        self.e_pmdec = np.sqrt(self.cov[4,4])*u.mas/u.yr 

        # Calculate correlations if requested
        if get_correlations:
            self.corr_ra_dec = self.cov[0,1] / (self.e_ra * self.e_dec)
            self.corr_ra_par = self.cov[0,2] / (self.e_ra * self.e_par)
            self.corr_ra_pmra = self.cov[0,3] / (self.e_ra * self.e_pmra)
            self.corr_ra_pmdec = self.cov[0,4] / (self.e_ra * self.e_pmdec)
            self.corr_dec_par = self.cov[1,2] / (self.e_dec * self.e_par)
            self.corr_dec_pmra = self.cov[1,3] / (self.e_dec * self.e_pmra)
            self.corr_dec_pmdec = self.cov[1,4] / (self.e_dec * self.e_pmdec)
            self.corr_par_pmra = self.cov[2,3] / (self.e_par * self.e_pmra)
            self.corr_par_pmdec = self.cov[2,4] / (self.e_par * self.e_pmdec)
            self.corr_pmra_pmdec = self.cov[3,4] / (self.e_pmra * self.e_pmdec)

        if 'e_vlos' in errors:
            # Log of surface gravity in cgs
            logg = np.log10((const.G * self.m / 
                            self.Rad**2.).to(u.cm / u.s**2).value) 
            self.e_vlos = radial_velocity_uncertainty(self.Gaia_GRVS, \
                                                self.T_eff, logg) * u.km/u.s
            self.e_vlos *= _vrad_scale[release]
    else:

        #record in the metadata that the ast spread function is NOT being used
        self.use_ast_sf = False
        
        # ============== Errors! ================== #
        from pygaia.errors.astrometric import proper_motion_uncertainty
        from pygaia.errors.astrometric import parallax_uncertainty
        from pygaia.errors.astrometric import position_uncertainty
        #from pygaia.errors.spectroscopic import radial_velocity_uncertainty

        #Calculate astrometric and radial velocity errors. May require 
        #calculation of apparent magnitudes in other bands if they aren't 
        #already available

        errs = {}
        # Parallax error (PyGaia) [mas]

        if 'e_par' in errors:        
            errs['e_par'] = (parallax_uncertainty(self.Gaia_G)/1000) * (u.mas)

        if 'e_ra' in errors:
            poserrs = position_uncertainty(self.Gaia_G)
            errs['e_ra'] = (poserrs[0]/1000) * (u.mas)

        if 'e_dec' in errors:
            if 'poserrs' not in locals():
                poserrs = position_uncertainty(self.Gaia_G)
            errs['e_dec'] = (poserrs[1]/1000) * (u.mas)

        # ICRS proper motion errors (PyGaia) [mas/yr]
        if 'e_pmra' in errors:
            pmerrs = proper_motion_uncertainty(self.Gaia_G)
            errs['e_pmra'] = (pmerrs[0]/1000) * (u.mas/u.yr)

        if 'e_pmdec' in errors:
            if 'pmerrs' not in locals():
                pmerrs = proper_motion_uncertainty(self.Gaia_G)/1000
            errs['e_pmdec'] = (pmerrs[1]/1000) * (u.mas/u.yr)

        # heliocentric radial velocity error [km/s]
        if 'e_vlos' in errors:
            logg = np.log10((const.G * self.m / 
                            self.Rad**2.).to(u.cm / u.s**2).value) 
            errs['e_vlos'] = radial_velocity_uncertainty(self.Gaia_GRVS, \
                                                self.T_eff, logg) * u.km/u.s

        for err in errors:
            setattr(self, err, errs[err] \
                    * _which_error[err][release])

    self.gaia_release = release   

#@photometry
def photometry(self, bands=['Bessell_V', 'Bessell_I', \
                   'Gaia_GRVS', 'Gaia_G', 'Gaia_BP', 'Gaia_RP'], method=None):

    '''
    Computes mock apparent magnitudes in the Gaia bands (and also others).

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
    method : None or string
        The method to use for calculating the mock photometry. Options are
        None, 'MIST' and 'Brutus'. If None, searches for a photometry_method attribute. If one can't be found, defaults to 'MIST'. 'MIST' uses the MIST isochrones to calculate the photometry, but is constrained to a small metallicity range of [-0.25, +0.25]. 'Brutus' also uses the MIST isochrones but interpolates them more intelligently. It is slower though.
    '''

    from galpy.util.coords import radec_to_lb
    from .utils.MIST_photometry import get_Mags
    import mwdust
    import sys

    if not hasattr(self,'dust'):
        print('Warning: A dust map is not provided. Attempting to default '\
                'to Combined15')
        try:
            self.dust = mwdust.Combined15()
        except ValueError:
            raise ValueError('Default dust map could not be loaded. See'\
                    'myexample.py ' \
                    'or https://github.com/jobovy/mwdust for more information.' \
                    ' Call speedystar.config.fetch_dust() to download dust ' \
                    'map(s) and set DUST_DIR environment variable.')

    if (not hasattr(self,'propagated') or not self.propagated):
        print('Warning: sample appears not to have been propagated. Proceeding with mock photometry regardless')

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

    if not all(hasattr(self,attr) 
            for attr in ['dist', 'm', 'met', 'T_eff', 'Rad', 'Lum']):
        raise ValueError('All of the following attributes are required as astropy quantities to compute mock photometry for the sample: distance "dist", mass "m", metallicity "met", effective temperate "T_eff", stellar radius "Rad", luminosity "Lum". Please ensure all these attributes exist.')

    if(self.size==0):
        self.Av, self.e_par, self.e_pmra, self.e_pmdec, self.e_vlos \
            = ([] for i in range(5))
        for band in bands:
            setattr(self,band,[])
    else:

        if method is None:
            if hasattr(self,'photometry_method'):
                method = self.photometry_method
            else:
                method = 'MIST'

        #self.photometry_method = method

        if not str(method):
            raise ValueError('Method must be either "MIST" or "Brutus". See speedystar.observation.photometry for more information.')

        elif method == 'MIST':
            # Calculates visual extinction, apparent magnitudes in 
            # appropriate bands
            self.Av, Mags = get_Mags(self.Av, self.dist.to('kpc').value, l,
                                       b, self.m.to('Msun').value, self.met,
                                       self.T_eff.value, self.Rad.value, 
                                       self.Lum.value, self.dust, bands)

            #Sets attributes
            for band in bands:
                setattr(self, band, Mags[band]*u.dimensionless_unscaled)

        elif method.upper() == 'BRUTUS':
            #Calculates extinction and apparent magnitudes using Brutus. See Examples/my_example_Brutus.ipynb notebook on the github for more information/.

            from brutus import filters
            from brutus.seds import SEDmaker
       
            global brutussed

            filts = np.array(filters.FILTERS)
            filts[0] = 'Gaia_G'
            filts[1] = 'Gaia_BP'
            filts[2] = 'Gaia_RP'        
        
            if 'brutussed' not in globals():

                if os.getenv('BRUTUS_TRACKS') is None or os.getenv('BRUTUS_NNS') is None:
                    sys.stdout.write('Environment variables BRUTUS_TRACKS and BRUTUS_NNS must be set to the paths of the Brutus tracks and NNs files. Please set them by calling speedystar.config_brutus(path_to_files_directory) or type the relative or absolute path here:')

                    self.config_brutus(path=input())

                brutussed = SEDmaker(mistfile=os.getenv('BRUTUS_TRACKS'),
                                    nnfile=os.getenv('BRUTUS_NNS'))

            self.Av = self.dust(l, b, self.dist.to(u.kpc).value) * 2.682

            mags = np.zeros( (self.size, len(filters.FILTERS)) )

            self.eep = -99.*np.ones(self.size)
            for i in tqdm(range(self.size), desc='Calculating magnitudes'):
    
                eep = brutussed.get_eep(loga = np.log10(self.tage[i].to('yr').value), eep = 300., mini = self.m[i].value, feh = self.met[i])
                
                mags[i,] = brutussed.get_sed(mini = self.m[i].value, eep=eep, feh=self.met[i], afe=0., av=self.Av[i], dist=self.dist[i].to('pc').value)[0]

                self.eep[i] = eep

            #Sets attributes
            for band in bands:
                mag = mags[:,np.where(filts==band)[0]]
                mag = mag.flatten()
                setattr(self, band, mag*u.dimensionless_unscaled)
        else:
            raise ValueError('Method must be either "MIST" or "Brutus". See speedystar.observation.photometry for more information.')

def zero_point(self):

    '''
        Calculate the predicted Gaia zero point offset for each mock HVS.
        NOT playtested or validated, proceed with caution

    '''
    import astropy.coordinates as coord
    from astropy.coordinates import SkyCoord
    from zero_point import zpt
    
    if not(hasattr(self,'Gaia_G') and hasattr(self,'Gaia_BP') \
                and hasattr(self,'Gaia_RP')):
        self.photometry(bands=['Gaia_G', 'Gaia_BP', 'Gaia_RP'])

    #Get the astrometric nu_eff (Lindegren et al. 2021 Eq. 3 (A&A, 649, A2))
    nu_eff = (1.76 - (1.61/np.pi)*np.arctan(0.531*(self.Gaia_BP \
                - self.Gaia_RP)))

    zpt.load_tables()

    #Get the ecliptic latitude of each source
    c = SkyCoord(ra=self.ra, dec=self.dec, frame='icrs')
    b = c.transform_to('barycentricmeanecliptic').lat.value

    #Estimate the parallax zero point offset
    self.zp = zpt.get_zpt(self.Gaia_G, nu_eff, -99*np.ones(len(b)), 
                            b, 31*np.ones(len(b)))

#@get_Punbound
def get_Punbound(self, potential, numsamp = int(5e1), par_cut_flag=True, 
                    par_cut_val = 0.2,solarmotion = None, 
                    zo=None, vo=None, ro=None, t = 0.*u.Myr):

    '''
    Sampling over provided observations w/ errors, returns probability 
    that star is unbound in the provided Galactic potential.

    Parameters:
    ---------------

    potential : galpy potential instance
        The assumed Galactic potential. MUST be either defined with physical
        units or `physicalized' with .turn_physical_on()

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

    solarmotion : length-3 list of floats
            UVW Solar peculiar velocity in km/s. 
            Galpy likes the U to be sign-flipped. Default is Schonrich+2010

    zo : Float or astropy distance quantity
             Offset of the Sun above or below the Galactic plane.
             Default is 20.8 pc (Bennett+Bovy 2019)
             If float, units are assumed to be kpc

    ro : Float (astropy length quantity)
        Distance to the Galactic Centre in kpc. Default is None, in which 
        case ro is set to the ro assumed by the provided potential

    vo : Float (astropy velocity quantity)
        Circular velocity at the Solar position in km/s. Default is None, 
        in which case vo is set to the ro assumed by the provided potential

    t : Float (astropy time quantity)
        Time at which the potential is evaluated.
        Only relevant if potential is time-dependent
    '''
        
    import astropy.coordinates as coord
    from astropy.table import Table
    from galpy.potential import evaluatePotentials
    from galpy.orbit import Orbit
    from galpy.potential.mwpotentials import MWPotential2014

    from .utils.dustmap import DustMap

    #Check to make sure astrometry exists
    if not ( hasattr(self,'ra') and hasattr(self,'dec') \
            and hasattr(self,'pmra') and hasattr(self,'pmdec') \
            and hasattr(self,'dist') and hasattr(self,'vlos')):
        raise ValueError('Computing unbound probabilities requires full equatorial positions and velocities (ra, dec, parallax/distance, pmra, pmdec, vlos). Please make sure your sample includes these attributes.')

    #Check to see if errors exist. Compute them if not
    if not hasattr(self,'cov'):
        if not ( hasattr(self,'e_par') and hasattr(self,'e_pmra') and \
                hasattr(self,'e_pmdec') and hasattr(self,'e_vlos') ):
            print('Computing unbound probabilities requires uncertainties on positions and velocities in the equatorial frame. Calculating...')
            self.photometry(self.dust)

    #Cut stars with parallax errors above par_cut_val
    if par_cut_flag:
        idx = (self.e_par.to(u.mas)/self.par<=par_cut_val)
        self.subsample(np.where(idx)[0])

    if(self.size==0):
        self.Pub = np.zeros(self.size)
        return

    #Initialize...
    self.Pub = np.empty(self.size)
    self.e_GCv = np.empty(self.size)*u.km/u.s

    #Reassign ro, vo, zo, solarmotion if required
    self._check_rovozoso(ro, vo, zo, solarmotion)
    
    '''
    if solarmotion is None:
        if self.solarmotion is None:
            print('Warning: UVW Solar motion not provided. Defaulting to [-11.1, 12.24, 7.25]*u.km/u.s (Schonrich+2010)')
            solarmotion = [-11.1, 12.24, 7.25]*u.km/u.s
        else:
            solarmotion = self.solarmotion

    if vo is None:
        vo = get_physical(potential)['vo']*u.km/u.s
        
    if ro is None:
        ro = get_physical(potential)['ro']*u.kpc

    if zo is None:
            if self.zo is None:
                print('Warning: zo not provided. Defaulting to 20.8 pc (Bovy & Bennett 2019)')
                zo = 0.0208*u.kpc
            else:
                zo = self.zo
    '''
    for i in tqdm(range(self.size),desc='Calculating unbound probability...'):

        #Don't even calculate if star is very fast
        if(self.v0[i]>1500*u.km/u.s):
            self.Pub[i] = 1.
            self.e_GCv[i] = 0.
            
            continue
        
        o = self._sample_errors(index=i, numsamp=numsamp)

        GCv2 = np.sqrt(o.vx(quantity=True)**2 + o.vy(quantity=True)**2 
                + o.vz(quantity=True)**2)

        R2 = np.sqrt(o.x(quantity=True)**2 + o.y(quantity=True)**2)

        z2 = o.z(quantity=True)

        phi = np.arctan2(o.y(quantity=True),o.x(quantity=True))

        #For each sampled entry, get escape velocity
        Vesc = np.zeros(numsamp)*u.km/u.s
        for j in range(numsamp):
            Vesc[j] = np.sqrt(2*(- evaluatePotentials(potential, 
                    R2[j],z2[j],phi=phi[j], t = t, quantity=True)))

        #Calculate fraction of iterations above escape velocity
        inds = (GCv2 > Vesc)
        self.Pub[i] = len(GCv2[inds])/len(Vesc)

        #Calculate spread of sampled galactocentric velocity
        self.e_GCv[i] = np.std(GCv2)

#@get_Punbound
def get_P_velocity_greater(self, vcut, numsamp = int(5e1), 
                            par_cut_flag=True, par_cut_val = 0.2, 
                            solarmotion = None, zo=None, vo=None, ro=None):

    '''
    Sampling over provided observations w/ errors, returns probability 
    that star is observed with a total velocity above a certain threshold.

    Parameters:
    ---------------

    vcut : float
        Galactocentric velocity (in km/s) that is used for the cut

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

    solarmotion : None or length-3 list of floats or astropy quantity (km/s)
            UVW Solar peculiar velocity in km/s. 
            Galpy likes the U to be sign-flipped.
            Defaults to self.solarmotion if it exists. If it does not, defaults
            to Schonrich+2010

    zo : None or Float or astropy distance quantity
             Offset of the Sun above or below the Galactic plane.
             Default is self.zo if it exists.
             If self.zo does not exists, defaults to 20.8 pc (Bennett+Bovy 2019)
             If float, units are assumed to be kpc

    ro : None or Float (astropy length quantity)
        Distance to the Galactic Centre in kpc. 
        Defaults to self.ro, if it exists
        If self.ro does not exist, reads it from .galpyrc config file

    vo : None or Float (astropy velocity quentity)
        Circular velocity at the Solar position in km/s. 
        Defaults to self.vo, if it exists.
        If self.vo does not exist, reads it from the .galpyrc config file

    '''

    import astropy.coordinates as coord
    from astropy.table import Table

    from galpy.potential import evaluatePotentials
    from galpy.orbit import Orbit
    from galpy.potential.mwpotentials import MWPotential2014

    from .utils.dustmap import DustMap

    #Check to make sure astrometry exists
    if not ( hasattr(self,'ra') and hasattr(self,'dec') \
            and hasattr(self,'pmra') and hasattr(self,'pmdec') \
            and hasattr(self,'dist') and hasattr(self,'vlos')):
        raise ValueError('Computing unbound probabilities requires full equatorial positions and velocities (ra, dec, parallax/distance, pmra, pmdec, vlos). Please make sure your sample includes these attributes.')

    #Check to see if errors exist. Compute them if not
    if not hasattr(self,'cov'):
        if not ( hasattr(self,'e_par') and hasattr(self,'e_pmra') and \
                hasattr(self,'e_pmdec') and hasattr(self,'e_vlos') ):
            print('Computing unbound probabilities requires uncertainties on positions and velocities in the equatorial frame. Calculating...')
            self.photometry(self.dust)

    if(self.size==0):
        self.p_GCvcut = np.zeros(self.size)
        self.e_GCv = np.zeros(self.size)
        return

    #Cut stars with parallax errors above par_cut_val
    if par_cut_flag:
        idx = (self.e_par.to(u.mas)/self.par<=par_cut_val)
        self.subsample(np.where(idx)[0])

    #Initialize...
    self.p_GCvcut = np.empty(self.size)
    self.GCv_lb = np.empty(self.size)*u.km/u.s
    self.GCv_ub = np.empty(self.size)*u.km/u.s

    #Reassign ro, vo, zo, solarmotion, if needed
    self._check_rovozoso(ro, vo, zo, solarmotion)
    
    if isinstance(vcut,float):
            vcut = vcut*u.km/u.s

    if u.get_physical_type('speed') != u.get_physical_type(vcut):
        raise ValueError('Error: Invalid velocity cut in get_P_velocity_greater(). Supplied cut must be either float (assumed to be km/s) or an astropy velocity quantity')

    if not (hasattr(self,attr) for attr in ['e_vlos', 'cov']) \
            or not (hasattr(self,attr) \
                for attr in ['e_par', 'e_pmra', 'e_pmdec', 'e_vlos']):
        raise ValueError('Error: sample needs an attribute "e_vlos" (line-of-sight velocity error) AND EITHER an attribute "cov" (astrometric covariance matrix) or all of the attributes "e_par", "e_pmra", "e_pmdec" to call this function. See speedystar.observation.get_Gaia_errors for more information')

    if any(np.isnan(self.e_par)):
        print('Warning: Some stars without valid errors are included in the sample, most likely because they are too dim. They are being removed')
        self.subsample(np.where(np.isreal(self.e_par))[0])

    for i in tqdm(range(self.size),
                desc='Calculating high-v prob...'):

        o = self._sample_errors(index=i, numsamp=numsamp)

        GCv2 = np.sqrt(o.vx(quantity=True)**2 + o.vy(quantity=True)**2 
                        + o.vz(quantity=True)**2)
        #Calculate fraction of iterations above escape velocity
        inds = (GCv2 > vcut)
        self.p_GCvcut[i] = len(GCv2[inds])/len(GCv2)

        #Calculate spread of sampled galactocentric velocity
        self.GCv_lb[i], self.GCv_ub[i] = np.quantile(GCv2, [0.16,0.84])

def get_e_beta(self, par_cut_flag=True, par_cut_val = 0.2, numsamp = int(7.5e2), use_dist = False, solarmotion = None, zo=None, vo=None, ro=None):

    '''
    Samples over observations and errors to return angles (polar and azimuthal) between HVS's velocity in Galactocentric frame and position in Galactocentric frame

    Parameters:
    ---------------
    
    par_cut_flag: bool
        If True, samples over HVS with relative parallax uncertainties less 
        than par_cut_value

    numsamp : integer
        Number of times observations are sampled  
    
    ro, vo: integer 
        values of ro and vo used in propagation. ro in kpc vo in km/s
        
    Returns:
    --------------
    self.beta_theta_samp: array-like
        A numsamp x self.size where each column is an HVS and each row is the
        beta_theta calculated for that sample.

    self.beta_phi_samp: array-like
        A numsamp x self.size where each column is an HVS and each row is the
        beta_phi calculated for that sample.
    '''

    import astropy.coordinates as coord
    from astropy.table import Table
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u
    from galpy.orbit import Orbit

    #Reassign ro, vo, zo, solarmotion, if needed
    self._check_rovozoso(ro, vo, zo, solarmotion)
    
    #Cut stars with parallax errors above par_cut_val
    if par_cut_flag:
        idx = (self.e_par.to(u.mas)/self.par<=par_cut_val)
        self.subsample(np.where(idx)[0])
    
    if not (hasattr(self,attr) for attr in ['e_vlos', 'cov']) \
            or not (hasattr(self,attr) \
                for attr in ['e_par', 'e_pmra', 'e_pmdec', 'e_vlos']):

        raise ValueError('Error: sample needs an attribute "e_vlos" (line-of-sight velocity error) AND EITHER an attribute "cov" (astrometric covariance matrix) or all of the attributes "e_par", "e_pmra", "e_pmdec" to call this function. See speedystar.observation.get_Gaia_errors for more information')

    if any(np.isnan(self.e_par)):
        print('Warning: Some stars without valid errors are included in the sample, most likely because they are too dim. They are being removed')
        self.subsample(np.where(np.isreal(self.e_par))[0])
    
    #For each sampled entry, get beta 
    #array to hold beta samples
    self.beta_phi_samp = np.zeros((self.size,numsamp))
    self.beta_theta_samp = np.zeros((self.size,numsamp))
    
    #for i in range(self.size):
    for i in tqdm(range(self.size),desc='Calculating deflection...'):

        o = self._sample_errors(index=i, numsamp=numsamp)
            
        for j in range(numsamp): 

            phi_pos = np.arctan2(o.y(quantity=True)[j].value,o.x(quantity=True)[j].value)
            phi_vel = np.arctan2(o.vy(quantity=True)[j].value,o.vx(quantity=True)[j].value)

            theta_pos = np.arctan2(o.R(quantity=True)[j].value, o.z(quantity=True)[j].value)
            theta_vel = np.arctan2(o.vR(quantity=True)[j].value, o.vz(quantity=True)[j].value)

            self.beta_theta_samp[i][j] = theta_vel - theta_pos
            self.beta_phi_samp[i][j] = phi_vel - phi_pos


def evolve(self,Zsun=0.02):
    '''
    Evolve a star of a certain mass and metallicity until a certain age
    using either the SSE module in AMUSE or the included hurley scripts

    Parameters
    ----------
    Zsun : float
        Solar metallicity. Default is 0.02, which is the value used in 
        the Hurley scripts. If using AMUSE, this value is ignored and 
        the metallicity is set to the value in the sample.
    '''
    
    from .utils.hurley_stellar_evolution import get_t_BAGB, get_t_MS
    from .utils.hurley_stellar_evolution import Radius, get_TempRadL

    self.T_eff = np.ones(self.size)*u.K
    self.Rad = np.ones(self.size)*u.Rsun
    self.Lum = np.ones(self.size)*u.Lsun
    self.stage = np.ones(self.size)
    
    if not hasattr(self,'amuseflag'):
        self.amuseflag = False

    if self.amuseflag:

        #using the SSE module within AMUSE
        from amuse.units import units
        from amuse.community.sse.interface import SSE
        #from amuse.test.amusetest import get_path_to_results
        from amuse import datamodel

        for z in (np.unique(self.met)):  

            #indices with ith metallicity
            idx = np.where(self.met==z)[0]

            #Initialize
            stellar_evolution = SSE()
            #Adjust metallicity for new Zsun assumption
            stellar_evolution.parameters.metallicity = Zsun*10**(z)
            star      = datamodel.Particles(len(self.m[idx].value))
            star.mass = self.m[idx].value | units.MSun

            age = self.tage[idx].to('Myr').value | units.Myr

            #Evolve the star
            star = stellar_evolution.particles.add_particles(star)
            stellar_evolution.commit_particles()
            stellar_evolution.evolve_model(end_time = age)

            stellar_evolution.stop()

            #Extract HVS effective temperature, radius, 
            #luminosity, evolutionary stage
            self.T_eff[idx] = star.temperature.as_astropy_quantity().to('K')
            self.Rad[idx] = star.radius.as_astropy_quantity().to('Rsun') 
            self.Lum[idx] = star.luminosity.as_astropy_quantity().to('Lsun')
            self.stage[idx] = star.stellar_type.as_astropy_quantity()

    else:
        #using the included Hurley scripts
        for i in tqdm(range(len(self.m)), desc='Evolving HVSs'):   

            #Get HVS effective temperature, radius, luminosity
            #for i in range(len(m[idx])):   
            T_eff, R, Lum = get_TempRadL(self.m[i].value,
                                    self.met[i], self.tage[i].value)

            self.T_eff[i] = T_eff*u.K
            self.Rad[i] = R*u.Rsun
            self.Lum[i] = Lum*u.Lsun

            self.stage[i] = 1