__author__ = 'Fraser Evans'
__version__ = '1.5'

import time
import os
from astropy import units as u
import numpy as np
import copy

from .eject import EjectionModel

class starsample:
    '''
    HVS sample class. Main features:

    - Generate a sample of HVS with a specified ejection model
    - Propagate the ejection sample in the Galaxy
    - Obtain mock photometry for the sample in a variety of bands
    - Obtain mock astrometric and radial velocity errors
    - Subsample the population according to various criteria
    - Save/Load catalogues FITS files

   # Common attributes
    ---------
    self.size : int
        Size of the sample
    self.name : str
        Catalog name, 'Unknown'  by default
    self.ejmodel_name : str
        String of the ejection model used to generate the sample.
       'Unknown' by default
    self.dt : Quantity
        Timestep used for orbit integration, 0.01 Myr by default
    self.T_MW : Quantity
        Milky Way maximum lifetime

    self.r0, self.phi0, self.theta0, self.v0, self.phiv0, self.thetav0
        Initial configuration at ejection in galactocentric 
        spherical coordinates
    self.tage, self.tflight : Quantity
        Age and flight time of the stars
    self.m
        Stellar masses of the sample
    self.a, self.P, self.q : Quantity/Quantity/Float
        Orbital separation, orbital period and mass ratio of HVS
        progenitor binary
    self.met : Float
        Metallicity xi = log10(Z/Z_sun)
    self.stage, self.stagebefore : integer
        Evolutionary stage of the star _today_ and at the moment of ejection
        see for the meanings of each stage:
        https://ui.adsabs.harvard.edu/abs/2000MNRAS.315..543H/abstract      
    self.Rad, self.T_eff, self.Lum : Quantity
        Radius, temperature and luminosity

    self.ra, self.dec, self.dist, self.pmra, self.pmdec, self.vlos : Quantity
        ICRS positions and velocities. 
        Proper motion in ra direction is declination-corrected.
    self.par : Quantity
        Parallax
    self.e_ra, self.e_dec, self.e_par : Quantity
        Position and parallax uncertainties
    self.e_pmra, self.e_pmdec, self.e_vlos : Quantity
        Proper motion and line-of-sight velocity uncertainties

    self.x, self.y, self.z, self.vx, self.vy, self.vz : Quantity
        Positions and velocities in Galactocentric Cartesian coordinates
    self.GCdist, self.GCv : Quantity
        Galactocentric total distance and total velocity
    self.e_GCv : Quantity
        Standard deviation of the star's total velocity, given
        observational errors
    self.thetaf, self.phif : Float
        Polar (+z towards xy) and azimuthal (+x towards +y) coordinates in 
        Galactocentric spherical coordinates

    self.Pub : Float
        Probability, given the star's observational error, that it will be
        it is fast enough to escape the Galaxy

    self.Vesc : Quantity
        Escape velocity to infinity from the star's current position

    self.GRVS, self.G, self.RP, ... : ndarray
        Apparent magnitude in many photometric bands

    Methods
    -------
    __init__():
        Initializes the class: loads catalog if one is provided, 
        otherwise creates one based on a given ejection model
    _eject(): 
        Initializes a sample at t=0 ejected from Sgr A*
    backprop():
        Propagates a sample backwards in time for a given max integration time
    propagate():
        Propagates a sample forewards in time for a given max integrating time
    get_vesc():
        Determines the escape velocity from the Galaxy for each mock HVS 
    photometry():
        Calculates the apparent magnitudes in several different bands.
        Can also calculate astrometric and radial velocity uncertainties
    get_P_unbound():
        Samples over observational errors,
        determines unbound probabilities depending on errors
    get_P_velocity_greater():
        Samples over observational errors, determines probabilities that mock
        HVSs are faster than a specified velocity cutoff
    subsample():
        For a propagated sample, returns a subsample - either specific indices,
        a random selection of stars or ones that meet given criteria
    save():
        Saves the sample in a FITS file
    _load():
        Load existing HVSsample, either ejected or propagated
    loadExt():
        If a sample was NOT created here, reads it in as an HVSsample
    likelihood():
        Checks the likelihood of the sample for a given potential
    zero_point():
        Estimates the Gaia parallax zero point for each star
    '''

    from .dynamics import propagate, backprop, get_vesc, _R
    from .dynamics import likelihood, get_betas, get_betas_Boubert
    from .observation import photometry, zero_point, get_Punbound 
    from .observation import get_P_velocity_greater, evolve, get_Gaia_errors
    from .observation import get_e_beta
    from .observation import _check_rovozoso, _sample_errors
    from .saveload import save, _load, _loadExt
    from .subsample import subsample
    from .config import fetch_dust, config_dust, config_astrosf
    from .config import config_brutus, query_yes_no
    from .config import config_rvssf, set_ast_sf, set_Gaia_release

    dt   = 0.01*u.Myr
    T_MW = 13.8*u.Gyr # MW maximum lifetime from Planck2015

    __attrs__ = {
    'size': '           Metavariable: size of the sample (integer)',
    'ejmodel_name': '   Metavariable: Ejection model name (string)',  
    'name': '           Metavariable: Catalog name (string)',
    'dt': '             Metavariable: Timestep for orbit integration (astropy time quantity)',
    'cut_factor': '     Metavariable: Factor by which sample is cut down relative to its true size (float)',
    't_mw': '           Metavariable: Milky Way maximum lifetime (astropy time quantity)',
    'use_ast_sf': '     Metavariable: Flag for using Gaia astrometric spread function (boolean)',
    'use_rv_sf': '      Metavariable: Flag for using Gaia RV selection function (boolean)',
    'gaia_release': '   Metavariable: The assumed Gaia data release (string)',
    'vo': '             Metavariable: Circular velocity at the Solar position (astropy velocity quantity)',
    'ro': '             Metavariable: Galactocentric radius at the Sun position (astropy distance quantity)',
    'zo': '             Metavariable: Height of the Sun above the Galactic midplane (astropy distance quantity)',
    'solarmotion': '    Metavariable: Solar velocity relative to LSR (length-3 array, astropy velocity quantity)',
    'alpha': '          Metavariable: HVS progenitor binary power-law slope of the separation distribution (float)',
    'gamma': '          Metavariable: HVS progenitor binary power-law slope of the mass ratio distribution (float)',
    'kappa': '          Metavariable: IMF power-law slope of the HVS progenitor binary primary',
    'tflightmax': '     Metavariable: Maximum flight time of the star (astropy time quantity)',
    'zsun': '           Metavariable: Solar metallicity (float)',
    'rate': '           Metavariable: Rate of HVS ejection (astropy 1/time quantity)',
    'm_bh': '           Metavariable: Mass of the (primary or only) MBH (astropy mass quantity)',
    'v_range': '        Metavariable: Range of allowed HVS velocities (length-2 astropy velocity quantity)',
    'm_range': '        Metavariable: Range of allowed HVS masses (length-2 astropy mass quantity)',
    'm_c': '            Metavariable: Mass of the companion BH (astropy mass quantity)',
    'qbhb': '           Metavariable: Mass ratio of the BHB (float)',
    'current_a': '      Metavariable: Separation of the BHB in the present day (astropy distance quantity)',
    'tlb': '            Metavariable: Time since the BHB merger ("LookBack" time) (astropy time quantity)',
    'a0': '             Metavariable: Initial separation of the BHB (astropy distance quantity)',
    'rho': '            Metavariable: Stellar density in the Milky Way nuclear star cluster (astropy mass density quantity)',
    'sigma': '          Metavariable: Stellar velocity dispersion in the Milky Way nuclear star cluster (astropy velocity quantity)',
    'saveorbit': '      Metavariable: Flag for saving the orbit of the BHB (boolean)',
    'orbitpath': '      Metavariable: Path to save the orbit of the BHB (string)',
    'vmin': '           Metavariable: Minimum velocity above which a star counts as "ejected" from the BHBH (astropy velocity quantity)',
    'propagated': '     Metavariable: Flag for whether the sample has been propagated (boolean)',
    'potential': '      Metavariable: Potential used for orbit propagation (galpy potential object)',
    'dust': '           Metavariable: Dust map used (dustmap object)',
    'r0': '             Initial galactocentric radius (astropy distance quantity)',
    'phi0': '           Initial galactocentric azimuthal angle (astropy angle  quantity)',
    'theta0': '         Initial galactocentric polar angle (astropy angle quantity)',
    'v0': '             Initial galactocentric radial velocity (astropy velocity quantity)',
    'phiv0': '          Initial galactocentric azimuthal velocity (astropy angle quantity)',
    'thetav0': '        Initial galactocentric polar velocity (astropy angle quantity)',
    'tage': '           Age of the star (astropy time quantity)',
    'tflight': '        Flight time of the star (astropy time quantity)',
    'm': '              Stellar mass (astropy mass quantity)',
    'P': '              Orbital period of the HVS progenitor binary (astropy time quantity)',
    'sep': '            Separation of the HVS progenitor binary (astropy distance quantity)',
    'q': '              Mass ratio of the HVS progenitor binary (float)',
    'met': '            Metallicity (float)',
    'mem': '            Whether the heavier (1) or lighter (2) star is the HVS (integer)',
    'a_BHB': '          Separation of the BHB at the moment of HVS ejection  (astropy distance quantity)',
    'aah_BHB': '        Separation of the BHB at the moment of HVS ejection as a fraction of its hardening radius (float)',
    'vc': '             Circular velocity at the moment of HVS ejection (astropy velocity quantity)',
    
    'stage': '          Evolutionary stage of the star (integer)',
    'stagebefore': '    Evolutionary stage of the primary star at the moment of ejection (integer)',
    'Rad': '            Radius of the star (astropy distance quantity)',
    'T_eff': '          Effective temperature of the star (astropy temperature quantity)',
    'Lum': '            Luminosity of the star (astropy luminosity quantity)',
    'ra': '             Right ascension of the star (astropy angle quantity)',
    'dec': '            Declination of the star (astropy angle quantity)',
    'dist': '           Heloiocentric distance to the star (astropy distance quantity)',
    'pmra': '           Proper motion in right ascension (astropy angular velocity uantity)',
    'pmdec': '          Proper motion in declination (astropy angular velocity quantity)',
    'vlos': '           Line-of-sight velocity of the star (astropy velocity quantity)',
    'par': '            Parallax of the star (astropy angle quantity)',
    'l': '              Galactic longitude of the star (astropy angle quantity)',
    'b': '              Galactic latitude of the star (astropy angle quantity)',
    'pml': '            Galactic proper motion in longitude (astropy angular velocity quantity)',
    'pmb': '            Galactic proper motion in latitude (astropy angular velocity quantity)',
    'vr': '             Spherical radial velocity (astropy velocity quantity)',
    'vT': '             Spherical/cylindrical tangential velocity (astropy velocity quantity)',
    'vtheta': '         Spherical polar velocity (astropy velocity quantity)',
    'Lz': '             Angular momentum in the z direction (astropy angular momentum quantity)',
    'R': '              Galactocentric cylindrical radius (astropy distance quantity)',
    'vR': '             Galactocentric cylindrical radial velocity (astropy velocity quantity)',
    'x': '              Galactocentric x position (astropy distance quantity)',
    'y': '              Galactocentric y position (astropy distance quantity)',
    'z': '              Galactocentric z position (astropy distance quantity)',
    'thetaf': '         Galactocentric polar angle (astropy angle quantity)',
    'phif': '           Galactocentric azimuthal angle (astropy angle quantity)',
    'vx': '             Galactocentric x velocity (astropy velocity quantity)',
    'vy': '             Galactocentric y velocity (astropy velocity quantity)',
    'vz': '             Galactocentric z velocity (astropy velocity quantity)',
    'GCdist': '         Galactocentric distance (astropy distance quantity)',
    'GCv': '            Galactocentric velocity (astropy velocity quantity)',
    'beta_theta': '     Polar angle between Galactocentric position vector and velocity vector (astropy angle quantity)',
    'beta_phi': '       Azimuthal angle between Galactocentric position vector and velocity vector (astropy angle quantity)',
    'orbits': '         Full forward orbit of the stars (galpy Orbit object)',
    'backorbits': '     Full backward orbit of the stars (galpy Orbit object)',
    'lnlike': '         Log-likelihood of trajectory in the given potential (float)',
    'zp': '             Zero point offset of the parallax (astropy angle quantity)',
    'e_ra': '           Uncertainty in right ascension (astropy angle quantity)',
    'e_dec': '          Uncertainty in declination (astropy angle quantity)',
    'e_par': '          Uncertainty in parallax (astropy angle 1uantity)',
    'e_pmra': '         Uncertainty in RA proper motion (astropy angular velocity quantity)',
    'e_pmdec': '        Uncertainty in declination proper motion (astropy angular velocity quantity)',
    'cov': '            Astrometric covariance matrix (5x5xsize float array)',
    'corr_ra_dec': '    Correlation between RA and dec errors (float)',
    'corr_ra_par': '    Correlation between RA and parallax errors (float)',
    'corr_ra_pmra': '   Correlation between RA and pmra errors (float)',
    'corr_ra_pmdec': '  Correlation between RA and pmdec errors (float)',
    'corr_dec_par': '   Correlation between dec and parallax errors (float)',
    'corr_dec_pmra': '  Correlation between dec and pmra errors (float)',
    'corr_dec_pmdec': ' Correlation between dec and pmdec errors (float)',
    'corr_par_pmra': '  Correlation between parallax and pmra errors (float)',
    'corr_par_pmdec': ' Correlation between parallax and pmdec errors (float)',
    'corr_pmra_pmdec': 'Correlation between pmra and pmdec errors (float)',
    'beta_phi_samp': '  Sampled beta_phi values from errors (n x size astropy angule quantity)',
    'beta_theta_samp': 'Sampled beta_theta values (n x size astropy angle quantity)',
    'e_vlos': '         Uncertainty in line-of-sight velocity (astropy velocity quantity)',
    'e_GCv': '          Uncertainty in galactocentric velocity (astropy velocity quantity)',
    'Pub': '            Probability of being unbound (float)',
    'P_GCvcut': '       Prob. of being faster than the given velocity cut (float)',
    'GCv_lb': '         1-sigma lower bound of the GCv given errors (astropy velocity quantity)',
    'GCv_ub': '         1-sigma upper bound of the GCv given errors (astropy velocity quantity)',
    'Vesc': '           Galactic escape velocity at the star position (astropy velocity quantity)',
    'Av': '             Dust extinction in V band (astropy magnitude quantity)',
    'Gaia_GRVS': '      Apparent magnitude in the Gaia RVS band (astropy dimentionsless quantity)',
    'Gaia_G': '         Apparent magnitude in the Gaia G band (astropy dimentionsless quantity)',
    'Gaia_RP': '        Apparent magnitude in the Gaia RP band (astropy  dimentionsless quantity)',
    'Gaia_BP': '        Apparent magnitude in the Gaia BP band (astropy  dimentionsless quantity)',
    'Bessell_U': '      Apparent magnitude in the Bessell U band (astropy  dimentionsless quantity)',
    'Bessell_V': '      Apparent magnitude in the Bessell V band (astropy  dimentionsless quantity)',
    'Bessell_B': '      Apparent magnitude in the Bessell B band (astropy  dimentionsless quantity)',
    'Bessell_R': '      Apparent magnitude in the Bessell R band (astropy  dimentionsless quantity)',
    'Bessell_I': '      Apparent magnitude in the Bessell I band (astropy  dimentionsless quantity)',
    'SDSS_u': '         Apparent magnitude in the SDSS u band (astropy  dimentionsless quantity)',
    'SDSS_g': '         Apparent magnitude in the SDSS g band (astropy  dimentionsless quantity)',
    'SDSS_r': '         Apparent magnitude in the SDSS r band (astropy  dimentionsless quantity)',
    'SDSS_i': '         Apparent magnitude in the SDSS i band (astropy  dimentionsless quantity)',
    'SDSS_z': '         Apparent magnitude in the SDSS z band (astropy  dimentionsless quantity)',
    'SDSS_Y': '         Apparent magnitude in the SDSS Y band (astropy  dimentionsless quantity)',
    'DECam_u': '        Apparent magnitude in the DECam u band (astropy  dimentionsless quantity)',
    'DECam_g': '        Apparent magnitude in the DECam g band (astropy  dimentionsless quantity)',
    'DECam_r': '        Apparent magnitude in the DECam r band (astropy  dimentionsless quantity)',
    'DECam_i': '        Apparent magnitude in the DECam i band (astropy  dimentionsless quantity)',
    'DECam_z': '        Apparent magnitude in the DECam z band (astropy  dimentionsless quantity)',
    'DECam_Y': '        Apparent magnitude in the DECam Y band (astropy  dimentionsless quantity)',
    'LSST_u': '         Apparent magnitude in the LSST u band (astropy  dimentionsless quantity)',
    'LSST_g': '         Apparent magnitude in the LSST g band (astropy  dimentionsless quantity)',
    'LSST_r': '         Apparent magnitude in the LSST r band (astropy  dimentionsless quantity)',
    'LSST_i': '         Apparent magnitude in the LSST i band (astropy  dimentionsless quantity)',
    'LSST_z': '         Apparent magnitude in the LSST z band (astropy  dimentionsless quantity)',
    'VISTA_Z': '        Apparent magnitude in the VISTA Z band (astropy  dimentionsless quantity)',
    'VISTA_Y': '        Apparent magnitude in the VISTA Y band (astropy  dimentionsless quantity)',
    'VISTA_J': '        Apparent magnitude in the VISTA J band (astropy  dimentionsless quantity)',
    'VISTA_H': '        Apparent magnitude in the VISTA H band (astropy  dimentionsless quantity)',
    'VISTA_Ks': '       Apparent magnitude in the VISTA Ks band (astropy  dimentionsless quantity)',
    'obsprob': '        Probability of observing the star (float)',
    }

    #@init
    def __init__(self, inputdata=None, name=None, isExternal=False,**kwargs):
        '''
        Parameters
        ----------
        inputdata : EjectionModel or str
            Instance of an ejection model or string to the catalog path
        name : str
            Name of the catalog
        isExternal : Bool
            Flag if the loaded catalog was externally generated, 
            i.e. not by this package
        '''

        if(inputdata is None):
            raise ValueError('Initialize the class by either providing an \
                                ejection model or an input HVS catalog.')

        #Name catalog
        if(name is None):
            self.name = 'HVS catalog '+str(time.time())
        else:
            self.name = name

        #By default, astrometric errors are NOT calculated using the actual
        #Gaia astrometric spread function, but rather with PyGaia based on 
        #pre-launch predicted Gaia performance.
        #Setting use_ast_sf to True with speedystar.config.set_ast_sf()
        #calculates astrometric errors using the Gaiaunlimited package, 
        #which is computationally more expensive but is more accurate,
        #particularly for bright sources.

        self.use_ast_sf = False

        #If inputdata is ejection model, create new ejection sample
        if isinstance(inputdata, EjectionModel):
            self._eject(inputdata, **kwargs)

        # If inputdata is a filename and isExternal=True, 
        # loads existing sample of star observations    
        if(isinstance(inputdata, str) and (isExternal)):
            self._loadExt(inputdata)

        #If inputdata is a filename and isExternal=False, 
        # loads existing already-propagated sample of stars
        if (isinstance(inputdata, str) and (not isExternal)):
            self._load(inputdata,**kwargs)


    def __getitem__(self,item):
        '''
        Slice the sameple, similar functionality to .subsample()
        '''

        if isinstance(item, slice):

            if item.step is None:
                step = 1
            else:
                step = item.step
            if item.start is None:
                item = slice(0, item.stop, step)
            elif item.stop is None:
                item = slice(item.start, self.size, step)
            else:
                item = slice(item.start, item.stop, step)

            clone = copy.copy(self)
            clone.subsample(np.arange(item.start, item.stop, item.step))
            return clone

        #Return a single star 
        elif isinstance(item, int):
            if item < 0: # Handle negative indices
                item += self.size
            if item < 0 or item >= self.size:
                
                raise IndexError("The index (%d) is out of range." % item)

            clone = copy.copy(self)
            clone.subsample(np.array([item]))
            return clone

        #Return a sample based on boolean array
        elif isinstance(item, np.ndarray):
            #if item.dtype == bool: 
            if np.issubdtype(item.dtype, np.integer):
                clone = copy.copy(self)
                clone.subsample(item)   
                #self.subsample(np.where(item)[0])
                return clone
        
        #If argument is a string, return the attribute
        #elif isinstance(item, str):
        #    if hasattr(self, item):
        #        return getattr(self, item)
        #    else:
        #        try:
        #            clone = copy.copy(self)
        #            clone.subsample(item)
        #            return clone
        #        except:
        #            raise AttributeError("Attribute %s not found." % item)

        #If argument is a string, return the attribute
        elif isinstance(item, str):
            clone = copy.copy(self)
            clone.subsample(item)
            return clone

        raise TypeError("Invalid argument type. __getitem__ only accepts integers, slices, strings or integer-typed numpy arrays as arguments.")

    def __setitem__(self, key, value):
        '''
        Set the value of an attribute
        '''

        setattr(self, key, value)

    def dir(self):
        '''
        Returns a list of all attributes of the class
        '''
        return [a for a in dir(self) if not a.startswith("_")]

    def attrsummary(self):
        '''
        Returns a summary of the attributes of the class
        '''
        for key in self.__attrs__:
            if key in vars(self):

                print(str(key) + ': ' + self.__attrs__[key])
            #else:
            #    print(str(key) + ': Unknown')
        for key in vars(self):
            if key not in self.__attrs__:
                print(str(key) + ': Unknown')

    def whatis(self, attrstr):
        '''
        Returns a description of the attribute
        '''
        if attrstr in self.__attrs__:
            print(str(attrstr) + ': ' + self.__attrs__[attrstr])
        else:
            print(str(attrstr) + ': Unknown')

    def list_cuts(self):
        '''
        Returns a list of all hard-coded cuts that can be applied to the sample
        '''

        my_cuts = {
            'Gaia_DR2': '     Stars detectable in Gaia Data Release 2 (G < 20.7)',
            'Gaia_EDR3': '    Stars detectable in Gaia Early Data Release 3 (G < 20.7)',
            'Gaia_DR3': '     Stars detectable in Gaia Data Release 3 (G < 20.7)',
            'Gaia_DR4': '     Stars detectable in Gaia Data Release 4 (G < 20.7)',
            'Gaia_DR5': '     Stars detectable in Gaia Data Release 5 (G < 20.7)',
            'Gaia_6D_DR2': '  Stars detectable in Gaia Data Release 2 radial velocity catalogue (G_RVS < 12, or using the actual selection function)',
            'Gaia_6D_EDR3': ' Stars detectable in Gaia Data Release 3 radial velocity catalogue (G_RVS < 12, or using the actual selection function)',
            'Gaia_6D_DR3': '  Stars detectable in Gaia Data Release 3 radial velocity catalogue ( (G_RVS < 14 AND T_eff < 6900 K) OR (G_RVS < 12 AND T_eff < 14500 K) , or using the actual selection function',
            'Gaia_6D_DR4': '  Stars detectable in Gaia Data Release 4 radial velocity catalogue ( (G_RVS < 16.2 AND T_eff < 6900 K) OR (G_RVS < 14) )',
            'Gaia_6D_DR5': '  Stars detectable in Gaia Data Release 5 radial velocity catalogue ( (G_RVS < 16.2 AND T_eff < 6900 K) OR (G_RVS < 14) )',
            'S5': '           Stars detectable in the S5 survey (see documentation)'
        }

        print('The following cuts are available:')
        for key in my_cuts:
            print(str(key) + ': ' + my_cuts[key])

    #@eject
    def _eject(self, ejmodel, **kwargs):
        '''
        Initializes the sample as an ejection sample
        '''

        self.ejmodel_name = ejmodel._name

        self.propagated = False
        
        ejargs = ejmodel.sampler(**kwargs)

        for key in list(ejargs.keys()):
            setattr(self,key,ejargs[key])