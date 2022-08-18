__author__ = 'Fraser Evans'
__version__ = '1.0'

__ImportError__ = "One or more required external packages have not been installed. Please see requirements.txt"

import time
import os

try:
    from tqdm import tqdm
    from astropy import units as u
    import scanninglaw.asf as astrospreadfunc
except ImportError:
    raise ImportError(__ImportError__)

from .eject import EjectionModel

from .utils.mwpotential import PotDiff

class starsample:

    from .dynamics import propagate, backprop, get_vesc
    from .observation import photometry, get_Punbound
    from .saveload import save, _load, _loadExt
    from .subsample import subsample

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
    photometry():
        Calculates the apparent magnitudes in several different bands.
        Can also calculate astrometric and radial velocity uncertainties
    Get_P_unbound():
        Samples over observational errors
        Determines unbound probabilities depending on errors
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
    '''

    dt   = 0.01*u.Myr
    T_MW = 13.8*u.Gyr # MW maximum lifetime from Planck2015

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

    #@eject
    def _eject(self, ejmodel, **kwargs):

        '''
        Initializes the sample as an ejection sample
        '''

        self.ejmodel_name = ejmodel._name

        self.cattype = 0

        # See ejmodel.sampler() for descriptions of returned attributes
        self.r0, self.phi0, self.theta0, self.v0, self.phiv0, self.thetav0, \
        self.m, self.tage, self.tflight, self.a, self.P, self.q, self.mem, \
        self.met, self.stage, self.stagebefore, self.Rad, self.T_eff, \
        self.Lum, self.size = ejmodel.sampler(**kwargs)

    def fetch_dust(self,path='./'):
        '''
        Download the desired dust map. Please see mwdust:
        https://github.com/jobovy/mwdust
        WARNING. Default installation maps take up 5.4 GB in total

        Alternatively, download maps directly from the following URLs
        Combined19 : https://zenodo.org/record/3566060/files/combine19.h5
        Combined15 : https://zenodo.org/record/31262/files/dust-map-3d.h5

        Arguments
        --------
        path: string
            directory that will contain the dust data
        '''

        envcommand = 'setenv DUST_DIR '+path
        os.system(envcommand)
        os.system('git clone https://github.com/jobovy/mwdust.git')
        os.chdir('./mwdust')
        os.system('python setup.py install --user')

    def config_dust(self,path='./'):
        '''
        Load in the dust map used for photometry calculations

        Arguments
        ----------
        path: string
            path where the desired dust map can be found            
        '''

        from .utils.dustmap import DustMap
        self.dust = DustMap(path)

    def config_rvssf(self,path):

        '''
        Fetch Gaia radial velocity selection functions

        Arguments
        ----------
        path: string
            path where you want the selection functions installed.
            Note -- requires ~473 Mb of space
        '''

        #import .utils.selectionfunctions.cog_v as CogV
        from .utils.selectionfunctions import cog_v as CogV
        from .utils.selectionfunctions.config import config

        config['data_dir'] = path
        CogV.fetch(subset='rvs')

    def config_astrosf(self,path):
        '''
        Fetch Gaia astrometric spread functions

        Arguments
        ----------
        path: string
            path where you want the selection functions installed.
            Note -- requires ~435 Mb of space
        '''

        try:
            from scanninglaw.config import config
            import scanninglaw.asf
        except ImportError:
            raise ImportError(__ImportError__)       

        config['data_dir'] = path
        scanninglaw.asf.fetch()
        scanninglaw.asf.fetch(version='dr3_nominal')
