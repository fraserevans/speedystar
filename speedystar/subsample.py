TeffError = 'Effective temperatures are required to determine which sources would appear in the Gaia DR2 radial velocity catalogue. Please ensure your sample has a "T_eff" attribute. This should have been added when the ejection sample was created. If your catalogue was loaded externally, T_eff can be calculated using evo_pop() or get_stellar_parameters()'

rvssfWarning = 'Gaia G and G_RP band apparent magnitude must be computed computed to select stars visible in the Gaia radial velocity catalogue. Calculating...'

radecError = 'Error: right ascension and declination of sources must be known. Please ensure "ra" and "dec" are attributes of your sample.'

GRVSWarning = 'Gaia G_RVS band apparent magnitude must be computed to select stars visible in the Gaia radial velocity catalogue. Calculating...'

GWarning = 'Gaia G band apparent magnitude must be known to compute Gaia astrometric spread function. Calculating...'

def subsample(self, cut=None, use_rvs_sf=False, probcut=None):
    '''
    Restrict the sample based on the values of cut.

    Parameters
    ----------
    cut : str or np.array or int
        If str, sample is cut according to pre-defined criteria
        If np.array, the indices corresponding to cut are selected,
        If int, N=cut number of stars are selected at random.

    use_rvs_sf: Boolean
        If true, uses the Gaia DR2 spectroscopic selection function to 
        determine the probability that each star would appear in the 
        radial veloity catalogue of Gaia DR2/EDR3. 
        ONLY applies if cut=='Gaia_6D_DR2' or cut=='Gaia_6D_EDR3'.

    probcut: float
        If use_rvs_sf is true, probcut is the probability cut applied
        to the stars in the sample. Stars with a probability of
        appearing in the Gaia DR2/EDR3 radial velocity catalogue
        greater than probcut are kept. Stars with a probability
        less than probcut are removed.
        If probcut is None, a random number in [0,1] is drawn for each
        star in the sample. Stars with a random number less than
        the probability of appearing in the Gaia DR2/EDR3 radial
        velocity catalogue are kept.
    '''

    global dr3sf, dr3astsf, dr3rvssf, dr3rvssfvar
    global dr2sf, dr2astsf, dr2rvssf, dr2rvssfvar, dr2rvssf2

    #from galpydev.galpy.util.coords import radec_to_lb
    import astropy
    import numpy  as np
    import xarray as xr
    import importlib
    from astropy import units as u

    from astropy.coordinates import SkyCoord
    import speedystar

    if use_rvs_sf:
        from gaiaunlimited.selectionfunctions import DR2SelectionFunction, DR3RVSSelectionFunction, DR3SelectionFunctionTCG
        from gaiaunlimited.utils import get_healpix_centers, coord2healpix
        from gaiaunlimited.selectionfunctions import EDR3RVSSelectionFunction

    def _make_probcut(probcut):
        #Helper function to decide which stars to keep if a selection 
        #function is applied
        if probcut is None:
            # Designate a star as appearing in Gaia DR2 6D 
            # if a randomly drawn number in [0,1] 
            # is less than obsprob  
            ur  = np.random.uniform(0,1,len(self.obsprob))
            idx = (ur<=self.obsprob)
            self.subsample(np.where(idx)[0])
        else:
            try:
                probcut = float(probcut)
            except:
                raise ValueError('probcut must be a float')
            if probcut < 0 or probcut > 1:
                raise ValueError('probcut must be between 0 and 1')
            
            #Return stars with observation probability greater than probcut
            self.subsample(np.where(self.obsprob>probcut)[0])

    namelist = []

    for name in vars(self).keys():
        if isinstance(vars(self)[name],astropy.units.quantity.Quantity):
            if isinstance(vars(self)[name].value,np.ndarray):
                if vars(self)[name].shape[-1]==self.size:
                    namelist.append(name)
        elif isinstance(vars(self)[name],np.ndarray):
            if vars(self)[name].shape[-1]==self.size:
                namelist.append(name)

    if self.size == 0:
        print("Warning: Sample already consists of zero stars. No further subsampling can be performed.")
        return

    if(type(cut) is np.ndarray):
        cut.flatten()
        for varname in namelist:
            try:
                setattr(self, varname, \
                        #getattr(self, varname)[cut].flatten())
                        getattr(self, varname)[...,cut])
            except:
                print(['oops! ',varname, ' is an attribute of the sample with the appropriate length but cannot be accessed. Something has gone wrong'])
                pass
        self.size = cut.size

    elif (cut == 'Gaia_6D_DR2_Gaiaverse'):
        '''
        #Old Gaia DR2 cut based on Gaiaverse project. Now superceded by 
        #GaiaUnlimited cut. Code left for posterity
        #Selects stars visible in the radial velocity catalogue of Gaia DR2
        #If use_rvs_sf, use Gaiaverse DR2 spectroscopic selection function
        #Otherwise, uses only RVS band magnitiude
        #Uses DR2 astrometric spread function for errors
        #Otherwise, scales up predicted DR4 errors from PyGaia

        if not hasattr(self,'T_eff'):
            raise ValueError(TeffError)

        #Only cool stars are assigned a radial velocity in 
        #Gaia DR2 (Katz+2019)
        idx = (self.T_eff<6900*u.K)
        self.subsample(np.where(idx)[0])

        if self.size > 0:
            if 'dr2rvssf' not in globals():

                #from utils.selectionfunctions.config import config
                #config['data_dir'] = '/net/alm/data1/Cats/cog_iv/'

                #Load in radial velocity selection function
                dr2rvssf = CoGV.subset_sf(map_fname='rvs_cogv.h5',
                            nside=32,basis_options= {'needlet' \
                            :'chisquare','p':1.0,'wavelet_tol':1e-2},
                            spherical_basis_directory='SphericalBasis')

            if not (hasattr(self,'Gaia_G') and hasattr(self,'Gaia_RP')):
                print(rvssfWarning)
                self.photometry(bands=['Gaia_G', 'Gaia_RP'])

            if not ( hasattr(self,'ra') and hasattr(self,'dec') ):
                raise ValueError(radecError)

            # Position, brightness and colour of each source
            rsource = rSource(self.ra,self.dec,frame='icrs', 
                        photometry={'gaia_g':self.Gaia_G,\
                        'gaia_g_gaia_rp':self.Gaia_G - self.Gaia_RP})
            
            #Get the probability for observing each star in DR2 6D
            self.obsprob = dr2rvssf(rsource)

            # Designate a star as appearing in Gaia DR2 6D if a 
            # randomly drawn number in [0,1] is less than obsprob  
            ur  = np.random.uniform(0,1,len(self.obsprob))
            idx = (ur<=self.obsprob)
            self.subsample(np.where(idx)[0])

        else:

            if not ( hasattr(self,'Gaia_GRVS') ):
                print(MagWarning)

                self.photometry(bands=['Gaia_GRVS'])

            # G_RVS < 12 is the nominal faint-end magnitude limit of 
            # the Gaia DR2 radial velocity catalogue
            idx = (self.Gaia_GRVS < 12)
            self.subsample(np.where(idx)[0])

        if self.size > 0:
            self.set_Gaia_release('DR2')
            self.get_Gaia_errors()
    '''

    elif (cut == 'Gaia_6D_DR2'):
        #Selects stars visible in the radial velocity catalogue of Gaia DR2
        #If use_rvs_sf, use Gaiaunlimited DR2 selection function
        #Otherwise, uses only RVS band magnitiude

        if not hasattr(self,'T_eff'):
            raise ValueError(TeffError)

        #Only cool stars are assigned a radial velocity in 
        #Gaia DR2 (Katz+2019)
        idx = (self.T_eff<6900*u.K)
        self.subsample(np.where(idx)[0])

        if self.size > 0:
            #print(self.size)
            if use_rvs_sf:
                if 'dr2rvssf' not in globals():

                    dr2rvssf = EDR3RVSSelectionFunction()
                    dr2sf = DR2SelectionFunction()

                    #from .utils.varmap import EDR3RVSSelectionFunctionVar
                    #dr2rvssfvar = EDR3RVSSelectionFunctionVar()

                if not (hasattr(self,'Gaia_G') and hasattr(self,'Gaia_RP')):
                    print(rvssfWarning)
                    self.photometry(bands=['Gaia_G', 'Gaia_RP'])

                if not ( hasattr(self,'ra') and hasattr(self,'dec') ):
                    raise ValueError(radecError)

                cc = SkyCoord(self.ra,self.dec,unit='deg',frame='icrs')
                self.obsprob = dr2rvssf.query(cc, g=self.Gaia_G.value, 
                                    c=self.Gaia_G.value - self.Gaia_RP.value)
                self.obsprob *= dr2sf.query(cc, (self.Gaia_G.value))

                _make_probcut(probcut)

            else:

                if not ( hasattr(self,'Gaia_GRVS') ):
                    print(GRVSWarning)

                    self.photometry(bands=['Gaia_GRVS'])

                # G_RVS < 12 is the nominal faint magnitude limit of 
                # the Gaia DR2 radial velocity catalogue
                idx = (self.Gaia_GRVS < 12)
                self.subsample(np.where(idx)[0])

            if self.size > 0:
                self.set_Gaia_release('DR2')
                self.get_Gaia_errors()

    elif (cut == 'Gaia_DR2'):
        #Selects stars visible in the Gaia DR2 source catalogue.
        #NOTE: Not guaranteed that these stars would have astrometry,
        # Gaia scannning law will be used in the future to update this

        if not ( hasattr(self,'Gaia_G') ):
            print(GWarning)
            self.photometry(bands=['Gaia_G'])

        # G < 20.7 is the nominal faint-end magnitude limit of 
        # the Gaia source catalogue
        idx = (self.Gaia_G < 20.7)
        self.subsample(np.where(idx)[0])

        self.set_Gaia_release('DR2')
        self.get_Gaia_errors()

    elif (cut == 'Gaia_6D_EDR3'):
        # Vvisible in the radial velocity catalogue of Gaia EDR3
        # REMINDER that EDR3 did not contain updated radial velocities, 
        # so selection function for the RVS catalogue is the same.
        # Astrometric spread function, though, is updated to reflect 
        # longer mission baseline. Errors are reduced.
        # If use_rvs_sf, use DR2 spectroscopic selection function 
        # to select visible stars. Otherwise, uses only RVS band magnitiude

        if not hasattr(self,'T_eff'):
            raise ValueError(TeffError)

        #Only cool stars are assigned a validated radial velocity 
        #in Gaia DR2/EDR3 (Katz+2019)
        idx = (self.T_eff<6900*u.K)
        self.subsample(np.where(idx)[0])

        if self.size>0:
        
            if use_rvs_sf:
                if 'dr2rvssf' not in globals():

                    #Load in radial velocity selection function
                    dr2rvssf = EDR3RVSSelectionFunction()
                    dr2sf = DR2SelectionFunction()

                if not ( hasattr(self,'Gaia_G') and hasattr(self,'Gaia_RP')):
                    print(rvssfWarning)
                    self.photometry(bands=['Gaia_G', 'Gaia_RP'])

                if not ( hasattr(self,'ra') and hasattr(self,'dec') ):
                    raise ValueError(radecError)

                #Position, brightness and colour of each source
                #Query EDR3 6D selection function
                cc = SkyCoord(self.ra,self.dec,unit='deg',frame='icrs')
                self.obsprob = dr2rvssf.query(cc, g = self.Gaia_G.value, 
                            c = self.Gaia_G.value - self.Gaia_RP.value)

                self.obsprob *= dr2sf.query(cc, self.Gaia_G.value)

                _make_probcut(probcut)

            else:

                if not ( hasattr(self,'Gaia_GRVS') ):
                    print(GRVSWarning)

                    self.photometry(dust,bands=['Gaia_GRVS'])

                #G_RVS < 12 is the nominal faint-end magnitude limit of 
                #the Gaia DR2 radial velocity catalogue
                idx = (self.Gaia_GRVS < 12)
                self.subsample(np.where(idx)[0])

            if self.size>0:

                self.set_Gaia_release('EDR3')
                self.get_Gaia_errors()

    elif (cut == 'Gaia_6D_EDR3_Gaiaverse'):
        '''
        #Old Gaia EDR3 cut based on Gaiaverse project. Now superceded by 
        #GaiaUnlimited cut. Code left for posterity
        # Visible in the radial velocity catalogue of Gaia EDR3.
        # Uses the Gaiaverse RVS selection function
        # REMINDER that EDR3 did not contain updated radial velocities, 
        # so selection function for the RVS catalogue is the same.
        # Astrometric spread function, though, is updated to reflect 
        # longer mission baseline. Errors are reduced.
        # If use_rvs_sf, use DR2 spectroscopic selection function 
        # to select visible stars. Otherwise, uses only RVS band magnitiude

        if not hasattr(self,'T_eff'):
            raise ValueError(TeffError)

        #Only cool stars are assigned a validated radial velocity 
        #in Gaia DR2/EDR3 (Katz+2019)
        idx = (self.T_eff<6900*u.K)
        self.subsample(np.where(idx)[0])

        if self.size>0:
            if use_rvs_sf:
                if 'dr2rvssf' not in globals():

                    #Load in radial velocity selection function
                    dr2rvssf = CoGV.subset_sf(map_fname='rvs_cogv.h5',\
                            nside=32, basis_options= \
                            {'needlet':'chisquare','p':1.0, \
                            'wavelet_tol':1e-2}, \
                            spherical_basis_directory='SphericalBasis')  

                if not ( hasattr(self,'Gaia_G') and hasattr(self,'Gaia_RP')):
                    print(rvssfWarning)
                    self.photometry(bands=['Gaia_G', 'Gaia_RP'])

                if not ( hasattr(self,'ra') and hasattr(self,'dec') ):
                    raise ValueError(radecError)

                #Position, brightness and colour of each source
                rsource = rSource(self.ra,self.dec,frame='icrs',
                        photometry={'gaia_g':self.Gaia_G,
                        'gaia_g_gaia_rp':self.Gaia_G - self.Gaia_RP})
            
                #Get the probability for observing each star in DR2 6D
                self.obsprob = dr2rvssf(rsource)

                #cc = SkyCoord(self.ra,self.dec,unit='deg',frame='icrs')
                #self.obsprob = dr3rvssf.query(cc, g = self.Gaia_G, 
                #                            c = self.Gaia_G - self.Gaia_RP)

                #self.obsprob *= dr2sf.query(cc, (self.Gaia_G.value))
                #self.obsvar = dr2rvssfvar.queryvar(cc, g=self.Gaia_G, 
                #                            c=self.Gaia_G - self.Gaia_RP)

                #Retrieve Gaiaunlimited k (number of stars with vrad)
                #and Gaiaunlimited n (total number of stars)
                #self.k = dr2rvssfvar.queryk(cc, g=self.Gaia_G, 
                #                            c=self.Gaia_G - self.Gaia_RP)
                #self.n = dr2rvssfvar.queryn(cc, g=self.Gaia_G, 
                #                            c=self.Gaia_G - self.Gaia_RP)

                # Take only mock HVSs in Gaia bins with more than zero
                #stars with a measured radial velocity
                #idx = (self.k>0)
                #self.subsample(np.where(idx)[0])

                #Designate a star as appearing in Gaia DR2 6D if a 
                #randomly drawn number in [0,1] is less than obsprob  
                ur  = np.random.uniform(0,1,len(self.obsprob))
                idx = (ur<=self.obsprob)
                self.subsample(np.where(idx)[0])

            else:

                if not ( hasattr(self,'Gaia_GRVS') ):
                    print(MagWarning)

                    self.photometry(dust,bands=['Gaia_GRVS'])

                #G_RVS < 12 is the nominal faint-end magnitude limit of 
                #the Gaia DR2 radial velocity catalogue
                idx = (self.Gaia_GRVS < 12)
                self.subsample(np.where(idx)[0])

            if self.size>0:
                    self.set_Gaia_release('EDR3')
                    set.get_Gaia_errors()
        '''

    elif  (cut == 'Gaia_EDR3'):
        #Selects stars visible in the Gaia EDR3 source catalogue
    
        if not ( hasattr(self,'Gaia_G') ):
            print(GWarning)
            self.photometry(bands=['Gaia_G'])

        #Astrometry is identical to full DR3
        self.subsample('Gaia_DR3')

        self.set_Gaia_release('EDR3')
        self.get_Gaia_errors()

    elif (cut == 'Gaia_6D_DR3'):

        # Visible in the radial velocity catalogue of Gaia DR3
        #Uses the Gaiaunlimited selection function

        if not hasattr(self,'T_eff'):
            raise ValueError(TeffError)

        #Only cool stars are assigned a validated radial velocity 
        #in Gaia DR2/EDR3 (Katz+2019)
        idx = (self.T_eff<14500*u.K)
        self.subsample(np.where(idx)[0])

        idx = np.isnan(self.Gaia_G)
        self.subsample(np.where(~idx)[0])

        if self.size>0:
            if use_rvs_sf:
                if 'dr3rvssf' not in globals():
                
                    #Retrieve selection function
                    dr3rvssf = DR3RVSSelectionFunction()
                    dr3sf = DR3SelectionFunctionTCG()

                    #from .utils.varmap import DR3RVSSelectionFunctionVar
                    #dr3rvssfvar = DR3RVSSelectionFunctionVar()

                if not ( hasattr(self,'Gaia_G') and hasattr(self,'Gaia_RP')):
                    print(rvssfWarning)
                    self.photometry(bands=['Gaia_G', 'Gaia_RP'])

                if not ( hasattr(self,'ra') and hasattr(self,'dec') ):
                    raise ValueError(radecError)

                cc = SkyCoord(self.ra,self.dec,unit='deg',frame='icrs')

                self.obsprob = dr3rvssf.query(cc, g = self.Gaia_G.value, 
                            c = self.Gaia_G.value - self.Gaia_RP.value)

                self.obsprob *= dr3sf.query(cc, (self.Gaia_G.value))
                #self.obsvar = dr3rvssfvar.queryvar(cc, g=self.Gaia_G.value, 
                            #c=self.Gaia_G.value - self.Gaia_RP.value)

                #Retrieve Gaiaunlimited k (number of stars with vad)
                #and Gaiaunlimited n (total number of stars)
                #self.k = dr3rvssfvar.queryk(cc, g=self.Gaia_G.value, 
                #            c=self.Gaia_G.value - self.Gaia_RP.value)
                #self.n = dr3rvssfvar.queryn(cc, g=self.Gaia_G.value, 
                #            c=self.Gaia_G.value - self.Gaia_RP.value)

                ipix = coord2healpix(cc, "icrs", 2**5, nest=True)
                ipix = xr.DataArray(np.atleast_1d(ipix))
                d = {}

                d['g'] = xr.DataArray(np.atleast_1d(self.Gaia_G.value))
                d['c'] = xr.DataArray(np.atleast_1d(self.Gaia_G.value - self.Gaia_RP.value))
                d["method"] = "nearest"
                d["kwargs"] = dict(fill_value=None)
                #outn = dr3rvssf.ds["n"].interp(ipix=ipix, **d).to_numpy()
                outk = dr3rvssf.ds["k"].interp(ipix=ipix, **d).to_numpy()

                if len(cc.shape) == 0:
                    #outn = outn.squeeze()
                    outk = outk.squeeze()

                # Take only mock HVSs in Gaia bins with more than zero
                #stars with a measured radial velocity
                idx = (outk>0) 
                self.subsample(np.where(idx)[0])

                _make_probcut(probcut)
            else:

                if not ( hasattr(self,'Gaia_GRVS') ):
                    print(GRVSWarning)

                    self.photometry(dust,bands=['Gaia_GRVS'])

                #Magnitude limit for DR3 radial velocity catalogue is 14 
                #for cool stars, 12 for hotter stars up to 14500 K 
                #(Sartoretti+2022)
                idx1 = (self.Gaia_GRVS<14)
                idx2 = (self.T_eff<6900*u.K) | ( (self.Gaia_GRVS<12) & \
                                        (self.T_eff<=14500*u.K) )
                idx = idx1*idx2
                self.subsample(np.where(idx)[0])

            if self.size>0:

                self.set_Gaia_release('DR3')
                self.get_Gaia_errors()

    elif (cut == 'Gaia_DR3'):
        #Selects stars visible in the Gaia DR3 astrometric catalogue

        if not ( hasattr(self,'Gaia_G') ):
            print(GWarning)
            self.photometry(bands=['Gaia_G'])

        # G < 20.7 is the nominal faint-end magnitude limit of 
        # the Gaia source catalogue
        idx = (self.Gaia_G < 20.7)
        self.subsample(np.where(idx)[0])

        self.set_Gaia_release('DR3')
        self.get_Gaia_errors()

    elif (cut == 'Gaia_6D_DR4'):
        #Selects stars visible in the radial velocity catalogue of Gaia DR4
        #REMINDER that DR4 spectroscopic is not available, 
        #this cut uses temperature and magnitude.

        if use_rvs_sf:
            print('WARNING: DR4 spectroscopic selection function not'\
                    ' available. ')
            print('Continuing with magnitudes alone.')

        if not hasattr(self,'T_eff'):
            raise ValueError(TeffError)

        if not ( hasattr(self,'Gaia_GRVS') ):
            print(GRVSWarning)
            self.photometry(bands=['Gaia_GRVS'])

        if self.size>0:
            # Magnitude limit of the RVS spectrometer is 16.2 (Katz+2019). 
            # For hot stars the limit is around 14
            idx1 = (self.Gaia_GRVS<16.2)
            idx2 = (self.T_eff<6900*u.K) | (self.Gaia_GRVS<14)
            idx  = idx1*idx2

            self.subsample(np.where(idx)[0])

            if self.size>0:
                self.set_Gaia_release('DR4')
                self.get_Gaia_errors()

    elif (cut == 'Gaia_DR4'):
        #Selects stars visible in the Gaia DR4 source catalogue

        if not ( hasattr(self,'Gaia_G') ):
            print(GWarning)
            self.photometry(bands=['Gaia_G'])

        # G < 20.7 is the nominal faint-end magnitude limit of 
        # the Gaia source catalogue
        idx = (self.Gaia_G < 20.7)
        self.subsample(np.where(idx)[0])

        self.set_Gaia_release('DR4')
        self.get_Gaia_errors()

    elif (cut == 'Gaia_DR5'):
        #Selects stars visible in the Gaia DR4 source catalogue

        if not ( hasattr(self,'Gaia_G') ):
            print(GWarning)
            self.photometry(bands=['Gaia_G'])

        # G < 20.7 is the nominal faint-end magnitude limit of 
        # the Gaia source catalogue
        idx = (self.Gaia_G < 20.7)
        self.subsample(np.where(idx)[0])

        self.set_Gaia_release('DR5')
        self.get_Gaia_errors()

    elif (cut == 'Gaia_6D_DR5'):
        #Selects stars visible in the radial velocity catalogue of Gaia DR4
        #REMINDER that DR5 spectroscopic is not available, 
        #this cut uses temperature and magnitude.

        if use_rvs_sf:
            print('WARNING: DR5 spectroscopic selection function not'\
                ' available. ')
            print('Continuing with magnitudes alone.')

        if not hasattr(self,'T_eff'):
            raise ValueError(TeffError)

        if not ( hasattr(self,'Gaia_GRVS') ):
            print(GRVSWarning)
            self.photometry(bands=['Gaia_GRVS'])

        # Magnitude limit of the Gaia RVS spectrometer is 16.2 (Katz+2019). 
        # For hot stars the limit is around 14
        idx1 = (self.Gaia_GRVS<16.2)
        idx2 = (self.T_eff<6900*u.K) | (self.Gaia_GRVS<14)
        idx  = idx1*idx2
        self.subsample(np.where(idx)[0])

        if self.size>0:

            self.set_Gaia_release('DR5')
            self.get_Gaia_errors()

    elif (cut == 'S5'):

        par = u.mas / self.dist.value
        idx7 = ( (par - 3*self.e_par.to(u.mas)) < 0.2*u.mas )

        if not ( hasattr(self,'Gaia_G') and hasattr(self,'DECam_g') and 
                    hasattr(self, 'DECam_r') ):
            print('Gaia G band and DECam g and r band apparent magnitudes must be computed to select stars within the S5 foot print. Calculating...')
            self.photometry(bands=['Gaia_G', 'DECam_g', 'DECam_r'])

        idx8 = (self.Gaia_G<20)

        gr = self.DECam_g - self.DECam_r

        idx1 = (self.DECam_g>15) & (self.DECam_g<19.5)
        idx2 = (gr>-0.4) & (gr<0.1)                   

        idx = idx1*idx2*idx7*idx8

        self.subsample(np.where(idx)[0])

        if self.size>0:
            S5foot = importlib.resources.files(speedystar).joinpath(
                        'utils/S5_selection.txt')

            fieldra, fielddec, fieldlim = np.loadtxt(S5foot, unpack=True,
                                            skiprows=1,usecols=(1,2,3))

            c = SkyCoord(self.ra,self.dec,unit='deg',frame='icrs')
            
            fieldmin = 180.*np.ones(len(self.ra))*u.deg

            for i in range(len(fieldra)):
                    
                cfield = SkyCoord(fieldra[i],fielddec[i],
                                unit='deg',frame='icrs')
            
                fieldsep = c.separation(cfield).degree * u.deg
                fieldmin = np.minimum(fieldmin,fieldsep)

            idx = (fieldmin < 1.*u.deg)

            self.subsample(np.where(idx)[0])

    elif (type(cut) is str):
        print(cut)
        print('cut doesnt exist')

    elif(type(cut) is int):
        if cut >= self.size:
            print("Warning: Attempting to cut to a number of stars larger than the sample size. Continuing with the full sample.")
        else:
            idx_e = np.random.choice(np.arange(int(self.size)), 
                                cut, replace=False)
            for varname in namelist:
                if varname=='cov':
                    setattr(self, varname, getattr(self, varname)[:,:,idx_e])
                else:
                    try:
                        setattr(self, varname, getattr(self, varname)[idx_e])
                    except:
                        pass

            self.size = cut

    elif('gaiaunlimited.selectionfunctions.subsample.SubsampleSelectionFunction' in str(type(cut))):

        from gaiaunlimited.utils import get_healpix_centers, coord2healpix

        if 'dr3sf' not in globals():
            from gaiaunlimited.selectionfunctions import DR3SelectionFunctionTCG
            
            #Retrieve selection function
            dr3sf = DR3SelectionFunctionTCG()

        cc = SkyCoord(self.ra,self.dec,unit='deg',frame='icrs')

        self.obsprob, self.obsvar = cut.query( cc, 
                                phot_g_mean_mag_ = self.Gaia_G.value, 
                                g_rp_ = (self.Gaia_G - self.Gaia_RP).value, return_variance = True ,fill_nan = False)

        self.obsprob *= dr3sf.query( cc, (self.Gaia_G.value))

        ipix = coord2healpix(cc, "icrs", 2**5, nest=True)
        ipix = xr.DataArray(np.atleast_1d(ipix))
        d = {}

        d['phot_g_mean_mag_'] = xr.DataArray(np.atleast_1d(self.Gaia_G.value))
        d['g_rp_'] = xr.DataArray(np.atleast_1d(self.Gaia_G.value - self.Gaia_RP.value))
        d["method"] = "nearest"
        d["kwargs"] = dict(fill_value=None)
        outk = cut.ds["k"].interp(ipix=ipix, **d).to_numpy()

        if len(cc.shape) == 0:
            outk = outk.squeeze()

        # Take only mock HVSs in Gaia bins with more than zero
        #stars which satisfy the selection
        idx = (outk>0) 
        #self.subsample(np.where(idx)[0])

        _make_probcut(probcut)
    else:
        print('Cut not understood. No cut applied.')
