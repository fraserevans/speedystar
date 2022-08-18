__TeffError__ = 'Effective temperatures are required to determine which sources would appear in the Gaia DR2 radial velocity catalogue. Please ensure your sample has a "T_eff" attribute. This should have been added when the ejection sample was created. If your catalogue was loaded externally, T_eff can be calculated using evo_pop() or get_stellar_parameters()'

__rvssfWarning__ = 'Gaia G and G_RP band apparent magnitude must be computed computed to select stars visible in the Gaia radial velocity catalogue. Calculating...'

__rvssfError__ = 'Right ascension and declination of sources must be known to query the Gaia spectroscopic  selection function. Please ensure "ra" and "dec" are attributes of your sample.'

__MagWarning__ = 'Gaia G_RVS band apparent magnitude must be computed to select stars visible in the Gaia radial velocity catalogue. Calculating...'

__asfWarning__ = 'Gaia G band apparent magnitude must be known to compute Gaia astrometric spread function. Calculating...'

__ImportError__ = "One or more required external packages have not been installed. Please see requirements.txt"

#try:
#    from tqdm import tqdm
#    import numpy  as np
#    from astropy import units as u
#    import astropy
#    import scanninglaw.asf as astrospreadfunc
#except ImportError:
#    raise ImportError(__ImportError__)

#from .eject import EjectionModel

#from .utils.mwpotential import PotDiff

#@subsample
def subsample(self, cut=None, use_rvs_sf=False,use_ast_sf=False):

        try:
            from galpy.util.coords import radec_to_lb
            import astropy
            import numpy  as np
            from astropy import units as u
        except ImportError:
            raise ImportError(__ImportError__)

        from scanninglaw.source import Source as aSource
        from .utils.selectionfunctions.source import Source as rSource
        from .utils.selectionfunctions import cog_v as CoGV

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
        use_ast_sf: Boolean
            If True, uses the Gaia DR2 or DR3 astrometric spread function to 
            determine position/parallax/proper motion uncertainties 
            and generate astrometric covariance matrix (see get_Punbound()). 
            ONLY applies if cut=='Gaia_6D_DR2', if cut=='Gaia_6D_EDR3', 
            if cut=='Gaia_6D_DR3' or if cut=='Gaia_6D_DR4'.
            If cut=='Gaia_6D_DR4', mock astrometric errors are determined by 
            scaling the DR3 astrometric errors downwards based on the predicted
            Gaia performance (see  ADD REFERENCE HERE)
        '''
        namelist = []

        for name in vars(self).keys():
            if isinstance(vars(self)[name],astropy.units.quantity.Quantity):
                if isinstance(vars(self)[name].value,np.ndarray):
                    if len(vars(self)[name])==self.size:
                        namelist.append(name)
            elif isinstance(vars(self)[name],np.ndarray):
                if len(vars(self)[name])==self.size:
                    namelist.append(name)

        if(type(cut) is np.ndarray):
                cut.flatten()
                for varname in namelist:
                    try:
                        setattr(self, varname, \
                                getattr(self, varname)[cut].flatten())
                    except:
                        pass
                self.size = cut.size

        elif (cut == 'Gaia_6D_DR2'):
            #Selects stars visible in the radial velocity catalogue of Gaia DR2
            #If use_rvs_sf, use DR2 spectroscopic selection function
            #Otherwise, uses only RVS band magnitiude
            #If use_ast_sf, use DR2 astrometric spread function for errors
            #Otherwise, scales up predicted DR4 errors from PyGaia

            if not hasattr(self,'T_eff'):
                raise ValueError(__TeffError__)

            #Only cool stars are assigned a radial velocity in 
            #Gaia DR2 (Katz+2019)
            idx = (self.T_eff<6900*u.K)
            self.subsample(np.where(idx)[0])

            if self.size > 0:
                if use_rvs_sf:
                    if 'dr2rvssf' not in locals():

                        #from utils.selectionfunctions.config import config
                        #config['data_dir'] = '/net/alm/data1/Cats/cog_iv/'

                        #Load in radial velocity selection function
                        dr2rvssf = CoGV.subset_sf(map_fname='rvs_cogv.h5',
                                    nside=32,basis_options= {'needlet' \
                                    :'chisquare','p':1.0,'wavelet_tol':1e-2},
                                   spherical_basis_directory='SphericalBasis')

                    if not (hasattr(self,'Gaia_G') and hasattr(self,'Gaia_RP')):
                        print(__rvssfWarning__)
                        self.photometry(bands=['Gaia_G', 'Gaia_RP'])

                    if not ( hasattr(self,'ra') and hasattr(self,'dec') ):
                        raise ValueError(__rvssfError__)

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
                        print(__MagWarning__)

                        self.photometry(bands=['Gaia_GRVS'])

                    # G_RVS < 12 is the nominal faint-end magnitude limit of 
                    # the Gaia DR2 radial velocity catalogue
                    idx = (self.Gaia_GRVS < 12)
                    self.subsample(np.where(idx)[0])

                if self.size > 0:
                    if use_ast_sf:
                        if 'dr2astsf' not in locals():
                            #Load in DR2 astrometric spread function
                            dr2astsf = astrospreadfunc.asf() 

                        if not ( hasattr(self,'Gaia_G') ):
                            print(__asfWarning__)
                            self.photometry(bands=['Gaia_G'])

                        if not ( hasattr(self,'ra') and hasattr(self,'dec') ):
                            raise ValueError(__asfError__)

                        #Position and brightness of each star
                        asource = aSource(self.ra,self.dec,frame='icrs',
                                        photometry={'gaia_g':self.Gaia_G})

                        #Calling the astrometric spread function. 
                        # Gives the 5x5 covariance matrix, whose diagonal
                        # elements are the variances of ra/dec/par/pmra/pmdec

                        self.cov = dr2astsf(asource)

                        #assign astrometric error
                        self.e_par   = np.sqrt(self.cov[2,2])*u.mas
                        self.e_pmra  = np.sqrt(self.cov[3,3])*u.mas/u.yr
                        self.e_pmdec = np.sqrt(self.cov[4,4])*u.mas/u.yr 

                    else:

                        #Inflate astrometric errors to go from DR4 to DR2
                        if hasattr(self,'e_par'):
                            self.e_par *= 1.7
                        if hasattr(self,'e_pmra'):
                            self.e_pmra *= 4.5
                        if hasattr(self,'e_pmdec'):
                            self.e_pmdec *= 4.5

                #Inflate PyGaia radial velocity error from DR4 to DR2
                if hasattr(self,'e_vlos'):
                    self.e_vlos *= 1.65

        elif (cut == 'Gaia_DR2'):
            #Selects stars visible in the Gaia DR2 source catalogue.
            #NOTE: Not guaranteed that these stars would have astrometry,
            # Gaia scannning law will be used in the future to update this
            #If use_ast_sf, use DR2 astrometric spread function for errors.
            #Otherwise, scales up predicted DR4 errors from PyGaia

            if not ( hasattr(self,'Gaia_G') ):
                print(__MagWarning__)
                self.photometry(bands=['Gaia_G'])

            # G < 20.7 is the nominal faint-end magnitude limit of 
            # the Gaia source catalogue
            idx = (self.Gaia_G < 20.7)
            self.subsample(np.where(idx)[0])

            if self.size > 0:
                if use_ast_sf:
                    if 'dr2astsf' not in locals():

                        #Load in DR2 astrometric spread function
                        dr2astsf = astrospreadfunc.asf() 

                    if not ( hasattr(self,'ra') and hasattr(self,'dec') ):
                        raise ValueError(__asfError__)

                    #Position and brightness of each star
                    asource = aSource(self.ra,self.dec,frame='icrs',
                                        photometry={'gaia_g':self.Gaia_G})

                    #Calling the astrometric spread function on asource gives 
                    # the 5x5 covariance matrix, whose diagonal elements 
                    # are the variances of ra/dec/par/pmra/pmdec

                    self.cov = dr2astsf(asource)

                    #assign astrometric error
                    self.e_par   = np.sqrt(self.cov[2,2])*u.mas
                    self.e_pmra  = np.sqrt(self.cov[3,3])*u.mas/u.yr
                    self.e_pmdec = np.sqrt(self.cov[4,4])*u.mas/u.yr 

                else:

                    #Inflate PyGaia astrometric errors to go from DR4 to DR2
                    if hasattr(self,'e_par'):
                        self.e_par *= 1.7
                    if hasattr(self,'e_pmra'):
                        self.e_pmra *= 4.5
                    if hasattr(self,'e_pmdec'):
                        self.e_pmdec *= 4.5

                #Inflate PyGaia radial velocity error from DR4 to DR2
                if hasattr(self,'e_vlos'):
                    self.e_vlos *= 1.65

        elif (cut == 'Gaia_6D_EDR3'):
            # Vvisible in the radial velocity catalogue of Gaia EDR3
            # REMINDER that EDR3 did not contain updated radial velocities, 
            # so selection function for the RVS catalogue is the same.
            # Astrometric spread function, though, is updated to reflect 
            # longer mission baseline. Errors are reduced.
            # If use_rvs_sf, use DR2 spectroscopic selection function 
            # to select visible stars. Otherwise, uses only RVS band magnitiude
            # If use_ast_sf, use DR3 astrometric spread function to assign 
            # astrometric errors. Otherwise, scales up predicted 
            # DR4 errors from PyGaia

            if not hasattr(self,'T_eff'):
                raise ValueError(__TeffError__)

            #Only cool stars are assigned a validated radial velocity 
            #in Gaia DR2/EDR3 (Katz+2019)
            idx = (self.T_eff<6900*u.K)
            self.subsample(np.where(idx)[0])

            print(self.size)

            if self.size>0:
                if use_rvs_sf:
                    if 'dr2rvssf' not in locals():

                        #Load in radial velocity selection function
                        dr2rvssf = CoGV.subset_sf(map_fname='rvs_cogv.h5',\
                                    nside=32, basis_options= \
                                    {'needlet':'chisquare','p':1.0, \
                                    'wavelet_tol':1e-2}, \
                                  spherical_basis_directory='SphericalBasis')  

                    if not ( hasattr(self,'Gaia_G') and hasattr(self,'Gaia_RP')):
                        print(__rvssfWarning__)
                        self.photometry(bands=['Gaia_G', 'Gaia_RP'])

                    if not ( hasattr(self,'ra') and hasattr(self,'dec') ):
                        raise ValueError(__rvssfError__)

                    #Position, brightness and colour of each source
                    rsource = rSource(self.ra,self.dec,frame='icrs',
                                photometry={'gaia_g':self.Gaia_G,
                                'gaia_g_gaia_rp':self.Gaia_G - self.Gaia_RP})
                    
                    #Get the probability for observing each star in DR2 6D
                    self.obsprob = dr2rvssf(rsource)

                    #Designate a star as appearing in Gaia DR2 6D if a 
                    #randomly drawn number in [0,1] is less than obsprob  
                    ur  = np.random.uniform(0,1,len(self.obsprob))
                    idx = (ur<=self.obsprob)
                    self.subsample(np.where(idx)[0])

                else:

                    if not ( hasattr(self,'Gaia_GRVS') ):
                        print(__MagWarning__)

                        self.photometry(dust,bands=['Gaia_GRVS'])

                    #G_RVS < 12 is the nominal faint-end magnitude limit of 
                    #the Gaia DR2 radial velocity catalogue
                    idx = (self.Gaia_GRVS < 12)
                    self.subsample(np.where(idx)[0])

                if self.size>0:
                    if use_ast_sf:

                        if 'dr3astsf' not in locals():

                            #Load in DR2 astrometric spread function
                            dr3astsf = astrospreadfunc.asf(version= \
                                                            'dr3_nominal')

                        if not ( hasattr(self,'Gaia_G') ):
                            print(__asfWarning__)

                            self.photometry(bands=['Gaia_G'])

                        if not ( hasattr(self,'ra') and hasattr(self,'dec') ):
                            raise ValueError(__asfError__)

                        #Position and brightness of each star
                        asource = aSource(self.ra,self.dec,frame='icrs', 
                                          photometry={'gaia_g':self.Gaia_G})

                        #Calling the astrometric spread function on asource 
                        #gives the 5x5 covariance matrix, whose diagonal 
                        #elements are the variances of ra/dec/par/pmra/pmdec
                        self.cov = dr3astsf(asource)

                        #assign astrometric error
                        self.e_par   = np.sqrt(self.cov[2,2])*u.mas
                        self.e_pmra  = np.sqrt(self.cov[3,3])*u.mas/u.yr
                        self.e_pmdec = np.sqrt(self.cov[4,4])*u.mas/u.yr

                    else:

                        #Inflate errors to go from DR4 to EDR3
                        if(hasattr(self,'e_par')):
                            self.e_par *= 1.335
                        if(hasattr(self,'e_pmra')):
                            self.e_pmra *= 1.335*1.776
                        if(hasattr(self,'e_pmdec')):
                            self.e_pmdec *= 1.335*1.78

                #Inflate PyGaia radial velocity error from DR4 to EDR3
                if(hasattr(self,'e_vlos')):
                    self.e_vlos *= 1.65

        elif  (cut == 'Gaia_EDR3'):
            #Selects stars visible in the Gaia EDR3 source catalogue

            #Astrometry is identical to full DR3
            self.subsample('Gaia_DR3')
            
            #Radial velocity errors must be scaled back to DR2 levels
            if(hasattr(self,'e_vlos')):
                self.e_vlos *= 1.65/1.33

        elif (cut == 'Gaia_6D_DR3'):
            #Selects stars visible in the radial velocity catalogue of Gaia DR3
            #REMINDER that DR3 spectroscopic is not yet available, this cut 
            #uses temperature and magnitude.
            #If use_ast_sf, use DR3 astrometric spread function 
            #to assign astrometric errors

            if use_rvs_sf:
                print('WARNING: DR3 spectroscopic selection function not yet'\
                        'available. Continuing with magnitudes alone')

            if not hasattr(self,'T_eff'):
                raise ValueError(__TeffError__)

            if not ( hasattr(self,'Gaia_GRVS') ):
                print(__MagWarning__)

                self.photometry(bands=['Gaia_GRVS'])

            #Magnitude limit for DR3 radial velocity catalogue is 14 for cool 
            #stars, 12 for hotter stars up to 14500 K (Sartoretti+2022)
            idx1 = (self.Gaia_GRVS<14)
            idx2 = (self.T_eff<6900*u.K) | ( (self.Gaia_GRVS<12) & \
                                            (self.T_eff<=14500*u.K) )
            idx = idx1*idx2
            self.subsample(np.where(idx)[0])

            if self.size>0:
                if use_ast_sf:

                    if 'dr3astsf' not in locals():

                        #Load in DR2 astrometric spread function
                        dr3astsf = astrospreadfunc.asf(version='dr3_nominal')

                    if not ( hasattr(self,'Gaia_G') ):
                        print(__asfWarning__)
                        self.photometry(bands=['Gaia_G'])

                    if not ( hasattr(self,'ra') and hasattr(self,'dec') ):
                        raise ValueError(__asfError__)

                    #Position and brightness of each star
                    asource = aSource(self.ra,self.dec,frame='icrs',
                                        photometry={'gaia_g':self.Gaia_G})

                    #Calling the astrometric spread function on asource gives 
                    # the 5x5 covariance matrix, whose diagonal elements 
                    #are the variances of ra/dec/par/pmra/pmdec
                    self.cov = dr3astsf(asource)

                    #assign astrometric error
                    self.e_par   = np.sqrt(self.cov[2,2])*u.mas
                    self.e_pmra  = np.sqrt(self.cov[3,3])*u.mas/u.yr
                    self.e_pmdec = np.sqrt(self.cov[4,4])*u.mas/u.yr

                else:

                    #Inflate PyGaia astrometric errors to go from DR4 to DR3
                    if(hasattr(self,'e_par')):
                        self.e_par *= 1.335
                    if(hasattr(self,'e_pmra')):
                        self.e_pmra *= 1.335*1.776
                    if(hasattr(self,'e_pmdec')):
                        self.e_pmdec *= 1.335*1.78

            #Inflate PyGaia radial velocity error from DR4 to DR3
            if(hasattr(self,'e_vlos')):
                self.e_vlos *= 1.33


        elif (cut == 'Gaia_DR3'):
            #Selects stars visible in the Gaia DR3 astrometric catalogue
            #If use_ast_sf, use DR3 astrometric spread function for errors
            #Otherwise, scales up predicted DR4 errors from PyGaia

            if not ( hasattr(self,'Gaia_G') ):
                print(__MagWarning__)
                self.photometry(bands=['Gaia_G'])

            # G < 20.7 is the nominal faint-end magnitude limit of 
            # the Gaia source catalogue
            idx = (self.Gaia_G < 20.7)
            self.subsample(np.where(idx)[0])

            if self.size > 0:
                if use_ast_sf:
                    if 'dr3astsf' not in locals():

                        #Load in DR2 astrometric spread function
                        dr3astsf = astrospreadfunc.asf(version='dr3_nominal') 

                    if not ( hasattr(self,'ra') and hasattr(self,'dec') ):
                        raise ValueError(__asfError__)

                    #Position and brightness of each star
                    asource = aSource(self.ra,self.dec,frame='icrs',
                                        photometry={'gaia_g':self.Gaia_G})

                    #Calling the astrometric spread function on asource gives 
                    # the 5x5 covariance matrix, whose diagonal elements 
                    # are the variances of ra/dec/par/pmra/pmdec

                    self.cov = dr3astsf(asource)

                    #assign astrometric error
                    self.e_par   = np.sqrt(self.cov[2,2])*u.mas
                    self.e_pmra  = np.sqrt(self.cov[3,3])*u.mas/u.yr
                    self.e_pmdec = np.sqrt(self.cov[4,4])*u.mas/u.yr 

                else:

                    #Inflate PyGaia astrometric errors to go from DR4 to DR3
                    if(hasattr(self,'e_par')):
                        self.e_par *= 1.335
                    if(hasattr(self,'e_pmra')):
                        self.e_pmra *= 1.335*1.776
                    if(hasattr(self,'e_pmdec')):
                        self.e_pmdec *= 1.335*1.78


                #Inflate PyGaia radial velocity error from DR4 to DR2
                if hasattr(self,'e_vlos'):
                    self.e_vlos *= 1.33

        elif (cut == 'Gaia_6D_DR4'):
            #Selects stars visible in the radial velocity catalogue of Gaia DR4
            #REMINDER that DR4 spectroscopic is not available, 
            #this cut uses temperature and magnitude.
            #If use_ast_sf, use DR3 astrometric spread function and scaled 
            #errors down to assign astrometric errors

            if use_rvs_sf:
                print('WARNING: DR4 spectroscopic selection function not'\
                        ' available. Continuing with magnitudes alone')

            if not hasattr(self,'T_eff'):
                raise ValueError(__TeffError__)

            if not ( hasattr(self,'Gaia_GRVS') ):
                print(__MagWarning__)
                self.photometry(bands=['Gaia_GRVS'])

            # Magnitude limit of the Gaia RVS spectrometer is 16.2 (Katz+2019). 
            # For hot stars the limit is around 14
            idx1 = (self.Gaia_GRVS<16.2)
            idx2 = (self.T_eff<6900*u.K) | (self.Gaia_GRVS<14)
            idx  = idx1*idx2
            self.subsample(np.where(idx)[0])

            if self.size>0:
                if use_ast_sf:

                    if 'dr3astsf' not in locals():

                        #Load in DR2 astrometric spread function
                        dr3astsf = astrospreadfunc.asf(version='dr3_nominal')

                    if not ( hasattr(self,'Gaia_G') ):
                        print(__asfWarning__)
                        self.photometry(bands=['Gaia_G'])

                    if not ( hasattr(self,'ra') and hasattr(self,'dec') ):
                        raise ValueError(__asfError__)

                    #Position and brightness of each star
                    asource = aSource(self.ra,self.dec,frame='icrs',
                                    photometry={'gaia_g':self.Gaia_G})

                    #Calling the DR3 astrometric spread function on asource 
                    #gives the 5x5 covariance matrix, whose diagonal elements 
                    #are the variances of ra/dec/par/pmra/pmdec. Scale down 
                    #errors based on predicted Gaia performance. Correlations 
                    #among errors are assumed to be unchanged from DR3 to DR4
                    self.cov         = dr3astsf(asource)
                    self.cov[2,2,:] /= 1.335
                    self.cov[3,3,:] /= 1.335*1.776
                    self.cov[4,4,:] /= 1.335*1.78

                    #assign astrometric error
                    self.e_par   = np.sqrt(self.cov[2,2])*u.mas
                    self.e_pmra  = np.sqrt(self.cov[3,3])*u.mas/u.yr
                    self.e_pmdec = np.sqrt(self.cov[4,4])*u.mas/u.yr

                else:
                    pass

        elif (cut == 'Gaia_DR4'):
            #Selects stars visible in the Gaia DR4 source catalogue
            #If use_ast_sf, use DR3 astrometric spread function for errors
            #Otherwise, uses PyGaia

            if not ( hasattr(self,'Gaia_G') ):
                print(__asfWarning__)
                self.photometry(bands=['Gaia_G'])

            # G < 20.7 is the nominal faint-end magnitude limit of 
            # the Gaia source catalogue
            idx = (self.Gaia_G < 20.7)
            self.subsample(np.where(idx)[0])

            if self.size > 0:
                if use_ast_sf:
                    if 'dr3astsf' not in locals():

                        #Load in DR2 astrometric spread function
                        dr3astsf = astrospreadfunc.asf(version='dr3_nominal') 

                    if not ( hasattr(self,'ra') and hasattr(self,'dec') ):
                        raise ValueError(__asfError__)

                    #Position and brightness of each star
                    asource = aSource(self.ra,self.dec,frame='icrs',
                                        photometry={'gaia_g':self.Gaia_G})

                    #Calling the astrometric spread function on asource gives 
                    # the 5x5 covariance matrix, whose diagonal elements 
                    # are the variances of ra/dec/par/pmra/pmdec
                    self.cov = dr3astsf(asource)

                    #Calling the DR3 astrometric spread function on asource 
                    #gives the 5x5 covariance matrix, whose diagonal elements 
                    #are the variances of ra/dec/par/pmra/pmdec. 
                    #Scale down DR3 errors based on predicted Gaia performance.
                    #Correlations among errors are assumed to be unchanged 
                    #from DR3 to DR4
                    self.cov[2,2,:] /= 1.335
                    self.cov[3,3,:] /= 1.335*1.776
                    self.cov[4,4,:] /= 1.335*1.78

                    #assign astrometric error
                    self.e_par   = np.sqrt(self.cov[2,2])*u.mas
                    self.e_pmra  = np.sqrt(self.cov[3,3])*u.mas/u.yr
                    self.e_pmdec = np.sqrt(self.cov[4,4])*u.mas/u.yr 

        elif (type(cut) is str):
            print(cut)
            print('cut doesnt exist')

        if(type(cut) is int):
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
