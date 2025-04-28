from astropy import units as u
import astropy
from astropy.table import Table, Column
from astropy.io import fits
import numpy as np

#@save
def save(self, path, verbose=False):
    '''
    Saves the sample in a FITS file to be grabbed later.
    ALL attributes which are arrays of length self.size are saved.
    See docstring of zippystar.starsample for list of common attributes
    Some metavariables saved as well.

    Parameters
    ----------
    path : str
        Path to the output fits file
    verbose : bool
        Whether or not to print warnings and messages
    '''

    import warnings
    warnings.filterwarnings('ignore',module='astropy.io')

    if( (self.size==0) and (verbose) ):
        print('No stars exist in sample. Saving to file anyways.')

    datalist = []
    namelist = []

    #Some metavariables are saved
    meta_var = {}

    #Every attribute which is a numpy/astropy array of length
    #self.size is saved to file
    for name in vars(self).keys():

        #Attributes are saved...
        if isinstance(vars(self)[name],astropy.units.quantity.Quantity):

            if isinstance(vars(self)[name].value,np.ndarray):

                #...if 1D length-self.size quantity array
                if vars(self)[name].shape == (self.size,):
                    datalist.append(getattr(self,name))
                    namelist.append(name)

                #...as metavariables if 1D length-1 quantity array
                #This never happens by default, but just in case
                elif vars(self)[name].shape == (1,):
                    meta_var[name] = getattr(self, name).value[0]
                else:
                    if verbose:
                        print('Warning: attribute not saved ' + str(name))

            #...as metavariables if single-value quantity
            elif ( (isinstance(vars(self)[name].value,float)) or 
                (isinstance(vars(self)[name].value,int)) ):
                meta_var[name] = getattr(self, name).value

        elif isinstance(vars(self)[name],np.ndarray):

            #...if 1D length-self.size numpy array
            if vars(self)[name].shape == (self.size,):
                datalist.append(getattr(self,name))
                namelist.append(name)

            #...as metavariables if 1D length-1 numpy array
            #This never happens by default, but just in case
            elif vars(self)[name].shape == (1,):
                meta_var[name] = getattr(self, name)[0]
            else:
                if verbose:
                    print('Warning: attribute not saved ' + str(name))

        #...as metavariables if string, bool, int, or float
        elif ( (isinstance(vars(self)[name],str)) or
            (isinstance(vars(self)[name],bool)) or
            (isinstance(vars(self)[name],int)) or
            (isinstance(vars(self)[name],float)) ):
            meta_var[name] = getattr(self, name)

        elif vars(self)[name] is None:
            pass

        else:
            if verbose:
                print('Warning: attribute not saved ' + str(name))

    #Recalculate the size in case it has changed at some point
    meta_var['size'] = len(datalist[0])

    data_table = Table(data=datalist, names=namelist, meta=meta_var)

    hdu = fits.BinTableHDU(data_table)

    #Add column and metadata descriptions to the header, if available
    for i, col in enumerate(data_table.columns.values(),start=1):
        if col.name in self.__attrs__:
            hdu.header[f'TCOMM{i}'] = self.__attrs__[col.name]
        else:
            hdu.header[f'TCOMM{i}'] = 'Description not available'
    
    for key, value in meta_var.items():
        if key in self.__attrs__:
            desc = self.__attrs__[key]
            hdu.header[key] = (value, desc)
        else:
            hdu.header[key] = (value, 'Description not available')

    #Save the file
    hdu.writeto(path, overwrite=True)


#@load
def _load(self, path):
    '''
        Loads a HVS sample from a fits table.
        Creates a starsample object with attributes corresponding
        to each column in the fits file.

    Parameters
    --------------

    path: string
        File path to be read from
    '''

    #List of common coloumns to be read in and their associated units.
    #Read columns do NOT need to be explicitly specified here, all columns
    #of the fits file are read. This is simply useful to assign units to
    #common quantities

    default_units = {'r0': u.pc, 'phi0': u.rad, 'theta0': u.rad, 
                    'v0': u.km/u.s, 'vc': u.km/u.s, 'phiv0': u.rad, 
                    'thetav0': u.rad, 'm': u.solMass, 'tage':u.Myr,
                    'tflight': u.Myr, 'a': u.Rsun,'P': u.day, 'ra': u.deg,
                    'dec': u.deg, 'pmra': u.mas/u.yr, 'pmdec': u.mas/u.yr,
                    'dist': u.kpc, 'vlos': u.km/u.s, 'T_eff': u.K,
                    'Lum': u.Lsun, 'Rad': u.Rsun,'par': u.mas,
                    'e_par': 1e-6*u.arcsec, 'e_pmra': u.mas/u.yr,
                    'e_pmdec': u.mas/u.yr, 'e_vlos': u.km/u.s,
                    'GCdist': u.kpc, 'GCv': u.km/u.s, 'e_GCv': u.km/u.s,
                    'GCv_lb': u.km/u.s, 'GCv_ub': u.km/u.s, 
                    'Vesc': u.km/u.s, 'theta': u.rad, 'phi': u.rad, 
                    'x': u.kpc, 'y': u.kpc, 'z': u.kpc, 'vx': u.km/u.s,
                    'vy': u.km/u.s, 'vz': u.km/u.s}
    
    meta_units = {'current_a': u.pc, 'tlb': u.Myr, 'mc': u.Msun, 
                    'm_bh': u.Msun, 'a0': u.pc, 'rho': u.Msun / u.pc**3, 
                    'sigma': u.km/u.s, 'tflightmax': u.Myr, 
                    'vmin': u.km/u.s, 'dt': u.Myr, 'm_range': u.Msun, 'v_range':u.km/u.s, 'eta': u.yr**-1, 'zo':u.kpc, 
                    'ro': u.kpc, 'vo': u.km/u.s}

    namelist = list(default_units.keys())
    meta_names = list(meta_units.keys())

    #Read in data
    data_table = Table.read(path)

    #Read in metadata
    #Ignore TCOMM keywords since these are just the column descriptions, 
    #not actual metadata
    data_table.meta = {k.lower(): v for k, v in data_table.meta.items() if not k.lower().startswith('tcomm')}

    #Assign metadata as attributes. Given them units if known
    for key in data_table.meta.keys():
        if key in meta_names:
            setattr(self, key, data_table.meta[key]*meta_units[key])
        else:
            setattr(self, key, data_table.meta[key])

    self.size = len(data_table)

    #Assign columns as attributes
    for colname in data_table.colnames:
        if colname in namelist:
            if data_table[colname].unit is None:
                setattr(self, colname, data_table[colname].quantity \
                            * default_units[colname])
            else:
                setattr(self, colname, data_table[colname].quantity)
        else:
            setattr(self, colname, data_table[colname].quantity)

#@loadExt
#Probably out of date, ignore
def _loadExt(self, path, ejmodel='Contigiani2018',dt=0.01*u.Myr):
        '''
            Loads an external HVS sample from external source (e.g., from literature)

        Parameters
        ----------
            path: str
                Path to catalog
            ejmodel = str
                Suspected ejection model generating the sample. Not sure if this would do anything right now if only the likelihood method is being used
            dt = float
                Timestep to be used for the back-propagation

            See self.likelihood() for other parameters

        '''

        from astropy.coordinates import SkyCoord
        from astropy.table import Table

        #namelist = ['r0', 'phi0', 'theta0', 'v0', 'phiv0', 'thetav0', 'm', 'tage', 'tflight', 'ra', 'dec', 'pmra',
        #            'pmdec', 'dist', 'vlos', 'GRVS', 'V', 'G', 'e_par', 'e_pmra', 'e_pmdec', 'GCdist', 'GCv']

        data_table = Table.read(path)

        #Manually set variables that would normally be in metadata
        self.ejmodel_name = ejmodel
        self.dt = dt
        #self.cattype = 2
        self.size = len(data_table)

        setattr(self,'m',data_table['M']*u.solMass)
        setattr(self,'pmra',data_table['pmra_Gaia']*u.mas/u.yr)
        setattr(self,'pmdec',data_table['pmdec_Gaia']*u.mas/u.yr)
        setattr(self,'vlos',data_table['vrad']*u.km/u.second)
        setattr(self,'dist',data_table['d']*u.kpc)
        setattr(self,'tage',data_table['tage']*u.Myr)
        setattr(self,'ID',data_table['ID'])
        setattr(self,'e_pmra',data_table['err_pmra_Gaia']*u.mas/u.yr)
        setattr(self,'e_pmdec',data_table['err_pmdec_Gaia']*u.mas/u.yr)
        setattr(self,'e_dist',data_table['d_errhi']*u.kpc)
        setattr(self,'e_vlos',data_table['vrad_errhi']*u.km/u.second)
        setattr(self,'ra',data_table['ra']*u.degree)
        setattr(self,'dec',data_table['dec']*u.degree)


        #self.pmra[1] = -0.175*u.mas/u.yr
        #setattr(self,'pmra[1]',-0.175*u.mas/u.yr)
        #setattr(self,'pmdec[1]',-0.719*u.mas/u.yr)
        #setattr(self,'err_pmra[1]',0.316*u.mas/u.yr)
        #setattr(self,'err_pmdec[1]',0.287*u.mas/u.yr)

        #Read in ra, dec in hhmmss.ss/DDmmss.ss, convert to degrees
        #ratmp = data_table['RA']
        #dectmp = data_table['Dec']
        #c = SkyCoord(ratmp,dectmp)
        #setattr(self,'ra',c.ra.value*u.degree)
        #setattr(self,'dec',c.dec.value*u.degree)

        #l = c.galactic.l.value*np.pi/180.0
        #b = c.galactic.b.value*np.pi/180.0

        #p1 = (np.cos(b)**2)*(np.cos(l)**2) + (np.cos(b)**2)*(np.sin(l)**2) + np.sin(b)**2
        #p2 = -16.0*np.cos(b)*np.cos(l)
        #p3 = 64 - data_table['RGC']**2

        #dist = np.zeros(self.size)

        #for i in range(self.size):
        #    dist[i] = max(np.roots([p1[i],p2[i],p3[i]]))

        #DATA
        #i=0
        #for colname in data_table.colnames:
        #    try:
        #        i = namelist.index(colname)
        #        setattr(self, colname, data_table[colname].quantity)
        #        i+=1
        #    except ValueError:
        #        print('Column not recognized: ' + str(colname))
        #        i+=1
        #        continue
