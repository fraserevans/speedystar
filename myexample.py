'''
    Simple catalog
'''

import numpy as np
from speedystar import starsample
from speedystar.eject import Hills
from galpy.potential.mwpotentials import McMillan17
from astropy import units as u
import scanninglaw.asf as astrospreadfunc

'''
    Create ejection catalog
'''

# Initialize an ejection model, i.e. how the spatial and velocity 
#distribution of the stars will be sampled
ejectionmodel = Hills()

# Eject a sample of stars from Sgr A*. 
mysample = starsample(ejectionmodel, name='My catalogue')

# Save ejection sample
mysample.save('./cat_ejection.fits')

'''
    Propagate ejection catalogue through the galaxy
'''

# Load ejection sample
mysample = starsample('./cat_ejection.fits')

# Assume a Galactic potential
default_potential = McMillan17

#idx = (mysample.m>1.*u.Msun)
#mysample.subsample(np.where(idx)[0])

#Propagate sample. Change timestep as needed
mysample.propagate(potential = default_potential)

#Save propagated sample
mysample.save('./cat_propagated.fits')

'''
  Calculate the apparent magnitudes of each star in your sample 
  By default gives magnitudes in Gaia and Johnson-Cousins passbands
'''

#Load a pre-existing propagated sample, if needed
mysample = starsample('./cat_propagated.fits')

#Magnitudes are excincted by Milky Way dust along the line of sight. 
#Before first use, dust maps must be downloaded, see docstring of
#speedystar.fetch_dust()
mysample.config_dust('/path/where/large/files/are/stored/chosen_dust_map.h5')
mysample.photometry()

#Save it
mysample.save('./cat_photometry.fits')


'''
  Subsample the catalogue, if needed, to save space or computation time, 
  or select only interesting objects
'''

#Download Gaia astrometric spread function and spectroscopic selection function
#Only need to do this once
#mysample.config_rvssf(path='/data1/VISTA/')
#mysample.config_astrosf(path='/data1/VISTA/')

#For cuts you make often, can hard-code a condition in mysample.subsample()
#Gaia cuts can use the spectroscopic selection function and/or
#astrometric spread function, these also need to be downloaded before first use,
#see speedystar.fetch_astrosf() and speedystar.fetch_rvssf() docstrings.
mysample.subsample('Gaia_6D_DR4',use_ast_sf=False,use_rvs_sf=False)

#Save it
mysample.save('./cat_Gaia_6D_DR4.fits')
