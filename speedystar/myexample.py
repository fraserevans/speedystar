'''
    Simple catalog
'''

import numpy as np
from sample_clone_package import HVSsample
from astropy.table import Table
#from ejection_clone_new import Rossi2017
from _eject import Rossi2017
from utils.dustmap import DustMap
from utils.mwpotential import MWPotential
from galpy.potential.mwpotentials import McMillan17, MWPotential2014
from astropy import units as u
import scanninglaw.asf as astrospreadfunc

#from scanninglaw.config import config
#config['data_dir'] = '/net/alm/data1/Cats/cog_iv/'

#asf = astrospreadfunc.asf() 


'''
    Create ejection catalog
'''

# Initialize an ejection model, i.e. how the spatial and velocity distribution of the stars will be sampled
ejectionmodel = Rossi2017(name_modifier='TEST',kappa=1.7)

# Eject a sample of stars from Sgr A*. You can pass a number n, the TOTAL number of stars the GC has ejected over the last 13.8 Gyr. Your assumed ejection rate is then implied to be n/1.38e10

#Any cuts on ejection parameters (see 'sampler' in ejection_example.py) will cut down number of returned stars

#mysample = HVSsample(ejectionmodel, name='My catalogue', eta=1e-4)

#idx = (mysample.v0>1000*u.km/u.s) & (mysample.m>1.*u.Msun)
#mysample.subsample(np.where(idx)[0])

# Save ejection sample
#mysample.save('./cat_ejection.fits')

'''
    Propagate ejection catalogue through the galaxy
'''

# Load ejection sample
mysample = HVSsample('./cat_ejection.fits')

#mysample.subsample(np.argwhere(mysample.stage>1))


# Assume a potential. Kind of non-trivial to change, check with me first
default_potential = MWPotential2014

#Propagate sample. Change timestep as needed
#mysample.propagate(potential = default_potential, dt=1*u.Myr, threshold = 1e-7,orbit_path='/home/evans/work/HVS/hvs-master/For_Alonso/flightstestfake/') # See documentation
mysample.propagate(potential = default_potential, orbit_path='./testpath/',dt=0.1*u.Myr, threshold = 1e-7) # See documentation

#Get final galactocentric distance and velocity for sample
#mysample.GetFinal()
#mysample.GetVesc(default_potential)

#Can get only unbound stars if you want
#idx = (mysample.GCv>mysample.Vesc)
#mysample.subsample(np.argwhere(mysample.GCv>mysample.Vesc))

#Save propagated sample
mysample.save('./cat_propagated.fits')

#Load a pre-existing propagated sample, if needed
mysample = HVSsample('./cat_propagated.fits')

mysample.backprop(potential=default_potential)


#idx = (mysample.GCv>1000*u.km/u.s) 
#mysample.subsample(np.where(idx)[0])

'''
  Calculate the apparent magnitudes of your sample in the Gaia and Johnson-Cousins passbands
'''

#Magnitudes are excincted by Milky Way dust along the line of sight. Loads MW dust map. Might take a bit
#dust = DustMap('./utils/dust-map-3d.h5')

#mysample.config_dust('/net/alm/data1/DustMaps/combined15/dust-map-3d.h5')

#dust = DustMap('/net/alm/data1/DustMaps/combined15/dust-map-3d.h5')
#Calculate photometry
#mysample.photometry(dust,bands=['Bessell_U', 'Bessell_B', 'Bessell_V', 'Bessell_R', 'Bessell_I', 'Gaia_GRVS', 'Gaia_G', 'Gaia_BP', 'Gaia_RP', 'VISTA_Y', 'VISTA_Z', 'VISTA_J', 'VISTA_H', 'VISTA_K', 'DECam_u', 'DECam_g', 'DECam_r', 'DECam_i', 'DECam_z', 'DECam_Y', 'LSST_u', 'LSST_g', 'LSST_r', 'LSST_i', 'LSST_z', 'LSST_y'],v=True)
#mysample.photometry(dust,bands=['Bessell_U', 'Bessell_V', 'Bessell_I', 'Gaia_G'],v=True)
#mysample.photometry(bands=['DECam_g'])
#mysample.photometry(dust,bands=['VISTA_J', 'VISTA_H', 'VISTA_K'],errors=['e_pmdec', 'e_vlos'],v=True)
#mysample.photometry()

#idx = (mysample.Gaia_G<21) 
#mysample.subsample(np.where(idx)[0])

#mysample.Gmag = mysample.Gaia_G - 5*np.log10(mysample.dist.value*1000/10)
#mysample.col = mysample.Gaia_BP - mysample.Gaia_RP

def main_sq(x):
    return 4.3*x + 0.5

#ms_band = main_sq(mysample.col)
#whereONMS = np.where((ms_band+2>mysample.Gmag)&(ms_band-2<mysample.Gmag))[0]

#Save it

#mysample.get_Punbound()

#mysample.save('./cat_photometry.fits')

'''
  Subsample the catalogue, if needed, to save space or computation time, or select only interesting objects
'''

#e.g. cut down mysample to a smaller one consisting of 5 randomly selected ones
#mysample.subsample(5)

#e.g. select only the stars in sample corresponding to given indices
#mysample.subsample(np.array[1,2,3,...])

#e.g. select only the stars which satisfy a certain condition (e.g. only stars above the escape velocity of the Galaxy)
#mysample.GetVesc(default_potential)
#idx = (mysample.GCv>mysample.Vesc)

#For cuts you make often, can hard-code a condition in mysample.subsample()
#mysample.subsample('Gaia_6D_DR4',use_ast_sf=False,use_rvs_sf=False)

#Save it
#mysample.save('./cat_Gaia_6D_DR4test.fits')
#idx = (mysample.v0>1000*u.km/u.s) & (mysample.m>1.*u.Msun)
#mysample.subsample(1)
