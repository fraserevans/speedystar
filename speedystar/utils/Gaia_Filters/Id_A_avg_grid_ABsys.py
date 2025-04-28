import numpy as np

import sys
#sys.path.append('/net/boorne/data2/marchetti/Functions_to_Import')
from scipy import interpolate

import functions

import astropy
import astropy.units as u

# ================ Input/Output ======================= $

path = './'
#Imports passbands
wlen, Tu = np.loadtxt(path+'u-passband.txt', unpack = True) # wavelength [nm], LSST u passband transmission
wlen, Tg = np.loadtxt(path+'g-passband.txt', unpack = True) # wavelength [nm], LSST g passband transmission
wlen, Tr = np.loadtxt(path+'r-passband.txt', unpack = True) # wavelength [nm], LSST r passband transmission
wlenVega, FVega =   np.loadtxt(path+'AB_spectrum.txt', unpack = True) # Vega Flux

F = interpolate.interp1d(wlenVega, FVega)
FVega = F(wlen)

uVega = 0.03 # Zero magntidue for a Vega-like star in the u band
gVega = 0.03 # Zero magntidue for a Vega-like star in the g band
rVega = 0.03 # Zero magntidue for a Vega-like star in the r band

wlen = wlen * u.nm
wlenVega = wlenVega * u.nm

MH = 0.5

out_file = open(path+'Id_A_avg_grid_MH_'+np.str(MH)+'_LSSTugr_Vega.txt',"w")

# =================================================== # 

if MH == -0.5:
        Id_max = 459
if MH == 0.:
        Id_max = 460
if MH == 0.5:
        Id_max = 456

Id = np.arange(1,Id_max,1) # Identifier for the Basel spectral library (metallicity = solar)
Av = np.linspace(0.,20.,30) # Average extinction

#For each spectrum
for i_I in range(0, len(Id)):
        #For each extinction
        for i_A in range(0, len(Av)):

                #Fetch Spectrum, with distance correction
                wlenf, Flux = functions.get_spectrum(Id[i_I], MH, '/data1/Cats/') #  wavelenght [nm],  Flux [photons/s/m/m/nm]

                #Wavelength-dependent extinction for the average extinction
                #Fetched using Cardelli extinction law
                A_l = functions.extinction(Av[i_A],wlen) # Extinction at the different wavelengths, [mag]

                #Actual magnitudes in each band for each extincted spectrum
                #uMag = functions.get_Magnitude_AB(Flux, wlenf, wlen, A_l, Tu)
                #gMag = functions.get_Magnitude_AB(Flux, wlenf, wlen, A_l, Tg)
                #rMag = functions.get_Magnitude_AB(Flux, wlenf, wlen, A_l, Tr)

                uMag = functions.get_Magnitude(Flux, wlenf, wlen, A_l, FVega, uVega, Tu)
                gMag = functions.get_Magnitude(Flux, wlenf, wlen, A_l, FVega, gVega, Tg)
                rMag = functions.get_Magnitude(Flux, wlenf, wlen, A_l, FVega, rVega, Tr)

                #Writes file with spectrum ID, extinction and GMag, VMag and IcMag
                out_file.write( np.str(Id[i_I]) + "     " + np.str(Av[i_A]) + " " + np.str(uMag) + "    " + np.str(gMag) + "    " + np.str(rMag)  + '\n')

out_file.close()
