import numpy as np

import sys
#sys.path.append('/net/boorne/data2/marchetti/Functions_to_Import')

import functions

import astropy
import astropy.units as u

# ================ Input/Output ======================= $

path = './'
#Imports passbands
wlen, TG = np.loadtxt(path+'G-passband-EDR3.txt', unpack = True) # wavelength [nm], Gaia G passband transmission
wlen, TBP = np.loadtxt(path+'BP-passband-EDR3.txt', unpack = True) # wavelength [nm], Gaia G passband transmission
wlen, TRP = np.loadtxt(path+'RP-passband-EDR3.txt', unpack = True) # wavelength [nm], Gaia G passband transmission
wlen, TV = np.loadtxt(path+'V-passband-new.txt', unpack = True) # wavelength [nm], Johnson-Cousins V Filter transmission
wlen, TIc = np.loadtxt(path+'I-passband-new.txt', unpack = True) # wavelength [nm], Johnson-Cousins Ic Filter transmission
wlen, FVega =   np.loadtxt(path+'Vega_spectrum.txt', unpack = True) # Vega Flux

wlen = wlen * u.nm

BPVega = 0.03 # Zero magntidue for a Vega-like star in the G band
GVega = 0.03 # Zero magntidue for a Vega-like star in the G band
BPVega = 0.03 # Zero magntidue for a Vega-like star in the G band
RPVega = 0.03 # Zero magntidue for a Vega-like star in the G band
VVega = 0.030 # Zero magntidue for a Vega-like star in the V band
IcVega = 0.033 # Zero magntidue for a Vega-like star in the Ic band

MH = 0.5

out_file = open(path+'Id_A_avg_grid_MH_'+np.str(MH)+'_wbprp.txt',"w")

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
                GMag = functions.get_Magnitude(Flux, wlenf, wlen, A_l, FVega, GVega, TG)
                BPMag = functions.get_Magnitude(Flux, wlenf, wlen, A_l, FVega, BPVega, TBP)
                RPMag = functions.get_Magnitude(Flux, wlenf, wlen, A_l, FVega, RPVega, TRP)
                VMag = functions.get_Magnitude(Flux, wlenf, wlen, A_l, FVega, VVega, TV)
                IcMag = functions.get_Magnitude(Flux, wlenf, wlen, A_l, FVega, IcVega, TIc)

                #Writes file with spectrum ID, extinction and GMag, VMag and IcMag
                out_file.write( np.str(Id[i_I]) + "     " + np.str(Av[i_A]) + " " + np.str(GMag) + "    " + np.str(VMag) + "    " + np.str(IcMag) + "    " + np.str(BPMag) + "    " + np.str(RPMag) + '\n')

out_file.close()
