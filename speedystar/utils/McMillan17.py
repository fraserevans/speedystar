# McMillan (2017) potential as first implemented in the galpy framework by
# Mackereth & Bovy (2018)
import numpy
from galpy.potential import NFWPotential
from galpy.potential import DiskSCFPotential
from galpy.potential import SCFPotential
from galpy.potential import scf_compute_coeffs_axi
from galpy.potential import mwpot_helpers
from galpy.potential import KeplerPotential
from galpy.util import bovy_conversion
from astropy import units as u

# Suppress the numpy floating-point warnings that this code generates...
old_error_settings= numpy.seterr(all='ignore')

def Mac17(ro=8.21,vo=233.1,Sigma0_thin=896.,Rd_thin=2.5,Sigma0_thick=183.,Rd_thick=3.02,rho0_bulge=98.4,rho0_halo=0.00854,rh=19.6):

    # Unit normalizations
    #ro= 8.21
    #vo= 233.1

    sigo= bovy_conversion.surfdens_in_msolpc2(vo=vo,ro=ro)
    rhoo= bovy_conversion.dens_in_msolpc3(vo=vo,ro=ro)

    #gas disk parameters (fixed in McMillan 2017...)
    Rd_HI= 7./ro
    Rm_HI= 4./ro
    zd_HI= 0.085/ro
    Sigma0_HI= 53.1/sigo
    Rd_H2= 1.5/ro
    Rm_H2= 12./ro
    zd_H2= 0.045/ro
    Sigma0_H2= 2180./sigo

    #parameters of best-fitting model in McMillan (2017)
    #stellar disks
    #Sigma0_thin= 896./sigo
    #Rd_thin= 2.5/ro
    #zd_thin= 0.3/ro #fixed, unimportant
    #Sigma0_thick= 183./sigo
    #Rd_thick= 3.02/ro
    #zd_thick= 0.9/ro #fixed, unimportant

    #bulge
    #rho0_bulge= 98.4/rhoo
    #r0_bulge= 0.075/ro # fixed
    #rcut= 2.1/ro # fixed

    #DM halo
    #rho0_halo= 0.00854/rhoo
    #rh= 19.6/ro

    Sigma0_thin/=sigo
    Rd_thin/=ro
    zd_thin = 0.3/ro #fixed, unimportant
    Sigma0_thick/= sigo
    Rd_thick/=ro
    zd_thick = 0.9/ro #fixed, unimportant

    #bulge
    rho0_bulge/=rhoo
    r0_bulge= 0.075/ro # fixed
    rcut= 2.1/ro # fixed

    #DM halo
    rho0_halo/=rhoo
    rh/=ro


    def gas_dens(R,z):
        return \
            mwpot_helpers.expsech2_dens_with_hole(R,z,Rd_HI,Rm_HI,zd_HI,Sigma0_HI)\
            +mwpot_helpers.expsech2_dens_with_hole(R,z,Rd_H2,Rm_H2,zd_H2,Sigma0_H2)

    def stellar_dens(R,z):
        return mwpot_helpers.expexp_dens(R,z,Rd_thin,zd_thin,Sigma0_thin)\
            +mwpot_helpers.expexp_dens(R,z,Rd_thick,zd_thick,Sigma0_thick)

    def bulge_dens(R,z):
        return mwpot_helpers.core_pow_dens_with_cut(R,z,1.8,r0_bulge,rcut,
                                                rho0_bulge,0.5)

    #dicts used in DiskSCFPotential 
    sigmadict = [{'type':'exp','h':Rd_HI,'amp':Sigma0_HI, 'Rhole':Rm_HI},
             {'type':'exp','h':Rd_H2,'amp':Sigma0_H2, 'Rhole':Rm_H2},
             {'type':'exp','h':Rd_thin,'amp':Sigma0_thin},
             {'type':'exp','h':Rd_thick,'amp':Sigma0_thick}]

    hzdict = [{'type':'sech2', 'h':zd_HI},
          {'type':'sech2', 'h':zd_H2},
          {'type':'exp', 'h':zd_thin},
          {'type':'exp', 'h':zd_thick}]

    #generate separate disk and halo potential - and combined potential
    McMillan_bulge= SCFPotential(\
        Acos=scf_compute_coeffs_axi(bulge_dens,20,10,a=0.1)[0],
        a=0.1,ro=ro,vo=vo)
    McMillan_disk= DiskSCFPotential(\
        dens=lambda R,z: gas_dens(R,z)+stellar_dens(R,z),
        Sigma=sigmadict,hz=hzdict,a=2.5,N=30,L=30,ro=ro,vo=vo)
    McMillan_halo= NFWPotential(amp=rho0_halo*(4*numpy.pi*rh**3),
                                a=rh,ro=ro,vo=vo)

    #BH mass MW
    #Mbh = 3.8557e-5#*u.Msun
    Mbh = 4.3e-6*4e6/(vo*vo*ro)

    bh = KeplerPotential(amp=Mbh, normalize=False,ro=ro,vo=vo)

    # Go back to old floating-point warnings settings
    numpy.seterr(**old_error_settings)
    McMillan17= McMillan_disk+McMillan_halo+McMillan_bulge+bh
    #McMillan17 = bh
    #McMillan17.turn_physical_on()

    return McMillan17
