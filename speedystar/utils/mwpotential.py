from galpy.potential import  HernquistPotential, MiyamotoNagaiPotential, KeplerPotential, evaluatePotentials, turn_physical_on, evaluaterforces,evaluatezforces,evaluateRforces, ChandrasekharDynamicalFrictionForce
from galpy.potential import NFWPotential, TriaxialNFWPotential#, PlummerSoftening#, MovingObjectPotential
from astropy import units as u
from astropy.constants import G
import numpy as np
#from . softeningtest2 import LMCSoftening
#from . MovingObjectPotential2 import MovingObjectPotential

def MWPotential(Ms=0.76, rs=24.8, c=1., T=True):
    '''
        Milky Way potential from Marchetti 2017b -- see galpy for the definitions of the potential components

        Parameters
        ----------
            Ms : float
                NFW profile scale mass in units of e12 Msun
            rs : float
                Radial profile in units of kpc
            c : float
                Axis ratio
            T : bool
                If True, use triaxialNFWPotential
    '''

    # NFW profile
    Ms = Ms*1e12*u.Msun
    rs = rs*u.kpc

    #Disk
    Md = 1e11 * u.Msun
    #Md = (10**-0.4)*1e11*u.Msun
    ad = 6.5 * u.kpc
    bd = 260. * u.pc

    #Bulge
    Mb = 3.4*1e10*u.Msun
    #Mb = (10**0.9315)*1e10*u.Msun
    Rb = 0.7*u.kpc

    #BH mass in 1e6 Msun
    Mbh = 4e6*u.Msun
    if(T):
        halop = TriaxialNFWPotential(amp=Ms, a=rs, c=c, normalize=False)
    else:
        halop = NFWPotential(amp=Ms, a=rs, normalize=False)
    diskp = MiyamotoNagaiPotential(amp=Md, a=ad, b=bd, normalize=False)
    bulgep = HernquistPotential(amp=2*Mb, a=Rb, normalize=False) #Factor 2 because of the galpy definition
    bh = KeplerPotential(amp=Mbh, normalize=False)

    #totpot = evaluatePotentials(diskp,0,Rb) + evaluatePotentials(halop,0,Rb) + evaluatePotentials(bulgep,0,Rb) + evaluatePotentials(bh,0,Rb)

    #return [halop, diskp, bulgep, bh,evaluatePotentials(bulgep,0,Rb)/totpot]
    return [halop, diskp, bulgep, bh]

def MWPotentialVaryDisk(Md=1, T=True):
    '''
        Milky Way potential from Marchetti 2017b -- see galpy for the definitions of the potential components

        Parameters
        ----------
            Ms : float
                NFW profile scale mass in units of e12 Msun
            rs : float
                Radial profile in units of kpc
            c : float
                Axis ratio
            T : bool
                If True, use triaxialNFWPotential
    '''

    # NFW profile
    Ms = 0.76*1e12*u.Msun
    rs = 24.8*u.kpc
    c=1.

    #Disk
    Md = Md*1e11 * u.Msun
    #Md = 0. * u.Msun
    ad = 6.5 * u.kpc
    bd = 260. * u.pc

    #Bulge
    Mb = 3.4*1e10*u.Msun
    Rb = 0.7*u.kpc

    #BH mass in 1e6 Msun
    Mbh = 4e6*u.Msun
    if(T):
        halop = TriaxialNFWPotential(amp=Ms, a=rs, c=c, normalize=False)
    else:
        halop = NFWPotential(amp=Ms, a=rs, normalize=False)
    diskp = MiyamotoNagaiPotential(amp=Md, a=ad, b=bd, normalize=False)
    bulgep = HernquistPotential(amp=2*Mb, a=Rb, normalize=False) #Factor 2 because of the galpy definition
    bh = KeplerPotential(amp=Mbh, normalize=False)

    #totpot = evaluatePotentials(diskp,0,Rb) + evaluatePotentials(halop,0,Rb) + evaluatePotentials(bulgep,0,Rb) + evaluatePotentials(bh,0,Rb)

    #return [halop, diskp, bulgep, bh,evaluatePotentials(bulgep,0,Rb)/totpot]
    return [halop, diskp, bulgep, bh]

def MWPotentialVaryBulge(Mb=3.4, T=True):
    '''
        Milky Way potential from Marchetti 2017b -- see galpy for the definitions of the potential components

        Parameters
        ----------
            Ms : float
                NFW profile scale mass in units of e12 Msun
            rs : float
                Radial profile in units of kpc
            c : float
                Axis ratio
            T : bool
                If True, use triaxialNFWPotential
    '''

    # NFW profile
    Ms = 0.76*1e12*u.Msun
    rs = 24.8*u.kpc
    c=1.

    #Disk
    Md = 1e11 * u.Msun
    #Md = 0. * u.Msun
    ad = 6.5 * u.kpc
    bd = 260. * u.pc

    #Bulge
    Mb = Mb*1e10*u.Msun
    Rb = 0.7*u.kpc

    #BH mass in 1e6 Msun
    Mbh = 4e6*u.Msun
    if(T):
        halop = TriaxialNFWPotential(amp=Ms, a=rs, c=c, normalize=False)
    else:
        halop = NFWPotential(amp=Ms, a=rs, normalize=False)
    diskp = MiyamotoNagaiPotential(amp=Md, a=ad, b=bd, normalize=False)
    bulgep = HernquistPotential(amp=2*Mb, a=Rb, normalize=False) #Factor 2 because of the galpy definition
    bh = KeplerPotential(amp=Mbh, normalize=False)

    #totpot = evaluatePotentials(diskp,0,Rb) + evaluatePotentials(halop,0,Rb) + evaluatePotentials(bulgep,0,Rb) + evaluatePotentials(bh,0,Rb)

    #return [halop, diskp, bulgep, bh,evaluatePotentials(bulgep,0,Rb)/totpot]
    return [halop, diskp, bulgep, bh]



def MWLMCPotential(Ms=0.76, rs=24.8, c=1., T=True):
    #print(galpy.__version__)
    from galpy.orbit import Orbit
    import astropy.coordinates as coord
    from astropy.table import Table
    import os
    '''
        Milky Way potential from Marchetti 2017b -- see galpy for the definitions of the potential components

        Parameters
        ----------
            Ms : float
                NFW profile scale mass in units of e12 Msun
            rs : float
                Radial profile in units of kpc
            c : float
                Axis ratio
            T : bool
                If True, use triaxialNFWPotential
    '''

    # U, V, W in km/s in galactocentric coordinates. Galpy notation requires U to have a minus sign.
    solarmotion = [-14., 12.24, 7.25]

    # NFW profile
    Ms = Ms*1e12*u.Msun
    rs = rs*u.kpc

    #Disk
    Md = 1e11 * u.Msun
    #Md = 0. * u.Msun
    ad = 6.5 * u.kpc
    bd = 260. * u.pc

    #Bulge
    Mb = 3.4*1e10*u.Msun
    Rb = 0.7*u.kpc

    #BH mass in 1e6 Msun
    Mbh = 4e6*u.Msun

    #LMC parameters
    Mlmc = 1.5*1e11*u.Msun
    Rlmc = 17.14*u.kpc

    if(T):
        halop = TriaxialNFWPotential(amp=Ms, a=rs, c=c, normalize=False)
    else:
        halop = NFWPotential(amp=Ms, a=rs, normalize=False)
    diskp = MiyamotoNagaiPotential(amp=Md, a=ad, b=bd, normalize=False)#.turn_physical_on()
    bulgep = HernquistPotential(amp=2*Mb, a=Rb, normalize=False) #Factor 2 because of the galpy definition
    bh = KeplerPotential(amp=Mbh, normalize=False)

    #print(dir(bulgep))

    #print(diskp.vcirc(10.))

    #totpot = evaluatePotentials(diskp,Rb,0) + evaluatePotentials(halop,Rb,0) + evaluatePotentials(bulgep,Rb,0) + evaluatePotentials(bh,Rb,0)
    totpot = evaluatePotentials(bulgep,41.426*u.kpc,0)
    #totpot = evaluatePotentials(bulgep,5.178*u.kpc,0)
    #print(totpot)

    #Orbit of LMC CoM
    #LMCorbit = Orbit(vxvv = [81.91*u.deg,-69.87*u.deg, 49.59*u.kpc, \
    LMCorbit = Orbit(vxvv = [78.76*u.deg,-69.19*u.deg, 49.59*u.kpc, \
                                    1.91*u.mas/u.yr, 0.229*u.mas/u.yr, 262.2*u.km/u.s], \
                                    solarmotion=solarmotion, radec=True).flip() 

    #totpot = [halop,diskp,bulgep,bh]
    LMCfric = ChandrasekharDynamicalFrictionForce(amp=1.0,GMs=G*Mlmc,gamma=1.0,rhm =(1+np.sqrt(2))*Rlmc,dens=[halop,diskp,bulgep,bh])

    ts = np.linspace(0, 1, 1000)*1000*u.Myr
    #LMCorbit.integrate(ts,[halop,diskp,bulgep,bh], method='dopr54_c')
    LMCorbit.integrate(ts,[halop,diskp,bulgep,bh,LMCfric], method='dopr54_c')
    #LMCorbit.integrate(ts,bh, method='dopr54_c')

    soft = LMCSoftening(m=Mlmc.value,r=Rlmc.value)
    #LMCp = MovingObjectPotential(orbit=LMCorbit,softening=HernquistPotential(amp=2*Mlmc,a=Rlmc, normalize=False),softening_length=0)
    #LMCp = MovingObjectPotential(orbit=LMCorbit)
    LMCp = MovingObjectPotential(orbit=LMCorbit,softening=soft)


    #vSun = [-solarmotion[0], solarmotion[1], solarmotion[2]] * u.km / u.s # (U, V, W)
    #vrot = [0., 220., 0.] * u.km / u.s
    #RSun = 8. * u.kpc
    #zSun = 0.025 * u.kpc
    #v_sun = coord.CartesianDifferential(vrot+vSun)
    #gc = coord.Galactocentric(galcen_distance=RSun, z_sun=zSun, galcen_v_sun=v_sun)
    #ICRS = coord.ICRS(ra=LMCorbit.ra(ts,use_physical=True)*u.deg, dec=LMCorbit.dec(ts,use_physical=True)*u.deg, \
    #    distance=LMCorbit.dist(ts,use_physical=True)*u.kpc, pm_ra_cosdec=LMCorbit.pmra(ts,use_physical=True)*u.mas/u.yr, \
    #    pm_dec=LMCorbit.pmdec(ts,use_physical=True)*u.mas/u.yr, radial_velocity=LMCorbit.vlos(ts,use_physical=True)*u.km/u.s)
    #gal = ICRS.transform_to(gc)

    #v_x, v_y, v_z = gal.v_x, gal.v_y, gal.v_z
    #xpos, ypos, zpos = gal.x, gal.y, gal.z
    #datalist=[ts, gal.x, gal.y, gal.z, gal.v_x, gal.v_y, gal.v_z]

    #namelist = ['t', 'x', 'y', 'z', 'v_x', 'v_y', 'v_z']
    #data_table = Table(data=datalist, names=namelist)

    #path='/run/media/evans/My_Passport/Backflights/dSphs_Backflights_LMC10wfric_Halos/'+str(np.round(np.log10(Ms.value)-12,decimals=4))+'_'+str(np.round(np.log10(rs.value),decimals=5))
    #path='/run/media/evans/My_Passport/Backflights/dSphs_Backflights_LMC10wfric'
    #path='/run/media/evans/My_Passport/Backflights/dSphs_Backflights_LMC10wfric_Halos/e'+str(np.round(c,decimals=1))

    #if not os.path.exists(path):
    #    os.mkdir(path)

    #data_table.write(path+'/Dwarfs_flight10_Cart_keep.fits', overwrite=True)

    #datalist=[ts, LMCorbit.ra(ts,use_physical=True)*u.deg, LMCorbit.dec(ts,use_physical=True)*u.deg, \
    #    LMCorbit.dist(ts,use_physical=True)*u.kpc, LMCorbit.pmra(ts,use_physical=True)*u.mas/u.yr, \
    #    LMCorbit.pmdec(ts,use_physical=True)*u.mas/u.yr, LMCorbit.vlos(ts,use_physical=True)*u.km/u.s]

    #namelist = ['t', 'ra', 'dec', 'dist', 'pm_ra', 'pm_dec', 'vlos']
    #data_table = Table(data=datalist, names=namelist)
    #data_table.write(path+'/Dwarf_flight10_keep.fits', overwrite=True)



    #print(evaluatePotentials(LMCp,41.426*u.kpc,0,phi=-90.53614*u.deg,t=0.0*u.Myr))

    #print(evaluatezforces(LMCp,41.426*u.kpc,0,phi=-90.53614*u.deg,t=0.0*u.Myr))

    #print(evaluatezforces(LMCp,41.426*u.kpc,0,phi=-1.580154*u.rad,t=0.0*u.Myr))

    #print(evaluaterforces(bulgep,41.426*u.kpc,0,phi=-90.53614*u.deg,t=0.0*u.Myr))
    
    #print(LMCp.zforce(41.426*u.kpc,0,phi=-1.561438,t=0*u.Myr))

    #print(LMCp.zforce(41.426,0,phi=0,t=0*u.Myr))

    #pottest = evaluatePotentials(LMCp,41.426*u.kpc,0*u.kpc,phi=-1.561438*u.rad,t=500*u.Myr)
    #print(pottest)
    #print(evaluatePotentials(halop,41.426*u.kpc,0*u.kpc,phi=-1.561438*u.rad,t=0*u.Myr),evaluatePotentials(bulgep,41.426*u.kpc,0*u.kpc,phi=-1.561438*u.rad,t=0*u.Myr),evaluatePotentials(diskp,41.426*u.kpc,0*u.kpc,phi=-1.561438*u.rad,t=0*u.Myr),evaluatePotentials(bh,41.426*u.kpc,0*u.kpc,phi=-1.561438*u.rad,t=0*u.Myr))

    #return [halop, diskp, bulgep, bh,evaluatePotentials(bulgep,0,Rb)/totpot]
    return [halop, diskp, bulgep, bh, LMCp]
    #return bulgep


def MWLMCM31M33Potential(Ms=0.76, rs=24.8, c=1., T=True):
    #print(galpy.__version__)
    from galpy.orbit import Orbit
    import astropy.coordinates as coord
    from astropy.table import Table
    import os
    '''
        Milky Way potential from Marchetti 2017b -- see galpy for the definitions of the potential components

        Parameters
        ----------
            Ms : float
                NFW profile scale mass in units of e12 Msun
            rs : float
                Radial profile in units of kpc
            c : float
                Axis ratio
            T : bool
                If True, use triaxialNFWPotential
    '''

    # U, V, W in km/s in galactocentric coordinates. Galpy notation requires U to have a minus sign.
    solarmotion = [-14., 12.24, 7.25]

    # NFW profile
    Ms = Ms*1e12*u.Msun
    rs = rs*u.kpc

    #Disk
    Md = 1e11 * u.Msun
    #Md = 0. * u.Msun
    ad = 6.5 * u.kpc
    bd = 260. * u.pc

    #Bulge
    Mb = 3.4*1e10*u.Msun
    Rb = 0.7*u.kpc

    #BH mass in 1e6 Msun
    Mbh = 4e6*u.Msun

    #LMC parameters
    Mlmc = 1.5*1e11*u.Msun
    Rlmc = 17.14*u.kpc

    
    #M31 parameters
    M31 = 1.5e12 * u.Msun

    #M33 parameters
    M33 = 5e11 * u.Msun

    if(T):
        halop = TriaxialNFWPotential(amp=Ms, a=rs, c=c, normalize=False)
    else:
        halop = NFWPotential(amp=Ms, a=rs, normalize=False)
    diskp = MiyamotoNagaiPotential(amp=Md, a=ad, b=bd, normalize=False)#.turn_physical_on()
    bulgep = HernquistPotential(amp=2*Mb, a=Rb, normalize=False) #Factor 2 because of the galpy definition
    bh = KeplerPotential(amp=Mbh, normalize=False)

    #Orbit of LMC CoM
    LMCorbit = Orbit(vxvv = [78.76*u.deg,-69.19*u.deg, 49.59*u.kpc, \
                                    1.91*u.mas/u.yr, 0.229*u.mas/u.yr, 262.2*u.km/u.s], \
                                    solarmotion=solarmotion, radec=True).flip() 
    LMCfric = ChandrasekharDynamicalFrictionForce(amp=1.0,GMs=G*Mlmc,gamma=1.0,rhm =(1+np.sqrt(2))*Rlmc,dens=[halop,diskp,bulgep,bh])
    ts = np.linspace(0, 1, 1700)*1700*u.Myr
    LMCorbit.integrate(ts,[halop,diskp,bulgep,bh,LMCfric], method='dopr54_c')
    soft = LMCSoftening(m=Mlmc.value,r=Rlmc.value)
    LMCp = MovingObjectPotential(orbit=LMCorbit,softening=soft)


    #Orbit of M31 CoM 
    M31orbit = Orbit(vxvv = [10.68333*u.deg,41.26917*u.deg, 770*u.kpc, \
                                    0.049*u.mas/u.yr, -0.038*u.mas/u.yr, -301*u.km/u.s], \
                                    solarmotion=solarmotion, radec=True).flip() 
    ts = np.linspace(0, 1, 1600)*1600*u.Myr
    M31orbit.integrate(ts,[halop,diskp,bulgep,bh,LMCp], method='dopr54_c')
    soft = LMCSoftening(m=M31.value,r=0.)
    M31p = MovingObjectPotential(orbit=M31orbit,softening=soft)

    #Orbit of M33 CoM 
    M33orbit = Orbit(vxvv = [23.4625*u.deg,30.6602*u.deg, 770*u.kpc, \
                                    0.024*u.mas/u.yr, 0.003*u.mas/u.yr, 794*u.km/u.s], \
                                    solarmotion=solarmotion, radec=True).flip() 
    ts = np.linspace(0, 1, 1500)*1500*u.Myr
    M33orbit.integrate(ts,[halop,diskp,bulgep,bh,LMCp,M31p], method='dopr54_c')
    soft = LMCSoftening(m=M33.value,r=0.)
    M33p = MovingObjectPotential(orbit=M33orbit,softening=soft)

    #vSun = [-solarmotion[0], solarmotion[1], solarmotion[2]] * u.km / u.s # (U, V, W)
    #vrot = [0., 220., 0.] * u.km / u.s
    #RSun = 8. * u.kpc
    #zSun = 0.025 * u.kpc
    #v_sun = coord.CartesianDifferential(vrot+vSun)
    #gc = coord.Galactocentric(galcen_distance=RSun, z_sun=zSun, galcen_v_sun=v_sun)
    #ICRS = coord.ICRS(ra=LMCorbit.ra(ts,use_physical=True)*u.deg, dec=LMCorbit.dec(ts,use_physical=True)*u.deg, \
    #    distance=LMCorbit.dist(ts,use_physical=True)*u.kpc, pm_ra_cosdec=LMCorbit.pmra(ts,use_physical=True)*u.mas/u.yr, \
    #    pm_dec=LMCorbit.pmdec(ts,use_physical=True)*u.mas/u.yr, radial_velocity=LMCorbit.vlos(ts,use_physical=True)*u.km/u.s)
    #gal = ICRS.transform_to(gc)
    #v_x, v_y, v_z = gal.v_x, gal.v_y, gal.v_z
    #xpos, ypos, zpos = gal.x, gal.y, gal.z
    #datalist=[ts, gal.x, gal.y, gal.z, gal.v_x, gal.v_y, gal.v_z]
    #namelist = ['t', 'x', 'y', 'z', 'v_x', 'v_y', 'v_z']
    #data_table = Table(data=datalist, names=namelist)

    #path='/run/media/evans/My_Passport/Backflights/dSphs_Backflights_LMC10wfric_Halos/'+str(np.round(np.log10(Ms.value)-12,decimals=4))+'_'+str(np.round(np.log10(rs.value),decimals=5))
    #path='/run/media/evans/My_Passport/Backflights/dSphs_Backflights_LMC10wfric_Halos/e'+str(np.round(c,decimals=1))
    #if not os.path.exists(path):
    #    os.mkdir(path)

    #data_table.write('/run/media/evans/My_Passport/Backflights/dSphs_Backflights_LMC10wfric/Dwarfs_flight10_Cart.fits', overwrite=True)
    #data_table.write(path+'/Dwarfs_flight10_Cart.fits', overwrite=True)

    return [halop, diskp, bulgep, bh, LMCp, M31p, M33p]
    #return bulgep




def PotDiffDefault(r1, r2, theta, Ms=0.76, rs=24.8, c=1., T=True):
    '''
        Milky Way potential from Marchetti 2017b -- see galpy for the definitions of the potential components

        Parameters
        ----------
            Ms : float
                NFW profile scale mass in units of e12 Msun
            rs : float
                Radial profile in units of kpc
            c : float
                Axis ratio
            T : bool
                If True, use triaxialNFWPotential
    '''

    # NFW profile
    Ms = Ms*1e12*u.Msun
    rs = rs*u.kpc

    #Disk
    Md = 1e11 * u.Msun
    ad = 6.5 * u.kpc
    bd = 260. * u.pc

    #Bulge
    Mb = 3.4*1e10*u.Msun
    Rb = 0.7*u.kpc

    #BH mass in 1e6 Msun
    Mbh = 4e6*u.Msun

    z1 = r1 * np.cos(theta)
    z2 = r2 * np.cos(theta)

    R1 = r1 * np.sin(theta)
    R2 = r2 * np.sin(theta)

    #phiBH = Mbh * (1./r2 - 1./r1)

    #phiB = Mb * (1./(Rb+r2) - 1./(Rb+r1))

    #phiNFW = Ms * ( (1./r2)*np.log(1 + r2/rs) - (1./r1)*np.log(1 + r1/rs) )

    #phiD = Md * (  np.sqrt(R2**2 + (ad + np.sqrt(z2**2 + bd**2) )**2 )**(-1) - np.sqrt(R1**2 + (ad + np.sqrt(z1**2 + bd**2) )**2 )**(-1) )

    #deltaphi = G * (phiBH + phiB + phiD + phiNFW)

    if(T):
        halop = TriaxialNFWPotential(amp=Ms, a=rs, c=c, normalize=False)
    else:
        halop = NFWPotential(amp=Ms, a=rs, normalize=False)
    diskp = MiyamotoNagaiPotential(amp=Md, a=ad, b=bd, normalize=False)
    bulgep = HernquistPotential(amp=2*Mb, a=Rb, normalize=False) #Factor 2 because of the galpy definition
    bh = KeplerPotential(amp=Mbh, normalize=False)


    #print(len(R1))
    #print(len(z1))
    totpot1 = evaluatePotentials(diskp,R1,z1) + evaluatePotentials(halop,R1,z1) + evaluatePotentials(bulgep,R1,z1) + evaluatePotentials(bh,R1,z1)
    totpot2 = evaluatePotentials(diskp,R2,z2) + evaluatePotentials(halop,R2,z2) + evaluatePotentials(bulgep,R2,z2) + evaluatePotentials(bh,R2,z2)

    #print((totpot1-totpot2)*u.km**2/u.s**2)

    deltaphi = (totpot1-totpot2)*u.km**2/u.s**2

    return(deltaphi)

def PotDiff(potential, r1, r2, theta=0,phi=None):

    z1 = r1 * np.cos(theta)
    z2 = r2 * np.cos(theta)

    R1 = r1 * np.sin(theta)
    R2 = r2 * np.sin(theta)

    #print(R1)
    #print(z1)
    #print(phi)
    totpot1 = evaluatePotentials(potential,R1,z1,phi=phi,t=0*u.Myr)
    totpot2 = evaluatePotentials(potential,R2,z2,phi=phi,t=0*u.Myr)

    deltaphi = (totpot1-totpot2)*u.km**2/u.s**2

    return(deltaphi)


def PotDiffTwoTheta(r1, r2, theta1, theta2, Ms=0.76, rs=24.8, c=1., T=True):
    '''
        Milky Way potential from Marchetti 2017b -- see galpy for the definitions of the potential components

        Parameters
        ----------
            Ms : float
                NFW profile scale mass in units of e12 Msun
            rs : float
                Radial profile in units of kpc
            c : float
                Axis ratio
            T : bool
                If True, use triaxialNFWPotential
    '''

    # NFW profile
    Ms = Ms*1e12*u.Msun
    rs = rs*u.kpc

    #Disk
    Md = 1e11 * u.Msun
    ad = 6.5 * u.kpc
    bd = 260. * u.pc

    #Bulge
    Mb = 3.4*1e10*u.Msun
    Rb = 0.7*u.kpc

    #BH mass in 1e6 Msun
    Mbh = 4e6*u.Msun

    z1 = r1 * np.cos(theta1)
    z2 = r2 * np.cos(theta2)

    R1 = r1*np.sin(theta1)
    R2 = r2*np.sin(theta2)

    phiBH = Mbh * (1./r2 - 1./r1)

    phiB = Mb * (1./(Rb+r2) - 1./(Rb+r1))

    phiNFW = Ms * ( (1./r2)*np.log(1 + r2/rs) - (1./r1)*np.log(1 + r1/rs) )

    phiD = Md * (  np.sqrt(R2**2 + (ad + np.sqrt(z2**2 + bd**2) )**2 )**(-1)- np.sqrt(R1**2 + (ad + np.sqrt(z1**2 + bd**2) )**2 )**(-1) )

    Gtest = 0.000004302*u.kpc*((u.km/u.s)**2)/u.Msun
    #print(Gtest - G.to('(km2 kpc) / (Msun s2)'))

    deltaphi =  G*(phiBH + phiB + phiD + phiNFW)

    print(Mbh/r1)

    halop = NFWPotential(amp=Ms, a=rs, normalize=False)#.turn_physical_on()
    diskp = MiyamotoNagaiPotential(amp=Md, a=ad, b=bd, normalize=False)#.turn_physical_on()
    bulgep = HernquistPotential(amp=2*Mb, a=Rb, normalize=False)#.turn_physical_on() #Factor 2 because of the galpy definition
    bh = KeplerPotential(amp=Mbh, normalize=False)#.turn_physical_on()

    #totpot1 = evaluatePotentials(diskp,R=r1,z=z1) + evaluatePotentials(halop,R=r1,z=z1) + evaluatePotentials(bulgep,R=r1,z=z1) + evaluatePotentials(bh,R=r1,z=z1)
    #totpot2 = evaluatePotentials(diskp,R=r2,z=z2) + evaluatePotentials(halop,R=r2,z=z2) + evaluatePotentials(bulgep,R=r2,z=z2) + evaluatePotentials(bh,R=r2,z=z2)
    #print(r1)
    #print(z1)
    #print(r2)
    #print(z2)
    #print(evaluatePotentials(diskp,r1,z1,use_physical=True))
    #+ evaluatePotentials(halop,r1,z1) + evaluatePotentials(bulgep,r1,z1) + evaluatePotentials(bh,r1,z1)
    totpot1 = evaluatePotentials(diskp,r1,z1) + evaluatePotentials(halop,r1,z1) + evaluatePotentials(bulgep,r1,z1) + evaluatePotentials(bh,r1,z1)
    totpot2 = evaluatePotentials(diskp,r2,z2) + evaluatePotentials(halop,r2,z2) + evaluatePotentials(bulgep,r2,z2) + evaluatePotentials(bh,r2,z2)

    print(evaluatePotentials(bh,r1,z1))


    #deltaphi = totpot1 - totpot2

    #print(deltaphi.to('m3 solMass / (kg kpc s2)'))
    return(deltaphi)

def GalaPotential(Ms=0.54, rs=15.62, c=1., T=True):
    '''
        Milky Way potential from Marchetti 2017b -- see galpy for the definitions of the potential components

        Parameters
        ----------
            Ms : float
                NFW profile scale mass in units of e12 Msun
            rs : float
                Radial profile in units of kpc
            c : float
                Axis ratio
            T : bool
                If True, use triaxialNFWPotential
    '''

    # NFW profile
    Ms = Ms*1e12*u.Msun
    rs = rs*u.kpc

    #Disk
    Md = 0.68e11 * u.Msun
    #Md = 0. * u.Msun
    ad = 3. * u.kpc
    bd = 280. * u.pc

    #Bulge
    Mb = 5.*1e9*u.Msun
    Rb = 1*u.kpc

    #BH mass in 1e6 Msun
    Mbh = 1.71e9*u.Msun
    rh = 0.07*u.kpc

    if(T):
        halop = TriaxialNFWPotential(amp=Ms, a=rs, c=c, normalize=False)
    else:
        halop = NFWPotential(amp=Ms, a=rs, normalize=False)
    diskp = MiyamotoNagaiPotential(amp=Md, a=ad, b=bd, normalize=False)
    bulgep = HernquistPotential(amp=2*Mb, a=Rb, normalize=False) #Factor 2 because of the galpy definition
    #bh = KeplerPotential(amp=Mbh, normalize=False)
    bh = HernquistPotential(amp=2*Mb, a=Rb, normalize=False)

    #totpot = evaluatePotentials(diskp,0,Rb) + evaluatePotentials(halop,0,Rb) + evaluatePotentials(bulgep,0,Rb) + evaluatePotentials(bh,0,Rb)

    #return [halop, diskp, bulgep, bh,evaluatePotentials(bulgep,0,Rb)/totpot]
    return [halop, diskp, bulgep, bh]

def NoPot():

    Mbh = 0*u.Msun
    bh = KeplerPotential(amp=Mbh, normalize=False)

    return [bh]

