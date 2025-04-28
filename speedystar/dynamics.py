import os
from astropy import units as u
import numpy as np
import signal
import time
from tqdm import tqdm
import platform

def handler(signum, frame):
    print('OOPS! A star took to long to integrate. Skipping.')

#Define a timeout handler if the propagation takes too long
def timeout_handler(signum, frame):
    raise TimeoutError

#Set up a signal alarm object for the timeout handling
signal.signal(signal.SIGALRM, timeout_handler)

#@propagate
def propagate(self, potential, dt=0.1*u.Myr, 
                  solarmotion=[-11.1, 12.24, 7.25],
                    zo=0.0208*u.kpc, orbit_path=None,intmethod='dopr54_c'):
    '''
    Propagates the sample in the Galaxy forwards in time.

    Requires
    ----------
    potential : galpy potential instance
        Potential instance of the galpy library used to integrate the orbits

    Optional
    ----------
    dt : Quantity (time)
        Integration timestep. Defaults to 0.01 Myr
    solarmotion : length-3 list of floats
        UVW Solar peculiar velocity in km/s. 
        Galpy likes the U to be sign-flipped. Default is Schonrich+2010
    zo : Float or astropy distance quantity
            Offset of the Sun above or below the Galactic plane.
            Default is 20.8 pc (Bennett+Bovy 2019)
            If float, units are assumed to be kpc
    orbit_path : None or string
        If supplied, full equatorial and Galactocentric Cartesian orbits 
        are saved to orbit_path. Useful for debugging        
    intmethod : string
        Integration method to be passed to galpy. Default is 'dopr54_c'
    '''

    from galpy.orbit import Orbit
    from galpy.util.conversion import get_physical
    from astropy.table import Table

    from galpy.potential import evaluatePotentials, evaluatephitorques, evaluateDensities, evaluatePotentials, evaluatephi2derivs, evaluateRforces, evaluatezforces, evaluaterforces

    from galpy.potential import mwpotentials
    from galpy.potential import MWPotential2014
    
    self.solarmotion=solarmotion
    self.zo = zo

    #Read vo and ro from potential
    self.potential = potential

    self.vo = get_physical(potential)['vo']*u.km/u.s
    self.ro = get_physical(potential)['ro']*u.kpc

    # Integration time step
    self.dt = dt
    
    # Number of integration steps
    nsteps = np.ceil((self.tflight/self.dt).to('1').value).astype(int)

    # Impose 100 timesteps at minimum
    nsteps[nsteps<100] = 100

    # Initialize position in cylindrical coords
    rho   = self.r0 * np.sin(self.theta0)
    z     = self.r0 * np.cos(self.theta0)
    phi   = self.phi0
    phiv0 = self.phiv0

    #... and velocity
    vx = self.v0 * np.sin(self.thetav0) * np.cos(phiv0)
    vy = self.v0 * np.sin(self.thetav0) * np.sin(phiv0)
    vz = self.v0 * np.cos(self.thetav0)

    vR = vx*np.sin(phi+0.5 *np.pi*u.rad) + vy*np.sin(phi)
    vT = vx*np.cos(phi+0.5 *np.pi*u.rad) + vy*np.cos(phi)

    #Initialize a lot of stuff
    self.vx, self.vy, self.vz = (np.zeros(self.size)*u.km/u.s
                                            for i in range(3))
    self.x, self.y, self.z    = (np.zeros(self.size)*u.kpc 
                                            for i in range(3))
    self.ra, self.dec         = (np.zeros(self.size)*u.deg 
                                            for i in range(2))
    self.dist                 = np.zeros(self.size)*u.kpc          
    self.par                             = np.zeros(self.size)*u.mas
    self.pmra, self.pmdec     = (np.zeros(self.size)*u.mas/u.yr 
                                            for i in range(2))
    self.vlos                 = np.zeros(self.size)*u.km/u.s

    self.Lz                   = np.zeros(self.size)*u.kpc*u.km/u.s

    self.orbits               = [None] * self.size

    self.b, self.l            = (np.zeros(self.size)*u.deg 
                                            for i in range(2))

    self.pmb, self.pml        = (np.zeros(self.size)*u.mas/u.yr 
                                            for i in range(2))

  
    self.vr = np.zeros(self.size)*u.km/u.s
    self.vR = np.zeros(self.size)*u.km/u.s
    self.vT = np.zeros(self.size)*u.km/u.s
   
    self.vtheta = np.zeros(self.size)*u.km/u.s

    #Integration loop for the self.size orbits
    for i in tqdm(range(self.size),desc='Propagating...'):

        # Galpy will hang on propagating an orbit on rare occasion. 
        # Is only a problem if the flight time is long and/or the star 
        # makes multiple close pericentres to the SMBH. 
        # The signal alarm prevents this.
        # Not yet implemented in Windows
        try:
            if platform.system()!='Windows':
                signal.alarm(5)

            #Get timesteps
            ts = np.linspace(-1, 0, nsteps[i])*self.tflight[i]

            #Initialize orbit using galactocentric cylindrical 
            #phase space info of stars

            self.orbits[i] = Orbit(vxvv = [rho[i], vR[i], vT[i], z[i], vz[i], \
                                    phi[i]], solarmotion=self.solarmotion, \
                                    zo=zo, **get_physical(potential))

            self.orbits[i].integrate(ts, potential, method=intmethod)

            # Export the final position
            self.ra[i] = self.orbits[i].ra(ts, quantity=True)[-1]
            self.dec[i] = self.orbits[i].dec(ts, quantity=True)[-1]
            self.pmra[i] = self.orbits[i].pmra(ts, quantity=True)[-1]
            self.pmdec[i] = self.orbits[i].pmdec(ts, quantity=True)[-1]
            self.dist[i] = self.orbits[i].dist(ts, quantity=True)[-1]
            self.par[i] = u.mas / self.dist[i].to('kpc').value
            self.vlos[i] = self.orbits[i].vlos(ts, quantity=True)[-1]

            self.b[i] = self.orbits[i].bb(ts, quantity=True)[-1]
            self.l[i] = self.orbits[i].ll(ts, quantity=True)[-1]
            self.pmb[i] = self.orbits[i].pmbb(ts, quantity=True)[-1]
            self.pml[i] = self.orbits[i].pmll(ts, quantity=True)[-1]

            self.vr[i] = self.orbits[i].vr(ts, quantity=True)[-1]
            self.vR[i] = self.orbits[i].vR(ts, quantity=True)[-1]
            self.vT[i] = self.orbits[i].vT(ts, quantity=True)[-1]
            self.vtheta[i] = self.orbits[i].vtheta(ts, quantity=True)[-1]

            self.x[i] = self.orbits[i].x(ts, quantity=True)[-1]
            self.y[i] = self.orbits[i].y(ts, quantity=True)[-1]
            self.z[i] = self.orbits[i].z(ts, quantity=True)[-1]
            self.vx[i] = self.orbits[i].vx(ts, quantity=True)[-1]
            self.vy[i] = self.orbits[i].vy(ts, quantity=True)[-1]
            self.vz[i] = self.orbits[i].vz(ts, quantity=True)[-1]
            self.Lz[i] = self.orbits[i].Lz(ts, quantity=True)[-1]

            #self.deltaE[i] = (self.orbits[i].E(ts, quantity=True)[-1]  - self.orbits[i].E(ts, quantity=True)[0]) / self.orbits[i].E(ts, quantity=True)[0]

            #Save the orbits of each HVS, if orbit_path is supplied
            if orbit_path is not None:
                #Only saves the orbits of the first 5e5 HVSs to prevent bloat

                if not os.path.exists(orbit_path):
                    raise SystemExit('Path '+orbit_path+' does not exist')

                if i<50000:
                    flightra    = self.orbits[i].ra(ts, quantity=True)
                    flightdec   = self.orbits[i].dec(ts, quantity=True)
                    flightdist  = self.orbits[i].dist(ts, quantity=True)
                    flightpmra  = self.orbits[i].pmra(ts, quantity=True)
                    flightpmdec = self.orbits[i].pmdec(ts, quantity=True)
                    flightvlos  = self.orbits[i].vlos(ts, quantity=True)
            
                    flightx  = self.orbits[i].x(ts, quantity=True)
                    flighty  = self.orbits[i].y(ts,quantity=True)
                    flightz  = self.orbits[i].z(ts,quantity=True)
                    flightvx = self.orbits[i].vx(ts,quantity=True)
                    flightvy = self.orbits[i].vy(ts,quantity=True)
                    flightvz = self.orbits[i].vz(ts,quantity=True)
                    flightL  = self.orbits[i].L(ts,quantity=True)
                    flightE = self.orbits[i].E(ts, quantity=True)

                    flightR = np.sqrt(flightx**2 + flighty**2)
                    flightr = np.sqrt(flightx**2 + flighty**2 + flightz**2)

                    flightvr = self.orbits[i].vr(ts, quantity=True)
                    flightvR = self.orbits[i].vR(ts, quantity=True)

                    flightphi = self.orbits[i].phi(ts, quantity=True)
                    flightvT = self.orbits[i].vT(ts, quantity=True)

                    flighttheta = np.arccos(flightz/np.sqrt(flightx**2 + flighty**2 + flightz**2))
                    flightvtheta = self.orbits[i].vtheta(ts, quantity=True)

                    flighttauphi = np.zeros(len(ts))
                    flightdens = np.zeros(len(ts))
                    flightpot = np.zeros(len(ts))
                    flightphi2derivs = np.zeros(len(ts))
                    flightRforces = np.zeros(len(ts))
                    flightzforces = np.zeros(len(ts))
                    flightrforces = np.zeros(len(ts))

                    flightax = np.zeros(len(ts))*u.km/(u.s**2)
                    flightax[1:] = (flightvx[1:] - flightvx[:-1])/(ts[1:] - ts[:-1])

                    flightay = np.zeros(len(ts))*u.km/(u.s**2)
                    flightay[1:] = (flightvy[1:] - flightvy[:-1])/(ts[1:] - ts[:-1])

                    flightaz = np.zeros(len(ts))*u.km/(u.s**2)
                    flightaz[1:] = (flightvz[1:] - flightvz[:-1])/(ts[1:] - ts[:-1])

                    flightaphi = (flightx*flightay - flighty*flightax) / (flightR) - (flightx*flightvy - flighty*flightvx)*(flightx*flightvx + flighty*flightvy) / (flightR**3)

                    #Writes cartesian orbits to file. Each star gets own file
                    datalist=[ts, flightx, flighty, flightz, flightvx, flightvy, flightvz, flightL, flightE, flightvr, flightvR,flightvT, flightvtheta, flightphi, flighttheta, flightra, flightdec,  flightdist, 
                    flightpmra, flightpmdec, flightvlos, flightr, flightR, flightax, flightay, flightaphi, flighttauphi, flightdens, flightpot, flightphi2derivs, flightRforces, flightzforces, flightrforces]

                    namelist = ['t', 'x', 'y', 'z', 'v_x', 'v_y', 'v_z', 'L',
                             'E', 'vr', 'vR', 'vT', 'vtheta', 'phi', 'theta', \
                                'ra', 'dec', 'dist', 'pm_ra', 'pm_dec', 'vlos', 'r', 'R', 'a_x', 'a_y', 'a_phi', 'tau_phi', 'dens', 'pot', 'phi2derivs', 'Rforces', 'zforces', 'rforces']

                    data_table = Table(data=datalist, names=namelist)
                    data_table.write(orbit_path+'flight'+str(i)+'.fits', 
                                    overwrite=True)

            if platform.system()!='Windows':
                signal.alarm(0)
        
        except TimeoutError:
            print(f"Warning: star {i+1} took too long to integrate. Moving on")

    self.propagated = True

    #Get Galactocentric distance and velocity and Galactic escape velocity
    #as well as final azimuthal and polar coordinates
    #in Galactocentric sherical coordinates
    if(self.size>0):
        self.get_vesc(potential=potential)
        self.GCdist = np.sqrt(self.x**2. + self.y**2. \
                                + self.z**2.).to(u.kpc)
        self.R = np.sqrt(self.x**2. + self.y**2.).to(u.kpc)
        self.GCv = np.sqrt(self.vx**2. + self.vy**2. + \
                                self.vz**2.).to(u.km/u.s)
        self.thetaf = np.arccos(self.z/self.GCdist)
        self.phif = np.arctan2(self.y,self.x)

    else:
        self.GCv = []*u.km/u.s 
        self.GCdist = []*u.kpc
        self.R = []*u.kpc
        self.Vesc = []*u.km/u.s 
        self.thetaf = []*u.rad
        self.phif = []*u.rad

#@propagate
def propagate_agama(self, potential, dt=0.1*u.Myr, 
                  solarmotion=[-11.1, 12.24, 7.25],
                    zo=0.0208*u.kpc, orbit_path=None):
        '''
        Propagates the sample in the Galaxy forwards in time. Not playtested.

        Requires
        ----------
        potential : galpy potential instance
            Potential instance of the galpy library used to integrate the orbits

        Optional
        ----------
        dt : Quantity (time)
            Integration timestep. Defaults to 0.01 Myr
        solarmotion : length-3 list of floats
            UVW Solar peculiar velocity in km/s. 
            Galpy likes the U to be sign-flipped. Default is Schonrich+2010
        zo : Float or astropy distance quantity
             Offset of the Sun above or below the Galactic plane.
             Default is 20.8 pc (Bennett+Bovy 2019)
             If float, units are assumed to be kpc
        orbit_path : None or string
            If supplied, full equatorial and Galactocentric Cartesian orbits 
            are saved to orbit_path. Useful for debugging        
        '''
        raise NotImplementedError('Agama not yet implemented')
        import signal

        #from galpy.orbit import Orbit
        #from galpy.util.conversion import get_physical
        #from astropy.table import Table
        #from galpy.potential import evaluatePotentials
        
        self.solarmotion=solarmotion
        self.zo = zo

        #Read vo and ro from potential
        self.potential = potential
        #self.vo = get_physical(potential)['vo']*u.km/u.s
        #self.ro = get_physical(potential)['ro']*u.kpc

        # Integration time step
        self.dt = dt
        
        # Number of integration steps
        nsteps = np.ceil((self.tflight/self.dt).to('1').value).astype(int)

        # Impose 100 timesteps at minimum
        nsteps[nsteps<100] = 100

        # Initialize position in cylindrical coords
        rho   = self.r0 * np.sin(self.theta0)
        x = self.r0 * np.sin(self.theta0) * np.cos(phi0)
        y = self.r0 * np.sin(self.theta0) * np.sin(phi0)
        z     = self.r0 * np.cos(self.theta0)
        phi   = self.phi0
        phiv0 = self.phiv0

        #... and velocity
        vx = self.v0 * np.sin(self.thetav0) * np.cos(phiv0)
        vy = self.v0 * np.sin(self.thetav0) * np.sin(phiv0)
        vz = self.v0 * np.cos(self.thetav0)

        vR = vx*np.sin(phi+0.5 *np.pi*u.rad) + vy*np.sin(phi)
        vT = vx*np.cos(phi+0.5 *np.pi*u.rad) + vy*np.cos(phi)

        #Initialize a lot of stuff
        self.vx, self.vy, self.vz = (np.zeros(self.size)*u.km/u.s
                                                for i in range(3))
        self.x, self.y, self.z    = (np.zeros(self.size)*u.kpc 
                                                for i in range(3))
        self.ra, self.dec         = (np.zeros(self.size)*u.deg 
                                                for i in range(2))
        self.dist                 = np.zeros(self.size)*u.kpc          
        self.par                             = np.zeros(self.size)*u.mas
        self.pmra, self.pmdec     = (np.zeros(self.size)*u.mas/u.yr 
                                                for i in range(2))
        self.vlos                 = np.zeros(self.size)*u.km/u.s

        self.orbits                          = [None] * self.size

        self.b, self.l            = (np.zeros(self.size)*u.deg 
                                                for i in range(2))
 
        self.pmb, self.pml        = (np.zeros(self.size)*u.mas/u.yr 
                                                for i in range(2))   

        #Integration loop for the self.size orbits
        for i in tqdm(range(self.size),desc='Propagating...'):

            # Galpy will hang on propagating an orbit on rare occasion. 
            # Is only a problem if the flight time is long and/or the star 
            # makes multiple close pericentres to the SMBH. 
            # The signal alarm prevents this.
            # Not yet implemented in Windows
            print(platform.system)
            if platform.system!='Windows':
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(5)

            #Get timesteps
            #ts = np.linspace(0, 1, nsteps[i])*self.tflight[i]
            ts = np.linspace(-1, 0, nsteps[i])*self.tflight[i]

            #Initialize orbit using galactocentric cylindrical 
            #phase space info of stars
        
            pos = np.array([np.array([x[i],y[i],z[i]])])
            vel = np.array([np.array([vx[i],vy[i],vz[i]])])

            self.orbits[i] = zagama.simpleorbit(potential,pos,vel,idealnorbits=2)
            self.orbits[i] = agama.orbit(ic=np.hstack([initpos,initvel]), potential=potential, time=self.tflight[i], dtype=object, accuracy=acc, lyapunov=True)
            #self.orbits[i] = Orbit(vxvv = [rho[i], vR[i], vT[i], z[i], vz[i], \
            #                         phi[i]], solarmotion=self.solarmotion, \
            #                         zo=zo, **get_physical(potential))

            #self.orbits[i].integrate(ts, potential, method='dopr54_c')

            # Export the final position
            self.ra[i] = self.orbits[i].ra(ts, quantity=True)[-1]
            self.dec[i] = self.orbits[i].dec(ts, quantity=True)[-1]
            self.pmra[i] = self.orbits[i].pmra(ts, quantity=True)[-1]
            self.pmdec[i] = self.orbits[i].pmdec(ts, quantity=True)[-1]
            self.dist[i] = self.orbits[i].dist(ts, quantity=True)[-1]
            self.par[i] = u.mas / self.dist[i].to('kpc').value
            self.vlos[i] = self.orbits[i].vlos(ts, quantity=True)[-1]

            self.b[i] = self.orbits[i].bb(ts, quantity=True)[-1]
            self.l[i] = self.orbits[i].ll(ts, quantity=True)[-1]
            self.pmb[i] = self.orbits[i].pmbb(ts, quantity=True)[-1]
            self.pml[i] = self.orbits[i].pmll(ts, quantity=True)[-1]

            self.x[i] = self.orbits[i].x(ts, quantity=True)[-1]
            self.y[i] = self.orbits[i].y(ts, quantity=True)[-1]
            self.z[i] = self.orbits[i].z(ts, quantity=True)[-1]
            self.vx[i] = self.orbits[i].vx(ts, quantity=True)[-1]
            self.vy[i] = self.orbits[i].vy(ts, quantity=True)[-1]
            self.vz[i] = self.orbits[i].vz(ts, quantity=True)[-1]

            #Save the orbits of each HVS, if orbit_path is supplied
            if orbit_path is not None:
                #Only saves the orbits of the first 5e5 HVSs to prevent bloat

                if not os.path.exists(orbit_path):
                    raise SystemExit('Path '+orbit_path+' does not exist')

                if i<50000:
                    flightra    = self.orbits[i].ra(ts, quantity=True)
                    flightdec   = self.orbits[i].dec(ts, quantity=True)
                    flightdist  = self.orbits[i].dist(ts, quantity=True)
                    flightpmra  = self.orbits[i].pmra(ts, quantity=True)
                    flightpmdec = self.orbits[i].pmdec(ts, quantity=True)
                    flightvlos  = self.orbits[i].vlos(ts, quantity=True)
             
                    flightx  = self.orbits[i].x(ts, quantity=True)
                    flighty  = self.orbits[i].y(ts,quantity=True)
                    flightz  = self.orbits[i].z(ts,quantity=True)
                    flightvx = self.orbits[i].vx(ts,quantity=True)
                    flightvy = self.orbits[i].vy(ts,quantity=True)
                    flightvz = self.orbits[i].vz(ts,quantity=True)
                    flightL  = self.orbits[i].L(ts,quantity=True)
             
                    #Table of equatorial coordinates for the star with time
                    #datalist = [ts, flightra, flightdec, flightdist, 
                    #            flightpmra, flightpmdec, flightvlos]
                    #namelist = ['t', 'ra', 'dec', 'dist', 
                    #            'pm_ra', 'pm_dec', 'vlos']
                    #data_table = Table(data=datalist, names=namelist)

                    #Writes equatorial orbits to file. Each star gets own file
                    #data_table.write(orbit_path+'flight'+str(i)+'_ICRS.fits', 
                    #                 overwrite=True)

                    #Writes cartesian orbits to file. Each star gets own file
                    datalist=[ts, flightx, flighty, flightz, flightvx, flightvy, 
                                flightvz, flightL, flightra, flightdec, 
                                flightdist, flightpmra, flightpmdec, flightvlos]
                    namelist = ['t', 'x', 'y', 'z', 'v_x', 'v_y', 'v_z', 'L', \
                                'ra', 'dec', 'dist', 'pm_ra', 'pm_dec', 'vlos']

                    data_table = Table(data=datalist, names=namelist)
                    data_table.write(orbit_path+'flight'+str(i)+'_trev.fits', 
                                     overwrite=True)

        if platform.system()!='Windows':
            signal.alarm(0)

        self.propagated = True

        #Get Galactocentric distance and velocity and Galactic escape velocity
        #as well as final azimuthal and polar coordinates
        #in Galactocentric sherical coordinates
        if(self.size>0):
            self.get_vesc(potential=potential)
            self.GCdist = np.sqrt(self.x**2. + self.y**2. \
                                    + self.z**2.).to(u.kpc)
            self.GCv = np.sqrt(self.vx**2. + self.vy**2. + \
                                    self.vz**2.).to(u.km/u.s)
            self.thetaf = np.arccos(self.z/self.GCdist)
            self.phif = np.arctan2(self.y,self.x)

        else:
            self.GCv = []*u.km/u.s 
            self.GCdist = []*u.kpc
            self.Vesc = []*u.km/u.s 
            self.thetaf = []*u.rad
            self.phif = []*u.rad

#@get_vesc
def get_vesc(self, potential):

    '''
    Returns the escape speed of a given potential 
    at each star's position in a propagated sample

    Requires
    ----------
    potential : galpy potential instance
        Potential instance of the galpy library used to integrate the orbits
    '''

    from galpy.potential import evaluatePotentials

    if (not hasattr(self,'propagated') or not self.propagated):
        print('Warning: sample may not have been propagated. Proceeding anyways...')

    self.Vesc = np.zeros(self.size)*u.km/u.s
    
    R = np.sqrt(self.x**2 + self.y**2)
    z = self.z
    phi = np.arctan2(self.y,self.x)

    #potential.turn_physical_on()
    for i in range(self.size):

        if hasattr(self,'tflight'):
            self.Vesc[i] = np.sqrt(2*(evaluatePotentials(potential, 
                            1e6*u.kpc, 0*u.kpc, phi=phi[i], 
                            t=-self.tflight[i],quantity=True) \
                            
                            - evaluatePotentials(potential, R[i], z[i], t=-self.tflight[i], phi=phi[i],quantity=True)))
        else:
            self.Vesc[i] = np.sqrt(2*(evaluatePotentials(potential, 
                        1e6*u.kpc, 0*u.kpc, phi=phi[i], quantity=True) \
                        - evaluatePotentials(potential, R[i], z[i], 
                        phi=phi[i],quantity=True)))


def backprop(self, potential, dt=0.1*u.Myr, tint_max = 100*u.Myr, \
                  solarmotion=[-11.1, 12.24, 7.25], zo=0.0208*u.kpc, \
                orbit_path='./', equatorialSave = True, cartesianSave = True):

    '''
    Propagates the sample in the Galaxy backwards in time.

    Requires
    ----------
    potential : galpy potential instance
            Potentialused to integrate the orbits

    Optional
    ----------
    dt : astropy quantity (time)
            Integration timestep. Defaults to 0.1 Myr

    tint_max : astropy quantity (time)
            Maximum backwards integration time.

    solarmotion : length-3 list of floats
            UVW Solar peculiar velocity in km/s. 
            Galpy likes the U to be sign-flipped. Default is Schonrich+2010

    zo : Float or astropy distance quantity
             Offset of the Sun above or below the Galactic plane.
             Default is 20.8 pc (Bennett+Bovy 2019)
             If float, units are assumed to be kpc

    orbit_path : None or string
            Orbits are saved to orbit_path.

    equatorialSave : Boolean
            If True, backwards trajectories in the equatorial frame (ra, dec, 
            distance, proper motion, radial velocity) are saved to file.

    cartesianSave : Boolean
            If True, backwards trajectories in the Galactocentric Cartesian
            frame (x, y, z, vx, vy, vz) are saved to file.

    '''
    #Propagate a sample backwards in time. Probably obsolete

    from galpy.orbit import Orbit
    from galpy.util.conversion import get_physical
    from astropy.table import Table
    import astropy.coordinates as coord

    #Creates path if it doesn't already exist
    if not os.path.exists(orbit_path):
            os.mkdir(orbit_path)

    self.solarmotion=solarmotion

    # Integration time step
    self.dt = dt

    # Number of integration steps
    nsteps = int(np.ceil((tint_max/self.dt).to('1').value))

    # Initialize
    self.orbits = [None] * self.size

    #Integration loop for the n=self.size orbits
    for i in tqdm(range(self.size),desc='Backpropagating...'):

        if platform.system!='Windows':
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(5)

        ts = np.linspace(0, -1, nsteps)*tint_max

        #Initialize orbit instance using astrometry and motion of the Sun,
        self.backorbits[i] = Orbit(vxvv = [self.ra[i], self.dec[i], 
                                    self.dist[i], self.pmra[i], self.pmdec[i], 
                                    self.vlos[i]], 
                                    solarmotion=self.solarmotion, 
                                    radec=True, zo=zo, 
                                    **get_physical(potential))

        self.backorbits[i].integrate(ts, potential, method='dopr54_c')

        # Uncomment these and comment the rest of the lines in the for loop to return only final positions
        #self.dist[i], self.ll[i], self.bb[i], 
        #self.pmll[i], self.pmbb[i], self.vlos[i] = \
        #           self.orbits[i].dist(self.tflight[i], use_physical=True), \
        #           self.orbits[i].ll(self.tflight[i], use_physical=True), \
        #           self.orbits[i].bb(self.tflight[i], use_physical=True), \
        #           self.orbits[i].pmll(self.tflight[i], use_physical=True) , \
        #           self.orbits[i].pmbb(self.tflight[i], use_physical=True)  , \
        #           self.orbits[i].vlos(self.tflight[i], use_physical=True)

        self.backra, self.backdec, self.backdist, \
            self.backpmra, self.backpmdec, self.backvlos = \
                    self.backorbits[i].ra(ts, quantity=True), \
                    self.backorbits[i].dec(ts, quantity=True), \
                    self.backorbits[i].dist(ts, quantity=True), \
                    self.backorbits[i].pmra(ts, quantity=True), \
                    self.backorbits[i].pmdec(ts, quantity=True), \
                    self.backorbits[i].vlos(ts, quantity=True) 

        self.backx, self.backy, self.backz, self.backvx, self.backvy, \
                self.backvz = \
                    self.backorbits[i].x(ts, quantity=True), \
                    self.backorbits[i].y(ts, quantity=True), \
                    self.backorbits[i].z(ts, quantity=True), \
                    self.backorbits[i].vx(ts, quantity=True), \
                    self.backorbits[i].vy(ts, quantity=True), \
                    self.backorbits[i].vz(ts, quantity=True)

        #L = np.sqrt((self.backy*self.backvz - self.backz*self.backvy)**2. + (self.backx*self.backvz-self.backz*self.backvx)**2. + (self.backx*self.backvy-self.backy*self.backvx)**2.)

        #r = np.sqrt(self.backx**2 + self.backy**2 + self.backz**2)

        #Assembles table of the equatorial/Cartesian coordinates 
        #for the particular star in each timestep
        datalist=[ts]
        namelist = ['t']        

        if cartesianSave:
            datalist.extend([self.backx, self.backy, self.backz, self.backvx, \
                    self.backvy, self.backvz])
            namelist.extend(['x', 'y', 'z', 'v_x', 'v_y', 'v_z'])

        if equatorialSave:
            datalist.extend([self.backra, self.backdec, self.backdist, \
                    self.backpmra, self.backpmdec, self.backvlos])
            namelist.extend(['ra', 'dec', 'dist', 'pm_ra', 'pm_dec', 'vlos'])

        data_table = Table(data=datalist, names=namelist)
        data_table.write(orbit_path+'/flight'+str(i)+'_backprop.fits', \
                            overwrite=True)
    
    if platform.system()!='Windows':
        signal.alarm(0)

def _R(self, vx, vy, vz, x, y, z, separate=False):
    '''
    Ejection rate distribution for likelihood. See Contigiani+2018

    Parameters
    ----------
        vx, vy, vz : Quantity
            HVS velocity at ejection point in Galactocentric Cartesian coords
        x, y, z : Quantity
            HVS position in Galactocentric Cartesian coords
        separate: Boolean
            if True, returns the L-only and r-only components of the distribution function. Helpful for debugging or adjusting the kernel size

    '''

    #Get the Galactocentric distance and velocity
    r = np.sqrt(x**2. + y**2. + z**2.)
    v = np.sqrt(vx**2. + vy**2. + vz**2.)

    #Get the angular momentum
    L = np.sqrt((y*vz - z*vy)**2. + (x*vz-z*vx)**2. + (x*vy-y*vx)**2.)

    r = ((r - self.centralr)/self.sigmar).to(1).value # normalized r
    L = (L/self.sigmaL).to(1).value # normalized L

    #Boundaries:
    idx = (r < self.Nsigma) & (r>=0) & (L <self.Nsigma)
    resultfull = np.full(r.shape, np.nan)
    resultfull[~idx] = 0

    resultfull[idx] = np.exp(-np.power(r[idx], 2.)/2.) * np.exp(-np.power(L[idx], 2.)/2.)

    if separate:
        #Boundaries:
        idx = (L <self.Nsigma)
        resultL = np.full(L.shape, np.nan)
        resultL[~idx] = 0

        resultL[idx] = np.exp(-np.power(L[idx], 2.)/2.)

        #Boundaries:
        idx = (r < self.Nsigma) & (r>=0)
        resultr = np.full(r.shape, np.nan)
        resultr[~idx] = 0

        resultr[idx] = np.exp(-np.power(r[idx], 2.)/2.)

        return resultfull, resultL, resultr 
    
    else:

        return resultfull

def likelihood(self, potential, dt=0.005*u.Myr, centralr = 3.*u.pc, \
               sigmar = 10.*u.pc, sigmaL = 10*u.pc*u.km/u.s, Nsigma = 4, \
                xi = 0, solarmotion=[-11.1, 12.24, 7.25], individual=False, weights=None, separate=False, tint_max=None):
    '''
    Computes the non-normalized ln-likelihood of a given potential and ejection model for a given potential.
    When comparing different ejection models or biased samples, make sure you renormalize the likelihood
    accordingly. See Contigiani+ 2018.

    Can return the ln-likelihoods of individual stars if individual is set to True.

    Parameters
    ----------
    potential : galpy potential
        Potential to be tested and to integrate the orbits with.
    ejmodel : EjectionModel
        Ejectionmodel to be tested.
    individual : bool
        If True the method returns individual likelihoods. The default value is False.
    weights : iterable
        List or array containing the weights for the ln-likelihoods of the different stars.
    xi : float or array
        Assumed metallicity for stellar lifetime
    tint_max : Quantity
        Integrate back all stars only for a time tint_max.
    separate: Boolean
        if True, returns the L-only and r-only components of the distribution function. Helpful for debugging or adjusting the kernel size

    Returns
    -------

    log likelihood values : numpy.array or float
        Returns the ln-likelihood of the entire sample or the log-likelihood for every single star if individual
        is True.

    '''
    from galpy.orbit import Orbit
    import astropy.coordinates as coord
    from galpy.util.conversion import get_physical
    #from hvs.utils import t_MS

    if (not hasattr(self,'propagated') or not self.propagated):
        print('Warning: Sample appears to not have been propagated. Calculating likelihood regardless.')

    if(self.size > 1e3):
        print("You are computing the likelihood of a large sample. This might take a while.")

    weights = np.array(weights)
    if((weights != None) and (weights.size != self.size)):
        raise ValueError('The length of weights must be equal to the number of HVS in the sample.')

    self.solarmotion = solarmotion       
    self.backwards_orbits = [None] * self.size
    self.back_dt = dt
    self.lnlike = np.ones(self.size) * (-np.inf)
    self.lnlikeL = np.ones(self.size) * (-np.inf)
    self.lnliker = np.ones(self.size) * (-np.inf)
    self.minL = np.ones(self.size) * (np.inf)*u.kpc*u.km/u.s
    self.minr = np.ones(self.size) * (np.inf)*u.kpc
    self.L0 = np.ones(self.size) * (np.inf)*u.kpc*u.km/u.s

    self.centralr = centralr
    self.sigmar = sigmar
    self.sigmaL = sigmaL
    self.Nsigma = Nsigma

    if(tint_max is None):
        lifetime = t_MS(self.m, xi)
        lifetime[lifetime>self.T_MW] = self.T_MW
    else:
        lifetime = tint_max*np.ones(self.size)

    #number of integration steps
    nsteps = np.ceil((lifetime/self.back_dt).to('1').value).astype(int)
    nsteps[nsteps<100] = 100

    for i in tqdm(range(self.size),desc='Backpropagating...'):
        
        #ts = np.linspace(0, 1, nsteps[i])*lifetime[i]
        ts = np.linspace(0, -1, nsteps[i])*lifetime[i]

        self.backwards_orbits[i] = Orbit(vxvv = [self.ra[i], self.dec[i], 
                    self.dist[i], self.pmra[i], self.pmdec[i], self.vlos[i]], \
                                solarmotion=self.solarmotion,zo=20.8*u.pc,**get_physical(potential), radec=True)#.flip()
        
        self.backwards_orbits[i].integrate(ts, potential, method='dopr54_c')

        x, y, z, vx, vy, vz = self.backwards_orbits[i].x(ts, quantity=True), \
                                            self.backwards_orbits[i].y(ts, quantity=True), \
                                            self.backwards_orbits[i].z(ts, quantity=True), \
                                            self.backwards_orbits[i].vx(ts, quantity=True), \
                                            self.backwards_orbits[i].vy(ts, quantity=True), \
                                            self.backwards_orbits[i].vz(ts, quantity=True)

        L = np.sqrt((y*vz - z*vy)**2. + (x*vz-z*vx)**2. + (x*vy-y*vx)**2.)

        if separate:
            like, likeL, liker = self._R(vx, vy, vz, x, y, z, separate=True)
            
            self.lnlike[i] = np.log( like.sum() )
            self.lnlikeL[i] = np.log( likeL.sum())
            self.lnliker[i] = np.log( liker.sum())
        else:
            like = self._R(vx, vy, vz, x, y, z)

            self.lnlike[i] = np.log  ( like.sum() )

        self.minL[i] = np.min(L)
        self.minr[i] = np.min(np.sqrt(x**2 + y**2 + z**2))
        self.L0[i] = L[0]

        if((self.lnlike[i] == -np.inf) and (not individual)):
            break

    if(individual):
        if separate:
            return self.lnlike, self.lnlikeL, self.lnliker
        else:
            return self.lnlike
    
    else:
        if separate:
            return self.lnlike.sum(), self.lnlikeL.sum(), self.lnliker.sum()
        else:                
            return self.lnlike.sum()

def get_betas(self):
    '''
    Calculates angle between HVS position and velocity vectors in the Galactocentric frame
    ------
    Returns:
        beta_theta: array
            polar angle between velocity and position
        beta_phi: array
            azimuthal angle between velocity and position
    '''

    R = np.sqrt(self.x**2+self.y**2)
    vR = np.sqrt(self.vx**2+self.vy**2)

    dot = (self.z*self.vz+ R * vR)
    cross = (self.z*vR - R*self.vz)

    self.beta_theta = np.arctan2(cross,dot)
     
    dot = (self.x*self.vx+self.y*self.vy)
    cross = (self.x*self.vy-self.y*self.vx)

    self.beta_phi = np.arctan2(cross,dot)

def get_betas_Boubert(self):
    '''
    DEPRECIATED. Calculates HVS deflection angle by solving for polar deflection between GC position and velocity vectors
        Uses Dot product method: |a||b|cos(phi) = a*b
    ------
    Paramters
        file: string 
            path to the file 
        T: bool
            if Ture, returns only observable HVSs
    Returns:
        theta: array
            angle between the position and velocity vector for HVS sample
        index: array 
            1d array of size n corresponding to the number of HVS in the sample
    '''

    R = np.sqrt(self.x**2+self.y**2)
    vR = np.sqrt(self.vx**2+self.vy**2)

    phi_pos = np.arctan2(self.y,self.x)
    phi_vel = np.arctan2(self.vy,self.vx)

    theta_pos = np.arctan2(self.z,R)
    theta_vel = np.arctan2(self.vz,vR)

    self.beta_theta_boubert = theta_vel - theta_pos
    self.beta_phi_boubert = phi_vel - phi_pos
