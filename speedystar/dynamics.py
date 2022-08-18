
__ImportError__ = "One or more required external packages have not been installed. Please see requirements.txt"

import os

try:
    from astropy import units as u
    import numpy as np
    from tqdm import tqdm
except ImportError:
    raise ImportError(__ImportError__)

#@propagate
def propagate(self, potential, dt=0.1*u.Myr, 
                  solarmotion=[-11.1, 12.24, 7.25],
                    zo=0.0208*u.kpc, orbit_path=None):

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
        '''

        import signal

        try:
            from galpy.orbit import Orbit
            from galpy.util.conversion import get_physical
            from astropy.table import Table
        except ImportError:
            raise ImportError(__ImportError__)

        def handler(signum, frame):
            print('OOPS! A star took to long to integrate.'\
                  'Problematic star at index')

        self.solarmotion=solarmotion

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

        self.orbits                          = [None] * self.size

        #Integration loop for the self.size orbits
        print('Propagating...')
        for i in tqdm(range(self.size)):

            # Galpy will hang on propagating an orbit on rare occasion. 
            # Is only a problem if the flight time is long and/or the star 
            # makes multiple close pericentres to the SMBH. 
            # The signal alarm prevents this
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(5)

            #Get timesteps
            ts = np.linspace(0, 1, nsteps[i])*self.tflight[i]

            #Initialize orbit using galactocentric cylindrical 
            #phase space info of stars
            self.orbits[i] = Orbit(vxvv = [rho[i], vR[i], vT[i], z[i], vz[i], \
                                     phi[i]], solarmotion=self.solarmotion, \
                                     **get_physical(potential))

            self.orbits[i].integrate(ts, potential, method='dopr54_c')

            # Export the final position
            self.ra[i] = self.orbits[i].ra(ts, use_physical=True)[-1]#*u.deg
            self.dec[i] = self.orbits[i].dec(ts, use_physical=True)[-1]#*u.deg
            self.pmra[i] = self.orbits[i].pmra(ts, use_physical=True)[-1]#*u.mas/u.yr
            self.pmdec[i] = self.orbits[i].pmdec(ts, use_physical=True)[-1]#*u.mas/u.yr
            self.dist[i] = self.orbits[i].dist(ts, use_physical=True)[-1]#*u.mas/u.yr
            self.vlos[i] = self.orbits[i].vlos(ts, use_physical=True)[-1]#*u.km/u.s
            self.par[i] = u.mas / self.dist[i].to('kpc').value
            self.x[i] = self.orbits[i].x(ts, use_physical=True)[-1]#*u.kpc
            self.y[i] = self.orbits[i].y(ts, use_physical=True)[-1]#*u.kpc
            self.z[i] = self.orbits[i].z(ts, use_physical=True)[-1]#*u.kpc
            self.vx[i] = self.orbits[i].vx(ts, use_physical=True)[-1]#*u.km/u.s
            self.vy[i] = self.orbits[i].vy(ts, use_physical=True)[-1]#*u.km/u.s
            self.vz[i] = self.orbits[i].vz(ts, use_physical=True)[-1]#*u.km/u.s

            #Save the orbits of each HVS, if orbit_path is supplied
            if orbit_path is not None:
                #Only saves the orbits of the first 5e5 HVSs to prevent bloat

                if not os.path.exists(orbit_path):
                    raise SystemExit('Path '+orbit_path+' does not exist')

                if i<50000:
                    self.testra, self.testdec, self.testdist, self.testpmra, \
                        self.testpmdec, self.testvlos = \
                        self.orbits[i].ra(ts, use_physical=True)*u.deg, \
                        self.orbits[i].dec(ts, use_physical=True)*u.deg, \
                        self.orbits[i].dist(ts, use_physical=True)*u.kpc, \
                        self.orbits[i].pmra(ts, use_physical=True)*u.mas/u.yr, \
                        self.orbits[i].pmdec(ts, use_physical=True)*u.mas/u.yr, \
                        self.orbits[i].vlos(ts, use_physical=True)*u.km/u.s
             
                    xpos = self.orbits[i].x(ts)
                    ypos = self.orbits[i].y(ts)
                    zpos = self.orbits[i].z(ts)
                    v_x = self.orbits[i].vx(ts)
                    v_y = self.orbits[i].vy(ts)
                    v_z = self.orbits[i].vz(ts)
                    L = self.orbits[i].L(ts)
             
                    #Table of equatorial coordinates for the star with time
                    datalist = [ts, self.testra, self.testdec, self.testdist, 
                                self.testpmra, self.testpmdec, self.testvlos]
                    namelist = ['t', 'ra', 'dec', 'dist', 
                                'pm_ra', 'pm_dec', 'vlos']
                    data_table = Table(data=datalist, names=namelist)

                    #Writes equatorial orbits to file. Each star gets own file
                    data_table.write(orbit_path+'flight'+str(i)+'_ICRS.fits', 
                                     overwrite=True)

                    #Writes cartesian orbits to file. Each star gets own file
                    datalist=[ts, xpos, ypos, zpos, v_x, v_y, v_z, L]
                    namelist = ['t', 'x', 'y', 'z', 'v_x', 'v_y', 'v_z', 'L']
                    data_table = Table(data=datalist, names=namelist)
                    data_table.write(orbit_path+'flight'+str(i)+'_Cart.fits', 
                                     overwrite=True)

        signal.alarm(0)

        #Get Galactocentric distance and velocity and Galactic escape velocity
        #as well as final azimuthal and polar coordinates
        #in Galactocentric sherical coordinates
        if(self.size>0):
            self.get_vesc(potential=potential)
            self.GCdist = np.sqrt(self.x**2. + self.y**2. + self.z**2.).to(u.kpc)
            self.GCv = np.sqrt(self.vx**2. + self.vy**2. + self.vz**2.).to(u.km/u.s)
            self.thetaf = np.arccos(self.z/self.GCdist)
            self.phif = np.arctan2(self.y,self.x)

        else:
            self.GCv = []*u.km/u.s 
            self.GCdist = []*u.kpc
            self.Vesc = []*u.km/u.s 
            self.thetaf = []*u.rad
            self.phif = []*u.rad

#@get_vesc
def get_vesc(self, potential, v=True):

        '''
        Returns the escape speed of a given potential 
        for each star in a propagated sample
        '''

        try:
            from galpy.potential import evaluatePotentials
        except ImportError:
            raise ImportError(__ImportError__)

        self.Vesc = np.zeros(self.size)*u.km/u.s
        
        R = np.sqrt(self.x**2 + self.y**2)
        z = self.z

        for i in range(self.size):
            #self.Vesc[i] = 220*np.sqrt(2*(evaluatePotentials(potential,1e6*u.kpc,0*u.kpc) - evaluatePotentials(potential,R[i],z[i])))*u.km/u.s
            self.Vesc[i] = np.sqrt(2*(evaluatePotentials(potential,1e6*u.kpc,0*u.kpc) - evaluatePotentials(potential,R[i],z[i])))#*u.km/u.s


#@backprop
def backprop(self, potential, dt=0.1*u.Myr, tint_max = 100.*u.Myr, \
                 solarmotion = [-11.1, 12.24, 7.25], threshold=None,  \
                 orbit_path='./'):

    '''
    Propagates the sample in the Galaxy backwards in time.

    Requires
    ----------
    potential : galpy potential instance
            Potential instance of the galpy library used to integrate the orbits

    Optional
    ----------
    dt : Quantity
            Integration timestep. Defaults to 0.1 Myr
    solarmotion : length-3 list of floats
            UVW Solar peculiar velocity in km/s. 
            Galpy likes the U to be sign-flipped. Default is Schonrich+2010
    threshold : float
            Maximum relative energy difference between the initial energy and 
            the energy at any point needed to consider an integration step an 
            energy outliar. E.g. for threshold=0.01, any excess or deficit 
            of 1% (or more) of the initial energy is enough to be registered 
            as outlier. A table E_data.fits is created in the working directory
             containing for every orbit the percentage of outliar points (pol)
    orbit_path : None or string
            Equatorial and Galactocentric Cartesian orbits 
            are saved to orbit_path. Useful for debugging        
    '''


    #Propagate a sample backwards in time. Probably obsolete

    try:
        from galpy.orbit import Orbit
        from galpy.util.coords import pmllpmbb_to_pmrapmdec, lb_to_radec
        from galpy.util.coords import vrpmllpmbb_to_vxvyvz, lbd_to_XYZ
        from galpy.util.conversion import get_physical
        from astropy.table import Table
        import astropy.coordinates as coord
    except ImportError:
        raise ImportError(__ImportError__)

    self.solarmotion=solarmotion

    #solarmotion=[-11.1, 12.24, 7.25]
    #solarmotion = [-8.6, 13.9, 7.1]

    if(threshold is None):
        check = False
    else:
        check = True

    # Integration time step
    self.dt = dt

    # Maximum integration time
    #tint_max = 100.*u.Myr

    # Number of integration steps
    nsteps = int(np.ceil((tint_max/self.dt).to('1').value))

    # Initialize
    self.orbits = [None] * self.size

    #Integration loop for the n=self.size orbits
    for i in range(self.size):
        print(self.name+' star index'+str(i))
        ts = np.linspace(0, 1, nsteps)*tint_max
        #ts = np.linspace(0, 1, nsteps[i])*self.tflight[i]

        #Initialize orbit instance using astrometry and motion of the Sun,
        #.flip() method reverses the orbit so we integrate backwards in time
        self.orbits[i] = Orbit(vxvv = [self.ra[i], self.dec[i], self.dist[i], \
                                self.pmra[i], self.pmdec[i], self.vlos[i]], \
                                    solarmotion=self.solarmotion, radec=True, \
                                    **get_physical(potential)).flip()

        self.orbits[i].integrate(ts, potential, method='dopr54_c')

        # Uncomment these and comment the rest of the lines in the for loop to return only final positions
        #self.dist[i], self.ll[i], self.bb[i], self.pmll[i], self.pmbb[i], self.vlos[i] = \
        #                                    self.orbits[i].dist(self.tflight[i], use_physical=True), \
        #                                    self.orbits[i].ll(self.tflight[i], use_physical=True), \
        #                                    self.orbits[i].bb(self.tflight[i], use_physical=True), \
        #                                    self.orbits[i].pmll(self.tflight[i], use_physical=True) , \
        #                                    self.orbits[i].pmbb(self.tflight[i], use_physical=True)  , \
        #                                    self.orbits[i].vlos(self.tflight[i], use_physical=True)

        self.testra, self.testdec, self.testdist, self.testpmra, self.testpmdec, self.testvlos = \
                    self.orbits[i].ra(ts, use_physical=True), \
                    self.orbits[i].dec(ts, use_physical=True), \
                    self.orbits[i].dist(ts, use_physical=True), \
                    self.orbits[i].pmra(ts, use_physical=True), \
                    self.orbits[i].pmdec(ts, use_physical=True), \
                    self.orbits[i].vlos(ts, use_physical=True) 

        #Creates path if it doesn't already exist
        if not os.path.exists(orbit_path):
            os.mkdir(orbit_path)

        #Assembles table of equatorial coordinates for the star in each timestep
        datalist=[ts, self.testra, self.testdec, self.testdist, \
                    self.testpmra, self.testpmdec, self.testvlos]
        namelist = ['t', 'ra', 'dec', 'dist', 'pm_ra', 'pm_dec', 'vlos']
        data_table = Table(data=datalist, names=namelist)

        #Writes equatorial orbits to file. Each star gets its own file
        data_table.write(orbit_path+'/flight'+str(i)+'.fits', overwrite=True)

        self.testx, self.testy, self.testz, self.testvx, self.testvy, \
                self.testvz = \
                    self.orbits[i].x(ts, use_physical=True), \
                    self.orbits[i].y(ts, use_physical=True), \
                    self.orbits[i].z(ts, use_physical=True), \
                    self.orbits[i].vx(ts, use_physical=True), \
                    self.orbits[i].vy(ts, use_physical=True), \
                    self.orbits[i].vz(ts, use_physical=True)

        datalist=[ts, self.testx, self.testy, self.testz, self.testvx, \
                    self.testvy, self.testvz]
        namelist = ['t', 'x', 'y', 'z', 'v_x', 'v_y', 'v_z']
        data_table = Table(data=datalist, names=namelist)
        data_table.write(orbit_path+'/flight'+str(i)+'_Cart.fits', \
                            overwrite=True)

        # Uncomment these to write final positions only
        # Radial velocity and distance + distance modulus
        #self.vlos, self.dist = self.vlos * u.km/u.s, self.dist * u.kpc

        # Sky coordinates and proper motion
        #data = pmllpmbb_to_pmrapmdec(self.pmll, self.pmbb, self.ll, self.bb, degree=True)*u.mas / u.year
        #self.pmra, self.pmdec = data[:, 0], data[:, 1]
        #data = lb_to_radec(self.ll, self.bb, degree=True)* u.deg
        #self.ra, self.dec = data[:, 0], data[:, 1]

        #datalist=[ts, self.ra, self.dec, self.dist, self.pmra, self.pmdec, self.vlos]
        #namelist = ['t', 'ra', 'dec', 'dist', 'pm_ra', 'pm_dec', 'vlos']
        #data_table = Table(data=datalist, names=namelist)
        #data_table.write('/path/to/where/you/want.fits', overwrite=True)

