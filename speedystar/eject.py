__ImportError__ = "One or more required external packages have not been installed. Please see requirements.txt"

import time

try:
    import numpy as np
    from tqdm import tqdm
    import astropy.constants as const
    from astropy import units as u
    from amuse.units import units
    from amuse.community.sse.interface import SSE
    from amuse.test.amusetest import get_path_to_results
    from amuse import datamodel
except ImportError:
    raise ImportError(__ImportError__)

from .utils.imfmaster.imf import imf

class EjectionModel:
    '''
        Ejection model class
    '''

    _name = 'Unknown'

    def __init__(self, name):
                self._name = name

    def sampler(self):
        '''
        Sampler of the ejection distribution
        '''
        raise NotImplementedError

class Hills(EjectionModel):

    '''
    HVS ejection model from Rossi+ 2017. Isotropic ejection from 3 pc from GC 
    and with a mass/velocity distribution based on MC. Can generate an 
    ejection sample using a Monte Carlo approach based on inverse transform 
    sampling. See also Marchetti+ 2018, Evans+2021, Evans+2022.

    Attributes
    ---------

    _name : string
        Name of the Ejection method
    v_range : Quantity
        Allowed range of HVS initial velocities
    m_range : Quantity
        Allowed range of HVS masses
    T_MW : Quantity
        Milky Way lifetime
    M_BH : Quantity
        Mass of the BH at the GC
    alpha : float
        Exponent of the power-law for the distribution of the semi-major axes 
        in binaries. Default is -1
    gamma : float
        Exponent of the power-law for the distribution of the mass ratio 
        in binaries. Default is 0
    kappa : 
        Exponent of the power-law initial mass function (or the m>0.5) 
        mass function. Default is -2.3
    '''

    T_MW = 13.8*u.Gyr # MW maximum lifetime from Planck2015

    #Initial ejection radius from GC
    centralr = 3*u.pc

    def __init__(self, M_BH = 4e6*u.Msun, Met=0.0, Zsun=0.0142, \
                tflightmax = 100.*u.Myr, name_modifier = None, alpha=-1, \
                gamma=0, rate=1e-4/u.yr, kappa=2.3, \
                v_range = [500, 5e4]*u.km/u.s, m_range = [0.5, 1e3]*u.Msun):

        '''
        Parameters
        ----------
        name_modifier : str
            Add a modifier to the class name

        M_BH: astropy Msun quantity
            Change mass of Sgr A*

        alpha/gamma/kappa: floats
           Change parameters that describe how Hills projenitor binaries 
           are drawn

        Met: float
           Metallicity log10(Z/Zsun)
           ^(Obselete, metallicity is drawn randomly now)

        Zsun : float
            Total metallicity of the sun. Default is 0.0142 (Asplund+2009) 
            0.02 is another common choice (Anders & Grevasse 1989)

         tflightmax : float
            Stars with a flight time more than tflightmax are tossed out 
            and aren't propagated. Provides a lot of speedup, as long as you're
            certain stars with tflight<tflightmax are not relevant for 
            your science case

        rate : Quantity
            Assumed ejection rate from the SMBH

        v_range : list Quantity
            Stars drawn with initial velocities outside this range are
            immediately discarded

        m_range : list Quantity
            Stars with masses outside this range are immediately discarded
        '''

        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.M_BH = M_BH
        self.Met = Met
        self.Zsun = Zsun
        self.tflightmax = tflightmax
        self.eta = rate
        self.v_range = v_range
        self.m_range = m_range

        if(name_modifier is not None):
            self._name = 'Hills - ' + name_modifier
        else:
            self._name = 'Hills'

    def _evo_star(self,mass,age=None,met=0.0):
        #Evolve a star of a certain mass and metallicity until a certain age 
        #using the SSE module within AMUSE

        #Initialize
        stellar_evolution = SSE()
        #Adjust metallicity for new Zsun assumption
        stellar_evolution.parameters.metallicity = self.Zsun*10**(met)
        star      = datamodel.Particles(len(mass))
        star.mass = mass | units.MSun

        if age is not None:
            age = age.to('Myr').value | units.Myr

        #Evolve the star
        star = stellar_evolution.particles.add_particles(star)
        stellar_evolution.commit_particles()
        stellar_evolution.evolve_model(end_time = age)

        stellar_evolution.stop()

        return star

    def _inverse_cumulative_mp_broken(self, x):
        '''
            Inverse of the CDF of a broken power law IMF as a function of x
            By default, breaks and low/intermediate regime slopes are set as 
            in Kroupa, high-end slope can be an argument to __init__().
            A caveat: if m_range[0] is less than break1 
            (as is the case by default), this class throws an error.
            You can avoid that by saying p1=p2 and setting break1 to anywhere 
            in between mmin and break2, as in the default class call here

        '''
        from imfmaster.imf import imf #%or wherever else you installed it to

        myimf = imf.Kroupa(mmin=0.1,mmax=100,p1=1.3,p2=1.3,p3=self.kappa,
                            break1=0.2,break2=0.5)

        #Uncomment to get a proper Kroupa IMF
        #myimf = imf.Kroupa(mmin=0.05,mmax=900,p1=0.3,p2=1.3,p3=self.kappa,
                            #break1=0.08,break2=0.5)

        result = imf.inverse_imf(x,massfunc=myimf)
        return result

    def _inverse_cumulative_mp(self, x):
        '''
            Inverse of the CDF of a single power law IMF as a function of x
        '''

        from .utils.imfmaster.imf import imf

        testimf = imf.Salpeter(alpha=self.kappa,mmin=0.1,mmax=100)
        result = imf.inverse_imf(x,mmin=0.1,mmax=100,massfunc=testimf)
        return result

    def _inverse_cumulative_q(self, x, mp):
        '''
           Inverse of the CDF of a single power law mass ratio distribution 
            as a function of x
        '''

        qmin = 0.1
        qmax = 1.

        if self.gamma==-1:
            return qmin*(qmax/qmin)**x
        else:
            q = (  (qmax**(1.+self.gamma) - qmin**(1.+self.gamma))*x \
                    + qmin**(1.+self.gamma) )**(1./(1.+self.gamma))
            return q 

    def _inverse_cumulative_logP(self,x,Rmax,mtot):

        '''
           Inverse of the CDF of a single power law log-period 
            distribution as a function of x
        '''

        #Pmin is set by Roche lobe overflow, Pmax is (arbitrarily) 2000 Rsun
        Pmin = np.log10(np.sqrt((4*np.pi**2 * (2.5*Rmax)**3) \
                        / (const.G*mtot)).to('second').value)
        Pmax = np.log10(np.sqrt((4*np.pi**2 * (2e3*u.Rsun)**3) \
                        / (const.G*mtot)).to('second').value)

        if self.alpha==-1:
            return (10**(Pmin*(Pmax/Pmin)**x)) * u.second
        else:
            return (  10**( ((Pmax**(1.+self.alpha) - Pmin**(1.+self.alpha))*x \
                    + Pmin**(1.+self.alpha) )**(1./(1.+self.alpha))))*u.second

    def sampler(self):
        '''
        Samples from the ejection distribution to generate an ejection sample.
        The distribution mass and velocity is generated using a Monte Carlo 
        approach (Rossi 2014). The functions _inverse_* dictate the 
        parameters of the progenitor binary population. 
        The velocity vector is assumed radial.

        Returns
        -------
            r0, phi0, theta0, v0, phiv0, thetav0 : Quantity
                Initial phase space position in spherical coordinates, 
                centered on the GC
            m, tage, tflight : Quantity
                Stellar mass of the HVS, age at observation and tflight 
                between ejection and observation      
            stage, stagebefore : int
                Evolutionary stage (e.g. main sequence, red giant) of your HVS 
                *today* and at the moment of ejection.
                Stage conventions follow Hurley et al. (2000) 
                https://ui.adsabs.harvard.edu/abs/2000MNRAS.315..543H/abstract             
            n : int
                Size of the output ejection sample

        '''
    
        import time
        from math import ceil

        try:
            from astropy import constants as const
            from astropy.table import Table
        except ImportError:
            raise ImportError(__ImportError__)

        from .utils.hurley_stellar_evolution import get_t_BAGB, get_t_MS

        PI = np.pi

        # Sample the binary properties q, a, mp using inverse sampling
        #Take only stars ejected within the last tflightmax Myr

        n       = np.rint((self.tflightmax*self.eta).decompose()).astype(int)
        tflight = np.random.uniform(0., self.tflightmax.to('Myr').value, n) \
                    *u.Myr

        # Inverse sampling a primary mass and mass ratio
        uniform_for_mp, uniform_for_q = np.random.uniform(0, 1, (2, n))
        mp = self._inverse_cumulative_mp(uniform_for_mp)
        q  = self._inverse_cumulative_q(uniform_for_q, mp)
        mp = mp*u.Msun

        #Randomly designate one member of the binary as the HVS and 
        #the other as the companion (C)
        ur  = np.random.uniform(0,1,n)
        idx = ur>=0.5

        #mem is 1 if the ejected HVS is the heavier star of the binary, 
        #2 if the lighter.
        #Generally not super helpful but can be good for debugging
        mem_HVS       = np.zeros(n) 
        mem_HVS[idx]  = 1
        mem_HVS[~idx] = 2

        M_HVS, M_C    = np.zeros(n)*u.Msun, np.zeros(n)*u.Msun
        M_HVS[idx]    = mp[idx]
        M_HVS[~idx]   = mp[~idx]*q[~idx]
        M_C[idx]      = mp[idx]*q[idx]
        M_C[~idx]     = mp[~idx]

        #Remove stars outside the imposed mass range
        idx        = (M_HVS > self.m_range[0]) & (M_HVS < self.m_range[1])
        tflight, q = tflight[idx], q[idx]
        m, mc, mem = M_HVS[idx], M_C[idx], mem_HVS[idx]

        n = idx.sum()

        #Solar metallicity
        #met = np.zeros(n) 

        #Uncomment this line to give each HVS a metallicity xi = log10(Z/Zsun) 
        #randomly distributed between -0.25 and +0.25 
        met = np.round(np.random.uniform(-0.25, 0.25, n),decimals=2)       

        #Calculate stellar ages and remove stars that are remnants today
        #Calculating maximum lifetimes (taken here as the time from ZAMS 
        #to the beginning of the AGB branch) takes a while.
        #Calculating main sequence lifetimes is much faster, can save time by 
        #pre-selecting using the MS lifetime.
        #We cut out stars older than 1.4 times their main sequence lifetime
        #(they are definitely dead)
        #For the rest, we calculate proper maximum lifetimes

        #Calculate the main sequence lifetime of the heavier star in the 
        #binary using the Hurley+2000 relation. These relations assume 
        #Zsun=0.02, so might need to be adjusted if self.Zsun is different

        T_maxbig = get_t_MS(np.maximum(m,mc),met+np.log10(self.Zsun/0.02))
        T_maxbig[T_maxbig>self.T_MW] = self.T_MW

        #Binary hung around the GC for a random fraction e1 
        #of its maximum lifetime
        #("maximum lifetime" here is taken as 1.4*t_MS, see above)
        e1       = np.random.random(n)

        t_before = 1.4*T_maxbig*e1

        #Calculate time remaining for the binary
        t_rest   = 1.4*T_maxbig*(1-e1)

        #Calculate the HVSs maximum lifetime. This may or may not be the same 
        #as the binary's maximum lifetime. 
        #It's longer if the HVS is the smaller member
        T_maxHVS       = T_maxbig.copy()
        T_maxHVS[m<mc] = get_t_MS(m[m<mc],met[m<mc]+np.log10(self.Zsun/0.02))
        T_maxHVS[T_maxHVS>self.T_MW] = self.T_MW

        #Calculate HVS's age
        tage = np.zeros(len(tflight))*u.Myr
        tage = t_before + tflight

        #Remove HVS from sample if its age is >1.4 times 
        #its main sequence lifetime
        tage[tage>=1.4*T_maxHVS] = np.nan

        idx = (~np.isnan(tage))

        m, mc, tage          = m[idx], mc[idx], tage[idx]
        tflight, q, mem, met = tflight[idx], q[idx], mem[idx], met[idx]
        e1                   = e1[idx]

        n = idx.sum()

        print('Getting primary star maximum lifetime')

        #Redo the last few steps, but now calculating actual 
        #proper lifetimes for the stars
        T_maxbig = get_t_BAGB(np.maximum(m,mc),met+np.log10(self.Zsun/0.02))
        T_maxbig[T_maxbig>self.T_MW] = self.T_MW

        t_before = T_maxbig*e1
        t_rest = T_maxbig*(1-e1)

        print('Getting HVS maximum lifetime')
        T_maxHVS = T_maxbig.copy()
        T_maxHVS[m<mc] = get_t_BAGB(m[m<mc],met[m<mc]+np.log10(self.Zsun/0.02))        
        T_maxHVS[T_maxHVS>self.T_MW] = self.T_MW

        tage = np.zeros(len(tflight))*u.Myr
        tage = t_before + tflight
        tage[tage>=T_maxHVS] = np.nan        

        idx = (~np.isnan(tage))

        m, mc                      = m[idx], mc[idx], 
        tage, t_before, tflight = tage[idx], t_before[idx], tflight[idx]
        q, mem, met                = q[idx], mem[idx], met[idx]

        n                          = idx.sum()

        #Need the larger star's size to draw an orbital period

        #Get the mass of the larger star in the binary
        mbig = m.copy()
        mbig[mc>m] = mc[mc>m]

        Rmax, stagebefore, R, Lum, stage, T_eff = (np.empty(n) \
                                                for i in range(6))

        #Stars with different metallicities can't be evolved at the same time 
        #with AMUSE (or so it seems to me), so need to do them in batches

        #Loop over unique metallicities
        #Should update, could be sped up
        print('Evolving stars...')
        #for z in np.unique(met):
        for z in tqdm(np.unique(met)):   

            #indices with ith metallicity, subset
            idx = np.where(met==z)[0]

            #Evolve larger star in each of the binaries
            star = self._evo_star(mbig[idx].value,t_before[idx],met=z)
            
            #Extract bigger star radii and stages
            Rmax[idx] = star.radius.as_astropy_quantity().to('Rsun').value
            stagebefore[idx] = star.stellar_type.as_astropy_quantity()

            #Evolve each HVS
            star = self._evo_star(m[idx].value,t_before[idx],met=z)

            #Extract HVS effective temperature, radius, 
            #luminosity, evolutionary stage
            T_eff[idx] = star.temperature.as_astropy_quantity().to('K').value
            R[idx] = star.radius.as_astropy_quantity().to('Rsun').value 
            Lum[idx] = star.luminosity.as_astropy_quantity().to('Lsun').value 
            stage[idx] = star.stellar_type.as_astropy_quantity()

        Rmax = Rmax*u.Rsun
        R = R*u.Rsun
        Lum = Lum*u.Lsun
        T_eff = T_eff*u.K

        #Inverse sample an orbital period for the binary
        uniform_for_P = np.random.uniform(0, 1, n)
        P = self._inverse_cumulative_logP(uniform_for_P,Rmax,m+mc)
        P = P.to('day')

        #Convert this to an orbital separation, Kepler's 3rd law
        sep = np.cbrt((const.G*(m+mc)*P**2)/(4*np.pi**2)).to('Rsun')

        #Calculate velocity, see Sari+2010, Kobayashi+2012, Rossi+2014
        V_HVS = (np.sqrt( 2.*const.G.cgs*mc / sep )  \
                * ( self.M_BH/(mc+m) )**(1./6.)).to('km/s')

        #Cut out stars that are outside the imposed velocity range
        idx = (V_HVS > self.v_range[0]) & (V_HVS < self.v_range[1])

        m, mc                   = m[idx], mc[idx] 
        tage, t_before, tflight = tage[idx], t_before[idx], tflight[idx] 
        q, mem, met             = q[idx], mem[idx], met[idx]
        v, P, sep               = V_HVS[idx], P[idx], sep[idx] 
        stagebefore, stage      = stagebefore[idx], stage[idx]
        T_eff, Lum, R           = T_eff[idx], Lum[idx], R[idx]

        n = idx.sum()

        #You can use this line below to cut stars that aren't relevant for you, 
        #e.g. stars that are too small, wrong evolutionary stage, etc.
        #Can save a lot of computation time
        idx = (v>0.*u.km/u.s)
        
        # Distance from GC at 3 pc
        r0 = np.ones(n)*self.centralr

        # Isotropic position unit vector in spherical coordinates
        phi0 = np.random.uniform(0,2*PI, n)*u.rad
        theta0 = np.arccos( np.random.uniform(-1,1, n))*u.rad

        # The velocity vector points radially.
        phiv0 = phi0.copy()
        thetav0 = theta0.copy()
    
        return r0[idx], phi0[idx], theta0[idx], v[idx], phiv0[idx], \
                    thetav0[idx], m[idx], tage[idx], tflight[idx], \
                    sep[idx], P[idx], q[idx], mem[idx], met[idx], \
                    stage[idx], stagebefore[idx], R[idx], T_eff[idx], \
                    Lum[idx], len(r0[idx])
