import numpy as nu
class ForceSoftening:
    """class representing a force softening kernel"""
    def __init__(self): #pragma: no cover
        pass

    def __call__(self,d): #pragma: no cover
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the force of the softening kernel
        INPUT:
           d - distance
        OUTPUT:
           softened force (amplitude; without GM)
        HISTORY:
           2011-04-13 - Written - Bovy (NYU)
        """
        raise AttributeError("'__call__' not implemented for this softening kernel")

    def potential(self,d): #pragma: no cover
        """
        NAME:
           potential
        PURPOSE:
           return the potential corresponding to this softening kernel
        INPUT:
           d - distance
        OUTPUT:
           potential (without GM)
        HISTORY:
           2011-04-13 - Written - Bovy (NYU)
        """
        raise AttributeError("'potential' not implemented for this softening kernel")

    def density(self,d): #pragma: no cover
        """
        NAME:
           density
        PURPOSE:
           return the density corresponding to this softening kernel
        INPUT:
           d - distance
        OUTPUT:
           density (without GM)
        HISTORY:
           2011-04-13 - Written - Bovy (NYU)
        """
        raise AttributeError("'density' not implemented for this softening kernel")


class LMCSoftening (ForceSoftening):
    from astropy import units as u
    """class representing a Plummer softening kernel"""
    def __init__(self,softening_length=0.00):
        from galpy.potential import  HernquistPotential
        from astropy import units as u
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a Plummer softening kernel
        INPUT:
           softening_length=
        OUTPUT:
        HISTORY:
           2011-04-13 - Written - Bovy (NYU)
        """
        self._softening_length= softening_length

    def __call__(self,R,z):
        from galpy.potential import  HernquistPotential
        from astropy import units as u
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the force of the softening kernel
        INPUT:
           d - distance
        OUTPUT:
           softened force (amplitude; without GM)
        HISTORY:
           2011-04-13 - Written - Bovy (NYU)
        """
        #return d/(d**2.+self._softening_length**2.)**1.5

        #LMC parameters
        Mlmc = 1.5*1e11*u.Msun
        Rlmc = 17.14*u.kpc

        pot=HernquistPotential(amp=2*Mlmc,a=Rlmc, normalize=False)
        return pot.Rforce(R,z)

    def potential(self,R,z):
        from galpy.potential import  HernquistPotential, evaluatePotentials
        from astropy import units as u
        """
        NAME:
           potential
        PURPOSE:
           return the potential corresponding to this softening kernel
        INPUT:
           d - distance
        OUTPUT:
           potential (without GM)
        HISTORY:
           2011-04-13 - Written - Bovy (NYU)
        """

        #LMC parameters
        Mlmc = 1.5*1e11*u.Msun
        Rlmc = 17.14*u.kpc
        pot=HernquistPotential(amp=2*Mlmc,a=Rlmc, normalize=False)
        #return pot.evaluatePotentials(R,z)
        print([R,z])
        #print(pot.__call__(R,z))
        print(evaluatePotentials(pot,R,z))
        return evaluatePotentials(pot,R,z,t=0)
        #return pot.__call__(R,z)

    def density(self,R,z):
        """
        NAME:
           density
        PURPOSE:
           return the density corresponding to this softening kernel
        INPUT:
           d - distance
        OUTPUT:
           density (without GM)
        HISTORY:
           2011-04-13 - Written - Bovy (NYU)
        """

        #LMC parameters
        Mlmc = 1.5*1e11*u.Msun
        Rlmc = 17.14*u.kpc

        pot=HernquistPotential(amp=2*Mlmc,a=Rlmc, normalize=False)
        return pot.evaluateDensities(R,z)
