from scipy import interpolate
import numpy
import healpy
import h5py

np = numpy
_DEGTORAD = numpy.pi/180.

class DustMap:
    '''
        Loads a dustmap in the h5 format like the ones in mwdust [doi:10.3847/0004-637X/818/2/130]

        Methods
        -------
            query_dust
                Returns the dust extinction E(B-V) in the SFD scale for the selected position l, b, mu
                (distance modulus) in Galactic coordinates.
            get_EBV
                self.query_dust for arrays

    '''
    def __init__(self, path):

        with h5py.File(path,'r') as data:
            self.pix_info= data['/pixel_info'][:]
            self.best_fit= data['/best_fit'][:]

        self.nsides = [64, 128, 256, 512, 1024, 2048]
        self.indexArray= numpy.arange(len(self.pix_info['healpix_index']))
        self.distmods= numpy.linspace(4.,19.,31)

    def query_dust(self, l, b, mu):
        '''
            Returns the dust extinction E(B-V) in the SFD scale for the selected position l, b, mu (distance modulus)
            in Galactic coordinates.

            Parameters
            ----------
                l : float
                    longitude (deg)
                b : float
                    latitude (deg)
                mu : float
                    distance modulus

            Returns
            -------
                float
                    EBV in SFD scale
        '''
        idx = None
        #print(np.max(l))
        #print(np.min(l))
        #print(np.max(b))
        #print(np.min(b))
        for nside in self.nsides:
            # Search for the pixel in this Nside level
            tpix = healpy.pixelfunc.ang2pix(nside,(90.-b)*_DEGTORAD, l*_DEGTORAD,nest=True)
            indx = (self.pix_info['healpix_index'] == tpix)*(self.pix_info['nside'] == nside)

            #Found something
            if(indx.sum()>0):
                idx = self.indexArray[indx]
                break

        interp = interpolate.InterpolatedUnivariateSpline(self.distmods, self.best_fit[idx], k=1)
        return interp(mu)


    def get_EBV(self, larray, barray, muarray):
        '''
            self.query_dust for input arrays
        '''
        return numpy.array([self.query_dust(l, b, m) for l, b, m in zip(larray, barray, muarray)])
