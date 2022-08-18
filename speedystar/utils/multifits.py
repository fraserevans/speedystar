from astropy.table import Table, vstack

def split_fits(mycat, mydir, n):
    '''
        Splits a fits table located in mycat into n fits tables located in the directory mydir

        Parameters
        ----------
        mydir : str
            path of the directory
        mycat : str
            path of the catalog
        n : int
            number of subcatalogs
    '''
    import os
    import numpy as np

    data = Table.read(mycat)

    if(not os.path.isdir(mydir)):
        os.mkdir(mydir)

    i=0
    for idxs_single in np.array_split(np.arange(len(data)), n):
        data[idxs_single].write(mydir+'/'+str(i)+'.fits', overwrite=True)
        i+=1


def concatenate_fits(mycat, mydir):
    '''
        Concatenates all fits tables ending in .fits located in mydir in a
        single catalog located in mycat.fits

        Parameters
        ----------
        mydir : str
            path of the directory
        mycat : str
            path of the catalog

    '''
    import glob

    a = glob.glob(mydir+'/*.fits')

    data = Table.read(a[0])

    for i in xrange(len(a)-1):
        data = vstack([data, Table.read(a[i+1])])

    data.write(mycat, overwrite=True)
