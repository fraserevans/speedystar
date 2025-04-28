import os
import sys
import numpy as np

def fetch_dusttmp(self,path='./'):

    '''
    Download the desired dust map. Please see mwdust:
    https://github.com/jobovy/mwdust
    WARNING. Default installation maps take up 5.4 GB in total

    Alternatively, download maps directly from the following URLs
    Combined19 : https://zenodo.org/record/3566060/files/combine19.h5
    Combined15 : https://zenodo.org/record/31262/files/dust-map-3d.h5

    Arguments
    --------
    path: string
        directory that will contain the dust data
    '''

    #envcommand = 'setenv DUST_DIR '+path
    #os.system(envcommand)
    os.putenv('DUST_DIR',path)
    os.system('git clone https://github.com/jobovy/mwdust.git')
    os.chdir('./mwdust')
    os.system('python setup.py install --user')


def fetch_dust(self,path, maps=['Combined15']):

    '''
    Download the desired dust maps. Please see mwdust:
    https://github.com/jobovy/mwdust
    WARNING. Default installation maps take up 5.4 GB in total

    Alternatively, download maps directly from the following URLs
    Combined19 : https://zenodo.org/record/3566060/files/combine19.h5
    Combined15 : https://zenodo.org/record/31262/files/dust-map-3d.h5

    Arguments
    --------
    path: string
        directory that will contain the dust data
    maps: string or list of strings
        Dust map to download. Options are all, Marshall06, Drimmel03, 
        Sale14, Green15, Green17, Green19, Combined15, Combined19
    '''

    import mwdust

    if not os.path.exists(path):
        raise SystemExit('Path '+path+' does not exist')

    #os.putenv('DUST_DIR',path)
    os.environ['DUST_DIR'] = path

    if type(maps)==str:
        maps = [maps]

    mapsizes = {'Marshall06': 0.005, 'Drimmel03': 0.026, 'Sale14':0.107, 
                    'Green15':4.46, 'Green17':4.27, 'Green19': 0.694, 
                'Combined15':0.558, 'Combined19': 3.7}

    mapfiles= {'Marshall06': 'marshall06/table1.dat' , 
                'Drimmel03': 'util/drimmeldata/data-for.tar.gz', 
                'Sale14': 'sale14/Amap.dat', 
                'Green15': 'green15/dust-map-3d.h5', 
                'Green17': 'green17/bayestar2017.h5', 
                'Green19': 'green19/bayestar2019.h5', 
                'Combined15': 'combined15/dust-map-3d.h5', 
                'Combined19': 'combined19/combine19.h5'}        

    if maps==['all']:
        maps = ['Marshall06', 'Drimmel03', 'Sale14', 'Green15', 
                'Green17', 'Green19', 'Combined15', 'Combined19']

    maptotsize = 0
    for map in maps:
        if map in list(mapsizes.keys()):
            if map != 'Drimmel03':
                print(os.path.join(os.environ['DUST_DIR'],mapfiles[map]))
                if  not os.path.exists( 
                        os.path.join(os.environ['DUST_DIR'],mapfiles[map])):
                    maptotsize += mapsizes[map]
            else:
                #print(os.path.join(mwdust.__path__[0],mapfiles[map]))
                if not os.path.exists(
                        os.path.join(mwdust.__path__[0],mapfiles[map])):
                    maptotsize += mapsizes[map]
        else:
            raise ValueError('map not recognized :' + map)

    if(maptotsize>0):
        proceedBool = query_yes_no('A total of '
                                + str(np.round(maptotsize,decimals=2)) +
                                ' Gb will be downloaded. Continue? [y/n]:')

        if proceedBool:        
            for map in maps:
                if map=='Marshall06':
                    from mwdust.Marshall06 import Marshall06
                    Marshall06.download()
                elif map=='Drimmel03':
                    from mwdust.Drimmel03 import Drimmel03
                    Drimmel03.download()
                elif map=='Sale14':
                    from mwdust.Sale14 import Sale14
                    Sale14.download()
                elif map=='Green15':
                    from mwdust.Green15 import Green15
                    Green15.download()
                elif map=='Green17':
                    from mwdust.Green17 import Green17
                    Green17.download()
                elif map=='Green19':
                    from mwdust.Green19 import Green19
                    Green19.download()
                elif map=='Combined15':
                    from mwdust.Combined15 import Combined15
                    Combined15.download()
                elif map=='Combined19':
                    from mwdust.Combined19 import Combined19
                    Combined19.download()

    #envcommand = 'setenv DUST_DIR '+path
    #print('DUSTSNARK')
    #os.system(envcommand)
    #os.system('git clone https://github.com/jobovy/mwdust.git')
    #os.chdir('./mwdust')
    #os.system('python setup.py install --user')

def config_dust(self,path='./'):
    '''
    Load in the dust map used for photometry calculations

    Arguments
    ----------
    path: string
        path where the desired dust map can be found            
    '''

    #os.putenv('DUST_DIR',path)
    os.environ['DUST_DIR'] = path

    #from .utils.dustmap import DustMap
    #self.dust = DustMap(path)

def config_rvssf(self,path):
    '''
    Fetch Gaia radial velocity selection functions

    Arguments
    ----------
    path: string
        path where you want the selection functions installed.
        Note -- requires ~473 Mb of space
    '''
    import os

    #import .utils.selectionfunctions.cog_v as CogV
    #from .utils.selectionfunctions import cog_v as CogV
    #from .utils.selectionfunctions.config import config

    #config['data_dir'] = path
    #CogV.fetch(subset='rvs')
    #try:
        #os.system('setenv GAIAUNLIMITED_DATADIR '+path)
    #print('SNARKKKK')
    #os.system('export GAIAUNLIMITED_DATADIR='+path)

    #envcommand = 'setenv GAIAUNLIMITED_DATADIR '+path
    #os.system(envcommand)
    os.putenv('GAIAUNLIMITED_DATADIR',path)


def config_astrosf(self,path):
    '''
    Fetch Gaia astrometric spread functions

    Arguments
    ----------
    path: string
        path where you want the selection functions installed.
        Note -- requires ~435 Mb of space
    '''

    #try:
    from scanninglaw.config import config
    import scanninglaw.asf
    #except ImportError:
    #raise ImportError(__ImportError__)       

    config['data_dir'] = path
    scanninglaw.asf.fetch()
    scanninglaw.asf.fetch(version='dr3_nominal')

def set_ast_sf(self,sfbool):
    '''
    Set whether or not to use the Gaia astrometric spread function

    Arguments
    ----------
    sfbool : Boolean
        Whether or not to use the astrometric spread function
    '''
    self.use_ast_sf = sfbool

def set_rv_sf(self,sfbool):
    '''
    Set whether or not to use the Gaia radial velocity selection function

    Arguments
    ----------
    sfbool : Boolean
        Whether or not to use the radial velocity selection function
    '''
    self.use_rv_sf = sfbool

def set_Gaia_release(self,Gaia_release):

    '''
    Set which Gaia release is assumed when errors are calculated

    Arguments
    ---------
    Gaia_release: string
        Gaia data release. Options are DR2, EDR3, DR3, DR4, DR5
    '''

    _Gaia_releases = ['DR2', 'EDR3', 'DR3', 'DR4', 'DR5']
    Gaia_release = Gaia_release.upper() 
    #Check to make sure supplied data release is valid option
    if Gaia_release not in _Gaia_releases:
        raise ValueError('Error: invalid Gaia data release. Options are DR2, EDR3, DR3, DR4, DR5. See speedystar.config.set_Gaia_release() docstring')
 
    self.gaia_release = Gaia_release

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}

    while True:
        sys.stdout.write(question)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n')")
