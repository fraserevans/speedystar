import numpy as np

def SkytoPlane(alpha,delta,D,alpha0,delta0,D0, i=0., theta=0.):
    '''
        Converts IRCS positions on the sky (coords=[ra,dec,dist]) to Cartesion coordinates for a system 
        centred on coords0=[ra0,dec0,dist0] with position angle theta and inclination i
    '''

    alpha = np.deg2rad(alpha)
    delta = np.deg2rad(delta)
    alpha0 = np.deg2rad(alpha0)
    delta0 = np.deg2rad(delta0)

    i = np.deg2rad(i)
    theta = np.deg2rad(theta)

    x = D*(-np.cos(delta)*np.sin(alpha-alpha0))

    y = D*(np.sin(delta)*np.cos(delta0) - np.cos(delta)*np.sin(delta0)*np.cos(alpha-alpha0))

    z = D0 - D*(np.cos(delta)*np.cos(delta0)*np.cos(alpha-alpha0) + np.sin(delta)*np.sin(delta0))

    xprime = x*np.cos(theta) + y*np.sin(theta)
    yprime = -x*np.sin(theta)*np.cos(i) + y*np.cos(theta)*np.cos(i) - z*np.sin(i)
    zprime = -x*np.sin(theta)*np.sin(i) + y*np.cos(theta)*np.sin(i) + z*np.cos(i)

    #result = xprime, yprime, zprime
    return xprime, yprime, zprime

