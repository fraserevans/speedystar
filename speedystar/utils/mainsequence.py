import numpy
from astropy import units as u

def t_MS(m,xi=0):
    '''
        Main sequence lifetime for a star of mass M and metallicity Z. Fit provided by Hurley+ 2000 
        [https://doi.org/10.1046/j.1365-8711.2000.03426.x]

        Parameters
        ----------
            M : 1D array (float)
                Mass in solar masses
            xi : 1D array (float)
                xi = log10(Z/0.02) 
        
        Returns
        -------
            Main sequence lifetime in Myr
    '''

    m = m.to('Msun').value

    return numpy.array([max( Mu_param(M,xi)*t_BGB(M,xi), x_param(xi)*t_BGB(M,xi) ) for M in m])* u.Myr


def a_coeff(xi,n):
    #Auxiliary function for t_MS()
    if n == 1:

            alpha = 1.593890e3
            beta = 2.053038e3
            gamma = 1.231226e3
            eta = 2.327785e2
            mu = 0.

    if n == 2:

            alpha = 2.706708e3
            beta = 1.483131e3
            gamma = 5.772723e2
            eta = 7.411230e1
            mu = 0.

    if n == 3:

            alpha = 1.466143e2
            beta = -1.048442e2
            gamma = -6.795374e1
            eta = -1.391127e1
            mu = 0.

    if n == 4:

            alpha = 4.141960e-2
            beta = 4.564888e-2
            gamma = 2.958542e-2
            eta = 5.571483e-3
            mu = 0.

    if n == 5:

            alpha = 3.426349e-1
            beta = 0.
            gamma = 0.
            eta = 0.
            mu = 0.

    if n == 6:

            alpha = 1.949814e1
            beta = 1.758178e0
            gamma = -6.008212e0
            eta = -4.470533e0
            mu = 0.

    if n == 7:

            alpha = 4.903830e0
            beta = 0.
            gamma = 0.
            eta = 0.
            mu = 0.

    if n == 8:

            alpha = 5.212154e-2
            beta = 3.166411e-2
            gamma = -2.750074e-3
            eta = -2.271549e-3
            mu = 0.

    if n == 9:

            alpha = 1.312179e0
            beta = -3.294936e-1
            gamma = 9.231860e-2
            eta = 2.610989e-2
            mu = 0.

    if n == 10:

            alpha = 8.073972e-1
            beta = 0.
            gamma = 0.
            eta = 0.
            mu = 0.

    return alpha + beta*xi + gamma*xi**2. + eta*xi**3. + mu*xi**4.

def x_param(xi):
    #Auxiliary function for t_MS()
    return max( 0.95, min(0.95 - 0.03*(xi + 0.30103), 0.99) )

def Mu_param(M,xi):
    #Auxiliary function for t_MS()
    return max( 0.5, 1.0 - 0.01*max( a_coeff(xi,6)/M**a_coeff(xi,7), a_coeff(xi,8) + a_coeff(xi,9)/M**a_coeff(xi,10) ) )

def t_BGB(M,xi):
    #Auxiliary function for t_MS()
    return (a_coeff(xi,1) + a_coeff(xi,2)*M**4. + a_coeff(xi,3)*M**5.5 + M**7.) / (a_coeff(xi,4)*M**2. + a_coeff(xi,5)*M**7.)

