
__ImportError__ = "One or more required external packages have not been installed. Please see requirements.txt"

try:
    import numpy as np
    import astropy
    import astropy.units as u
    import astropy.constants as const
    from tqdm import tqdm
except ImportError:
    raise ImportError(__ImportError__)

# This functions implement analytic formulae for stellar evolution presented in Hurley, Pols, and Tout 2000 (MNRAS 315, 543-569)


def get_Z(xi):
        return 0.02 * 10.**xi


# ================ Coefficients from Appendix A ================================== #


def a_coeff(xi,n):

        # ---------- tMS + LTMS: ---------------

        if n == 1:
                params = [1.593890e3, 2.053038e3, 1.231226e3, 2.327785e2, 0.]

        if n == 2:
                params = [2.706708e3, 1.483131e3, 5.772723e2, 7.411230e1, 0.]

        if n == 3:
                params = [1.466143e2, -1.048442e2, -6.795374e1, -1.391127e1, 0.]

        if n == 4:
                params = [4.141960e-2, 4.564888e-2, 2.958542e-2, 5.571483e-3, 0.]

        if n == 5:
                params = [3.426349e-1, 0., 0., 0., 0.]

        if n == 6:
                params = [1.949814e1, 1.758178e0, -6.008212e0, -4.470533e0, 0.]

        if n == 7:
                params = [4.903830e0, 0., 0., 0., 0.]

        if n == 8:
                params = [5.212154e-2, 3.166411e-2, -2.750074e-3, -2.271549e-3, 0.]

        if n == 9:
                params = [1.312179e0, -3.294936e-1, 9.231860e-2, 2.610989e-2, 0.]

        if n == 10:
                params = [8.073972e-1, 0., 0., 0., 0.]

        if n == 11:
                params = [1.031538e0, -2.434480e-1, 7.732821e0, 6.460705e0, 1.374484e0]

        if n == 12:
                params = [1.043715e0, -1.577474e0, -5.168234e0, -5.596506e0, -1.299394e0]

        if n == 13:
                params = [7.859573e2, -8.542048e0, -2.642511e1, -9.585707e0, 0.]

        if n == 14:
                params = [3.858911e3, 2.459681e3, -7.630093e1, -3.486057e2, -4.861703e1]

        if n == 15:
                params = [2.888720e2, 2.952979e2, 1.850341e2, 3.797254e1, 0.]

        if n == 16:
                params = [7.196580e0, 5.613746e-1, 3.805871e-1, 8.398728e-2, 0.]

        # --------- RTMS ----------------

        if n == 17:
                Z = get_Z(xi)
                sigma = np.log10(Z)
                loga = max(0.097 - 0.1072*(sigma+3), max(0.097, min(0.1461, 0.1461 + 0.1237*(sigma+2))))
                return 10.**loga

        if n == 18:
                params = [2.187715e-1, -2.154437e0, -3.768678e0, -1.975518e0, -3.021475e-1]

        if n == 19:
                params = [1.466440e0, 1.839725e0, 6.442199e0, 4.023635e0, 6.957529e-1]

        if n == 20:
                params = [2.652091e1, 8.178458e1, 1.156058e2, 7.633811e1, 1.950698e1]

        if n == 21:
                params = [1.472103e0, -2.947609e0, -3.312828e0, -9.945065e-1, 0.]

        if n == 22:
                params = [3.071048e0, -5.679941e0, -9.745523e0, -3.594543e0, 0.]

        if n == 23:
                params = [2.617890e0, 1.019135e0, -3.292551e-2, -7.445123e-2, 0.]

        if n == 24:
                params = [1.075567e-2, 1.773287e-2, 9.610479e-3, 1.732469e-3, 0.]

        if n == 25:
                params = [1.476246e0, 1.899331e0, 1.195010e0, 3.035051e-1, 0.]

        if n == 26:
                params = [5.502535e0, -6.601663e-2, 9.968707e-2, 3.599801e-2, 0.]

        # ----------- LBGB --------------------

        if n == 27:
                params = [9.511033e1, 6.819618e1, -1.045625e1, -1.474939e1, 0.]

        if n == 28:
                params = [3.113458e1, 1.012033e1, -4.650511e0, -2.463185e0, 0.]

        if n == 29:
                params = [1.413057e0, 4.578814e-1, -6.850581e-2, -5.588658e-2, 0.]

        if n == 30:
                params = [3.910862e1, 5.196646e1, 2.264970e1, 2.873680e0, 0.]

        if n == 31:
                params = [4.597479e0, -2.855179e-1, 2.709724e-1, 0., 0.]

        if n == 32:
                params = [6.682518e0, 2.827718e-1, -7.294429e-2, 0., 0.]

        # --------- Delta L ----------------

        if n == 33:
                a = min(1.4, 1.5135 + 0.3769*xi)
                return max(0.6355 - 0.4192*xi, max(1.25, a))

        if n == 34:
                params = [1.910302e-1, 1.158624e-1, 3.348990e-2, 2.599706e-3, 0.]

        if n == 35:
                params = [3.931056e-1, 7.277637e-2, -1.366593e-1, -4.508946e-2, 0.]

        if n == 36:
                params = [3.267776e-1, 1.204424e-1, 9.988332e-2, 2.455361e-2, 0.]

        if n == 37:
                params = [5.990212e-1, 5.570264e-2, 6.207626e-2, 1.777283e-2, 0.]

        # ----------- Delta R --------------

        if n == 38:
                params = [7.330122e-1, 5.192827e-1, 2.316416e-1, 8.346941e-3, 0.]

        if n == 39:
                params = [1.172768e0, -1.209262e-1, -1.193023e-1, -2.859837e-2, 0.]

        if n == 40:
                params = [3.982622e-1, -2.296279e-1, -2.262539e-1, -5.219837e-2, 0.]

        if n == 41:
                params = [3.571038e0, -2.223625e-2, -2.611794e-2, -6.359648e-3, 0.]

        if n == 42:
                params = [1.9848e0, 1.1386e0, 3.5640e-1, 0., 0.]

        if n == 43:
                params = [6.300e-2, 4.810e-2, 9.840e-3, 0., 0.]

        if n == 44:
                params = [1.200e0, 2.450e0, 0., 0., 0.]

        # ----------- alpha L --------------

        if n == 45:
                params = [2.321400e-1, 1.828075e-3, -2.232007e-2, -3.378734e-3, 0.]

        if n == 46:
                params = [1.163659e-2, 3.427682e-3, 1.421393e-3, -3.710666e-3, 0.]

        if n == 47:
                params = [1.048020e-2, -1.231921e-2, -1.686860e-2, -4.234354e-3, 0.]

        if n == 48:
                params = [1.555590e0, -3.223927e-1, -5.197429e-1, -1.066441e-1, 0.]

        if n == 49:
                params = [9.7700e-2, -2.3100e-1, -7.5300e-2, 0., 0.]

        if n == 50:
                params = [2.4000e-1, 1.8000e-1, 5.9500e-1, 0., 0.]

        if n == 51:
                params = [3.3000e-1, 1.3200e-1, 2.1800e-1, 0., 0.]

        if n == 52:
                params = [1.1064e0, 4.1500e-1, 1.8000e-1, 0., 0.]

        if n == 53:
                params = [1.1900e0, 3.7700e-1, 1.7600e-1, 0., 0.]

        # ----------- beta L --------------

        if n == 54:
                params = [3.855707e-1, -6.104166e-1, 5.676742e0, 1.060894e1, 5.284014e0]

        if n == 55:
                params = [3.579064e-1, -6.442936e-1, 5.494644e0, 1.054952e1, 5.280991e0]

        if n == 56:
                params = [9.587587e-1, 8.777464e-1, 2.017321e-1, 0., 0.]

        if n == 57:
                a = min(1.4, 1.5135 + 0.3769*xi)
                return max(0.6355 - 0.4192*xi, max(1.25, a))

        # ---------- alpha R ---------------

        if n == 58:
                params = [4.907546e-1, -1.683928e-1, -3.108742e-1, -7.202918e-2, 0.]

        if n == 59:
                params = [4.537070e0, -4.465455e0, -1.612690e0, -1.623246e0, 0.]

        if n == 60:
                params = [1.796220e0, 2.814020e-1, 1.423325e0, 3.421036e-1, 0.]

        if n == 61:
                params = [2.256216e0, 3.773400e-1, 1.537867e0, 4.396373e-1, 0.]

        if n == 62:
                params = [8.4300e-2, -4.7500e-2, -3.5200e-2, 0., 0.]

        if n == 63:
                params = [7.3600e-2, 7.4900e-2, 4.4260e-2, 0., 0.]

        if n == 64:
                params = [1.3600e-1, 3.5200e-2, 0., 0., 0.]

        if n == 65:
                params = [1.564231e-3, 1.653042e-3, -4.439786e-3, -4.951011e-3, -1.216530e-3]

        if n == 66:
                params = [1.4770e0, 2.9600e-1, 0., 0., 0.]

        if n == 67:
                params = [5.210157e0, -4.143695e0, -2.120870e0, 0., 0.]

        if n == 68:
                params = [1.1160e0, 1.6600e-1, 0., 0., 0.]

        # ---------- beta R ---------------

        if n == 69:
                params = [1.071489e0, -1.164852e-1, -8.623831e-2, -1.582349e-2, 0.]

        if n == 70:
                params = [7.108492e-1, 7.935927e-1, 3.926983e-1, 3.622146e-2, 0.]

        if n == 71:
                params = [3.478514e0, -2.585474e-2, -1.512955e-2, -2.833691e-3, 0.]

        if n == 72:
                params = [9.132108e-1, -1.653695e-1, 0., 3.636784e-2, 0.]

        if n == 73:
                params = [3.969331e-3, 4.539076e-3, 1.720906e-3, 1.897857e-4, 0.]

        if n == 74:
                params = [1.600e0, 7.640e-1, 3.322e-1, 0., 0.]

        # ----------- gamma --------------

        if n == 75:
                params = [8.109e-1, -6.282e-1, 0., 0., 0.]

        if n == 76:
                params = [1.192334e-2, 1.083057e-2, 1.230969e0, 1.551656e0, 0.]

        if n == 77:
                params = [-1.668868e-1, 5.818123e-1, -1.105027e1, -1.668070e1, 0.]

        if n == 78:
                params = [7.615495e-1, 1.068243e-1, -2.011333e-1, -9.371415e-2, 0.]

        if n == 79:
                params = [9.409838e0, 1.522928e0, 0., 0., 0.]

        if n == 80:
                params = [-2.7110e-1, -5.7560e-1, -8.3800e-2, 0., 0.]

        if n == 81:
                params = [2.4930e0, 1.1475e0, 0., 0., 0.]

        # -------------------------

        alpha, beta, gamma, eta, mu = params

        a = alpha + beta*xi + gamma*xi**2. + eta*xi**3. + mu*xi**4.

        if n == 11:
                return a * a_coeff(xi, 14)
        elif n == 12:
                return a * a_coeff(xi, 14)

        elif n == 18:
                return a * a_coeff(xi,20)
        elif n == 19:
                return a * a_coeff(xi,20)

        elif n == 29:
                return a**a_coeff(xi,32)

        elif n == 42:
                return min(1.25, max(1.1, a))
        elif n == 44:
                return min(1.3, max(0.45, a))

        elif n == 49:
                return max(a, 0.145)
        elif n == 50:
                return min(a, 0.306 + 0.053*xi)
        elif n == 51:
                return min(a, 0.3625 + 0.062*xi)
        elif n == 52:
                Z = get_Z(xi)
                a = max(a, 0.9)
                if Z > 0.01:
                        return min(a, 1.0)
                else:
                        return a
        elif n == 53:
                Z = get_Z(xi)
                a = max(a, 1.0)
                if Z > 0.01:
                        return min(a, 1.1)
                else:
                        return a

        elif n == 62:
                return max(0.065, a)
        elif n == 63:
                Z = get_Z(xi)
                if Z < 0.004:
                        return min(0.055, a)
                else:
                        return a
        elif n == 64:
                a = max( 0.091, min(0.121, a) )
                if a_coeff(xi, 68) > (a_coeff(xi, 66)):
                        a = alpha_R(a_coeff(xi, 66), xi)
                return a
        elif n == 66:
                a = max(a, min(1.6, -0.308 - 1.046*xi))
                return max(0.8, min(0.8 - 2.0*xi, a))
        elif n == 68:
                return max(0.9, min(a, 1.0))

        elif n == 72:
                Z = get_Z(xi)
                if Z > 0.01:
                        return max(a, 0.95)
                else:
                        return a
        elif n == 74:
                return max(1.4, min(a, 1.6))

        elif n == 75:
                a = max(1.0, min(a, 1.27))
                return max(a, 0.6355 - 0.4192*xi)
        elif n == 76:
                return max(a, -0.1015564 - 0.2161264*xi - 0.05182516*xi**2.)
        elif n == 77:
                return max(-0.3868776 - 0.5457078*xi -0.1463472*xi**2., min(0.0, a))
        elif n == 78:
                return max(0.0, min(a, 7.454 + 9.046*xi))
        elif n == 79:
                return min(a, max(2.0, -13.3 - 18.6*xi))
        elif n == 80:
                return max(0.0585542, a)
        elif n == 81:
                return min(1.5, max(0.4, a))

        else:
                return a

def b_coeff(xi,n):
        Z = get_Z(xi)
        sigma = np.log10(Z)
        rho = xi + 1.

        params = [0., 0., 0., 0., 0.]

        # ---------- GB Radius: ---------------

        if n == 1:
                params = [0.397, 0.28826, 0.5293, 0., 0.]

        if n == 4:
                params = [0.9960283, 0.8164393, 2.383830, 2.223436, 0.8638115]

        if n == 5:
                params = [2.561062e-1, 7.072646e-2, -5.444596e-2, -5.798167e-2, -1.349129e-2]

        if n == 6:
                params = [1.157338, 1.467883, 4.299661, 3.31305, 6.99208e-1]

        if n == 7:
                params = [4.022765e-1, 3.050010e-1, 9.962137e-1, 7.914079e-1, 1.728098e-1]

        # ---------- Luminosity at He ignition ---------

        if n == 9:
                params = [2.751631e+3, 3.557098e2, 0., 0., 0.]

        if n == 10:
                params = [-3.820831e-2, 5.872664e-2, 0., 0., 0.]

        if n == 11:
                params = [1.071738e+2, -8.970339e+1, -3.949739e+1, 0., 0.]

        if n == 12:
                params = [7.348793e+2, -1.531020e+2, -3.793700e+1, 0., 0.]

        if n == 13:
                params = [9.219293, -2.005865, -5.561309e-1, 0., 0.]
        
        # ------------ Core Helium Burning ------------ #

        if n == 14:
                params = [2.917412, 1.575290, 5.751814e-1, 0., 0.]

        if n == 15:
                params = [3.629118, -9.112722e-1, 1.042291, 0, 0]

        if n == 16:
                params = [4.916389, 2.862149, 7.844850e-1, 0., 0.]

        # --------- RTMS ----------------

        if n == 18:
                params = [5.496045e+1, -1.289968e+1, 6.385758, 0., 0.]

        if n == 19:
                params = [1.832694, -5.766608e-2, 5.696128e-2, 0., 0.]

        if n == 20:
                params = [1.211104e+2, 0., 0., 0., 0.]

        if n == 21:
                params = [2.214088e+2, 2.187113e+2, 1.170177e+1, -2.635340e+1, 0.]

        if n == 22:
                params = [2.063983e+0, 7.363827e-1, 2.654323e-1, -6.140719e-2, 0.]

        if n == 23:
                params = [2.003160, 9.388871e-1, 9.656450e-1, 2.362266e-1, 0.]

        if n == 24:
                params = [1.609901e+1, 7.391573, 2.277010e+1, 8.334227, 0.]

        if n == 25:
                params = [1.747500e-1, 6.271202e-2, -2.324229e-2, -1.844559e-2, 0.]

        if n == 27:
                params = [2.752869, 2.729201e-2, 4.996927e-1, 2.496551e-1, 0.]

        if n == 28:
                params = [3.518506, 1.112440, -4.556216e-1, -2.179426e-1, 0.]

        if n == 29:
                params = [1.626062e+2, -1.168838e+1, -5.498343, 0., 0.]

        if n == 30:
                params = [3.336833e-1, -1.458043e-1, -2.011751e-2, 0., 0.]

        if n == 31:
                params = [7.425137e+1, 1.790236e+1, 3.033910e+1, 1.018259e+1, 0.]

        if n == 32:
                params = [9.268325e+2, -9.739859e+1, -7.702152e+1, -3.158268e+1, 0.]

        if n == 33:
                params = [2.474401, 3.892972e-1, 0., 0., 0.]

        if n == 34:
                params = [1.127018e+1, 1.622158, -1.443664, -9.474699e-1, 0.]

        if n == 36:
                params = [1.445216e-1, -6.180219e-2, 3.093878e-2, 1.567090e-2, 0.]

        if n == 37:
                params = [1.304129, 1.395919e-1, 4.142455e-3, -9.732503e-3, 0.]

        if n == 38:
                params = [5.114149e-1, -1.160850e-2, 0., 0., 0.]

        if n == 39:
                params = [1.314955e+2, 2.009258e+1, -5.143082e-1, -1.379140, 0.]

        if n == 40:
                params = [1.823973e+1, -3.074559, -4.307878, 0., 0.]

        if n == 41:
                params = [2.327037, 2.403445, 1.208407, 2.087263e-1, 0.]

        if n == 42:
                params = [1.997378, -8.126205e-1, 0., 0., 0.]

        if n == 43:
                params = [1.079113e-1, 1.762409e-2, 1.096601e-2, 3.058818e-3, 0.]

        if n == 44:
                params = [2.327409, 6.901582e-1, -2.158431e-1, -1.084117e-1, 0.]

        if n == 46:
                params = [2.214315, -1.975747, 0., 0., 0.]

        if n == 48:
                params = [5.072525, 1.146189e+1, 6.961724, 1.316965, 0.]

        if n == 49:
                params = [5.139740, 0., 0., 0., 0.]

        if n == 51:
                params = [1.125124, 1.306486, 3.622359, 2.601976, 3.031270e-1]
        if n == 52:
                params = [3.349489e-1, 4.531269e-3, 1.131793e-1, 2.300156e-1, 7.632745e-2]
        if n == 53:
                params = [1.467794, 2.798142, 9.455580, 8.963904, 3.339719]
        # ----------- beta L --------------

        if n == 54:
                params = [4.658512e-1, 2.597451e-1, 9.048179e-1, 7.394505e-1, 1.607092e-1]

        if n == 55:
                params = [1.0422, 1.3156e-1, 4.5000e-2, 0., 0.]

        if n == 56:
                params = [1.110866, 9.623856e-1, 2.735487, 2.445602, 8.826352e-1]

        if n == 57:
                params = [-1.584333e-1, 1.728865e-1, -4.461431e-1, -3.925259e-1, -1.276203e-1]

        alpha, beta, gamma, eta, mu = params

        b = alpha + beta*xi + gamma*xi**2. + eta*xi**3. + mu*xi**4.

        if n == 1:
                return min(0.54, b)

        elif n == 2:
                b = 10**(-4.6739 - 0.9394*sigma)
                return min( max( b, -0.04167 + 55.67*Z), 0.4771 - 9329.21*(Z**2.94) )

        elif n == 3:
                b = max( -0.1451, -2.2794 - 1.5175*sigma - 0.254*(sigma**2) )
                b = 10**b
                if(Z>=0.004):
                    return max( b, 0.7307 + 14265.1*(Z**3.395) )
                else:
                    return b

        elif n == 4:
                return b + 0.1231572*(xi**5)

        elif n == 6:
                return b + 0.01640687*(xi**5)

        elif n == 11:
                return b**2

        elif n == 13:
                return b**2

        elif n == 14:
                return b ** b_coeff(xi,15)

        elif n == 16:
                return b ** b_coeff(xi,15)

        elif n == 17:
                if xi > -1:
                    return 1. - 0.3880523*(xi+1.)**2.862149
                else:
                    return 1.

        elif n == 24:
                return b**b_coeff(xi,28)

        elif n == 26:
                return 5.0 - 0.09138012*(Z**-0.3671407)

        elif n == 27:
                return b**(2*b_coeff(xi,28))

        elif n == 31:
                return b**b_coeff(xi,33)

        elif n == 34:
                return b**b_coeff(xi,33)

        elif n == 36:
                return b**4

        elif n == 37:
                return 4.*b

        elif n == 38:
                return b**4

        elif n == 40:
                return max(b,1.)

        elif n == 41:
                return b**b_coeff(xi,42)

        elif n == 44:
                return b**5

        elif n == 45:
                if rho<=0.0:
                    return 1.0
                else:
                    return 1.0 - (2.47162*rho - 5.401682*rho**2 + 3.247361*rho**3)

        elif n == 46:
                return -b*np.log10(M_HeF(xi)/M_FGB(Z))

        elif n == 47:
                return 1.127733*rho + 0.2344416*rho**2 - 0.3793726*rho**3

        elif n == 50:
                return b_coeff(xi,55)*b_coeff(xi,3)

        elif n == 51:
                return b - 0.1343798*xi**5

        elif n == 53:
                return b + 0.4426929*xi**5

        elif n == 55:
                return min(0.99164-743.123*(Z**2.83),b)

        elif n == 56:
                return b + 0.1140142*(xi**5)

        elif n == 57:
                return b - 0.01308728*(xi**5)
        else:
                #print(params)
                #print(n)
                if params == [0, 0, 0, 0, 0]:
                    print('ERROR, parameter has not yet been coded in')
                    print(n)
                    exit
                else:
                    return b

# ====================== Main Sequence Lifetime ===================== #


def x_param(xi):
        # Eq. 6:
        return max( 0.95, min(0.95 - 0.03*(xi + 0.30103), 0.99) )

def Mu_param(M,xi):
        # Eq. 7:
        return max( 0.5, 1.0 - 0.01*max( a_coeff(xi,6)/(M**a_coeff(xi,7)), a_coeff(xi,8) + a_coeff(xi,9)/(M**a_coeff(xi,10)) ) )

def t_BGB(M,xi):
        # Base of the Giant Branch (BGB) lifetime, Eq. 4:
        return (a_coeff(xi,1) + a_coeff(xi,2)*M**4. + a_coeff(xi,3)*M**5.5 + M**7.) / (a_coeff(xi,4)*M**2. + a_coeff(xi,5)*M**7.)

def t_BAGB(M,xi):
        # Base of the AGB lifetime
        M_c = M_cHeI(M,xi)
        return t_HeI(M,xi)+t_He(M,xi,M_c)

def t_hook(M, xi):
        return Mu_param(M, xi) * t_BGB(M, xi)

def t_MS(M,xi):
        # Main Sequence (MS) lifetime, Eq. 5:
        return max( t_hook(M, xi), x_param(xi)*t_BGB(M,xi) )

def t_HeI(M,xi):
        # Time to He ignition, Eq. 43
        p, q, B, D, A_H, M_x, L_x, t_inf1, t_inf2, t_x = GB_params(M,xi)    

        if L_HeI(M,xi)<=L_x:
            return t_inf1 - ( 1 / ( (p-1)*A_H*D ) ) * ( (D/L_HeI(M,xi))**( (p-1)/p) )
        else:
            return t_inf2 - ( 1 / ( (q-1)*A_H*B ) ) * ( (B/L_HeI(M,xi))**( (q-1)/q) )

def M_cHeI(M,xi):
        #Core mass at helium ignition, see text below eq. 65
        p, q, B, D, A_H, M_x, L_x, t_inf1, t_inf2, t_x = GB_params(M,xi) 
        L = L_HeI(M,xi)
        
        if M<M_HeF(xi):
            #Eq. 31
            alpha_1 = (b_coeff(xi,9)*M_HeF(xi)**b_coeff(xi,10) - L_HeF(xi)) / L_HeF(xi)
            #print([L,D,1+alpha_1*np.exp(15*(M-M_HeF(xi))),p])
            return (L/D)**(1/p)
        elif M>=M_HeF(xi):
            #print('snark')
            L = L_HeI(M_HeF(xi),xi)
            #Eq. 31
            Mc = (L/D)**(1/p)
            #Eq. 44
            C = Mc**4 - 9.20925e-5*M_HeF(xi)**5.402216
            #print([L,D,p])
            #Eq. 44
            #if 0.95*M_cBAGB(M,xi) < (C + 9.20925e-5*M**5.402216)**0.25:
            #    print([M,'1'])
            #else:
            #    print([M,'2'])
            return min(0.95*M_cBAGB(M,xi), (C + 9.20925e-5*M**5.402216)**0.25 )
            #return (C + 9.20925e-5*M**5.402216)**0.25 

def L_HeI(M,xi):
        # Luminosity at He ignition, Eq. 49
        if M<M_HeF(xi):
            alpha_1 = (b_coeff(xi,9)*M_HeF(xi)**b_coeff(xi,10) - L_HeF(xi)) / L_HeF(xi)
            return ( b_coeff(xi,9) * M**b_coeff(xi,10) ) / (1 + alpha_1*np.exp(15*(M - M_HeF(xi))) )
        else:
            return ( b_coeff(xi,11) + b_coeff(xi,12)*(M**3.8) ) / (b_coeff(xi,13) + M**2)

def R_HeI(M,xi,M_c):
        # Radius at He ignition, eq. 50 and above
        Z = get_Z(xi)

        if M<M_FGB(Z):
            return R_GB(M,xi,L_HeI(M,xi))
        elif M>=max(M_FGB(Z),12):
            return R_mHe(M,xi,M_c)
        else:
            mu = (np.log10(M/12))/(np.log10(M_FGB(Z)/12))
            return R_mHe(M,xi,M_c)*( R_GB(M,xi,L_HeI(M,xi))/R_mHe(M,xi,M_c))**mu

def M_hook(xi):
        # Initial mass above which a hook appears in the MS, Eq. 1:
        return 1.0185 + 0.16015 * xi + 0.0892 * xi**2.

def M_HeF(xi):
        # Maximum initial mass for which He ignites degenerately in a helium flash, Eq. 2:
        return 1.995 + 0.25*xi + 0.087*xi**2.

def L_HeF(xi):
        # Luminosity at He ignition for a star of mass M_HeF, below eq. 49
        return ( b_coeff(xi,11) + b_coeff(xi,12)*(M_HeF(xi)**3.8) ) / (b_coeff(xi,13) + M_HeF(xi)**2)

def M_FGB(Z):
        # Maximum initial mass for which He ignites on the first giant branch, Eq. 3:
        return (13.048 * (Z/0.02)**0.06) / (1. + 0.0012 * (0.02/Z)**1.27 )

def R_GB(M, xi, L):
        #Giant branch radius when luminosity is known, Eq. 46

        A = min (b_coeff(xi,4) * M**(-b_coeff(xi,5)) , b_coeff(xi,6)*M**(-b_coeff(xi,7)) )
        return A * ( L**b_coeff(xi,1) + b_coeff(xi,2)*L**b_coeff(xi,3) )

def L_minHe(M,xi):
        #Minimum luminosity during HeB for IM stars Eq. 51
        Z = get_Z(xi)
        c = ( b_coeff(xi,17) / (M_FGB(Z)**0.1) ) + ( b_coeff(xi,16)*b_coeff(xi,17) - b_coeff(xi,14) ) / (M_FGB(Z)**(b_coeff(xi,15)+0.1))

        return L_HeI(M,xi) * ( b_coeff(xi,14) + c*M**(b_coeff(xi,15)+0.1) ) / (b_coeff(xi, 16) + M**(b_coeff(xi,15)) )

def mu_HeB(M,xi,M_c):
        #continuity parameter for ZAHB, Eq. 52
        #print([M,M_c,M_HeF(xi)])
        return ( M - M_c ) / ( M_HeF(xi) - M_c )

def L_ZAHB(M,xi,M_c):
        #Zero age horizontal branch luminosity, Eq. 53
        mu = mu_HeB(M,xi,M_c)
        alpha_2 = (b_coeff(xi, 18) + L_ZHe(M_c, xi) - L_minHe(M_HeF(xi),xi)) / ( L_minHe(M_HeF(xi),xi) - L_ZHe(M_c, xi) )
        #print([M_c,L_ZHe(M_c, xi) + ( (1+b_coeff(xi,20)) / (1 + b_coeff(xi,20)*mu**1.6479 ) ) * ( ( b_coeff(xi,18)*mu**b_coeff(xi,19) ) / (1 + alpha_2*np.exp(15*(M - M_HeF(xi))) ) )])
        return L_ZHe(M_c, xi) + ( (1+b_coeff(xi,20)) / (1 + b_coeff(xi,20)*mu**1.6479 ) ) * ( ( b_coeff(xi,18)*mu**b_coeff(xi,19) ) / (1 + alpha_2*np.exp(15*(M - M_HeF(xi))) ) )

def R_ZAHB(M,xi,M_c):
        #Zero age horizontal branch radius, Eq. 54

        mu = mu_HeB(M,xi,M_c)

        f = ( (1. + b_coeff(xi,21))*mu**b_coeff(xi,22) ) / (1. + b_coeff(xi,21)*mu**b_coeff(xi,23))

        return (1 - f)*R_ZHe(M_c,xi) + f*R_GB(M, xi,L_ZAHB(M,xi,M_c))

def f_bl(M,xi,M_c):
        # Helper parameter for tau_bl, below Eq. 58
        #print('f_bl')
        #print(M)    

        #print([M,b_coeff(xi,48),R_mHe(M,xi,M_c),R_AGB(M,xi,L_HeI(M,xi)),b_coeff(xi,49)])
        #print( (M**b_coeff(xi,48)) * ( (1 - ( R_mHe(M,xi,M_c) / R_AGB(M,xi,L_HeI(M,xi) ) ) )**b_coeff(xi,49) ) )
        #print([b_coeff(xi,24),b_coeff(xi,25),b_coeff(xi,26),b_coeff(xi,27),b_coeff(xi,28)])
        #print([(M**b_coeff(xi,48)), ( (1 - ( R_mHe(M,xi,M_c) / R_AGB(M,xi,L_HeI(M,xi) ) ) )**b_coeff(xi,49) )])
        if(np.isnan((M**b_coeff(xi,48)) * ( (1 - ( R_mHe(M,xi,M_c) / R_AGB(M,xi,L_HeI(M,xi) ) ) )**b_coeff(xi,49) ))):
            pass
            #print([M,M**b_coeff(xi,48), R_mHe(M,xi,M_c),R_AGB(M,xi,L_HeI(M,xi)),b_coeff(xi,49),( (1 - ( R_mHe(M,xi,M_c) / R_AGB(M,xi,L_HeI(M,xi) ) ) )**b_coeff(xi,49) )])
            #print('f_bl error')
            #exit()
        return (M**b_coeff(xi,48)) * ( (1 - ( R_mHe(M,xi,M_c) / R_AGB(M,xi,L_HeI(M,xi) ) ) )**b_coeff(xi,49) )
    
def tau_bl(M, xi, M_c):
        # Duration of the 'blue' phase of HeB relative to the total CHeB time, Eq. 58
        
        Z = get_Z(xi)

        if M<M_HeF(xi):

            return 1

        elif M>=M_HeF(xi) and M<M_FGB(Z):
            alpha_bl = (1 - b_coeff(xi,45)*(M_HeF(xi)/M_FGB(Z))**0.414)*(-1)*np.abs(np.log10(M_HeF(xi)/M_FGB(Z)))**(-b_coeff(xi,46)) 
            #print([(1 - b_coeff(xi,45)*(M_HeF(xi)/M_FGB(Z))**0.414),(np.log10(M_HeF(xi)/M_FGB(Z)))**(-b_coeff(xi,46))])
            #print([(1 - b_coeff(xi,45)*(M_HeF(xi)/M_FGB(Z))**0.414),np.log10(M_HeF(xi)/M_FGB(Z)),(-b_coeff(xi,46))])
            #print(b_coeff(xi,45)*(M/M_FGB(Z))**0.414 + alpha_bl*(np.log10(M/M_FGB(Z))**b_coeff(xi,46)))
            #print([M,M_FGB(Z)])
            #print('xxxxx')
            #print([b_coeff(xi,45)*(M/M_FGB(Z))**0.414,alpha_bl,np.log10(M/M_FGB(Z))+0.0001,b_coeff(xi,46)])
            #print([b_coeff(xi,45)*(M/M_FGB(Z))**0.414,alpha_bl,(np.log10(M/M_FGB(Z))+0.0001)**b_coeff(xi,46)])

            if(np.isnan(b_coeff(xi,45)*(M/M_FGB(Z))**0.414 + alpha_bl*(-1)*np.abs(np.log10(M/M_FGB(Z)))**b_coeff(xi,46))):
                pass
                #print('tau error')
                #print(b_coeff(xi,45),M,M_FGB(Z),alpha_bl,np.abs(np.log10(M/M_FGB(Z)))**b_coeff(xi,46))
                #exit()
            return b_coeff(xi,45)*(M/M_FGB(Z))**0.414 + alpha_bl*(-1)*np.abs(np.log10(M/M_FGB(Z)))**b_coeff(xi,46)

        else:
            #print('tau_bl')
            #print(f_bl(M,xi,M_c))
            #print(( 1 - b_coeff(xi,47)) * (f_bl(M,xi,M_c) / (f_bl(M_FGB(Z),xi,M_c))))
            if(np.isnan(( 1 - b_coeff(xi,47)) * (f_bl(M,xi,M_c) / (f_bl(M_FGB(Z),xi,M_c))))): 
                pass
                #print('tau error 2')
                #print(b_coeff(xi,47),f_bl(M,xi,M_c),(f_bl(M_FGB(Z),xi,M_c)))  
                #exit()                  
            return ( 1 - b_coeff(xi,47)) * (f_bl(M,xi,M_c) / (f_bl(M_FGB(Z),xi,M_c)))           

def L_BAGB(M,xi):
        #Luminosity at the base of the AGB branch, eq. 56
        if M < M_HeF(xi):
            L_BAGBMHeF = ( b_coeff(xi,31) + b_coeff(xi,32)*M**(b_coeff(xi,33)+1.8) ) / ( b_coeff(xi,34) + M**b_coeff(xi,33) ) 

            alpha_3 = ( b_coeff(xi,29)*M_HeF(xi)**b_coeff(xi,30) - L_BAGBMHeF ) / L_BAGBMHeF 

            return (b_coeff(xi,29)*M**b_coeff(xi,30) ) / ( 1 + alpha_3*np.exp(15*(M-M_HeF(xi))))
        else:

            return ( b_coeff(xi,31) + b_coeff(xi,32)*M**(b_coeff(xi,33)+1.8) ) / ( b_coeff(xi,34) + M**b_coeff(xi,33) ) 

def L_ZHe(M,xi):
        #Zero age mean sequence luminosity of a naked He star, Eq. 77
        return (15262 * M**10.25) / ( M**9 + 29.54*M**7.5 + 31.18*M**6 + 0.0469 )

def R_ZHe(M,xi):
        #Zero age main sequence radius of a naked He star, Eq. 78
        return (0.2391*M**4.6) / (M**4 + 0.162*M**3 + 0.0065)

def t_HeMS(M,xi):
        #He burning lifetime of a naked he star, Eq. 79
        return (0.4129 + 18.81*M**4 + 1.853*M**6) / (M**6.5)

def R_mHe(M,xi,M_c):
        #Minimum radius during blue loop, eq. 55
        if M < M_HeF(xi):

            #mu = mu_HeB(M,xi,M_c)
            mu = M/M_HeF(xi)
            #R_mHeMHeF = ( b_coeff(xi,24)*M + ((b_coeff(xi,25)*M)**(b_coeff(xi,26))) * M**b_coeff(xi,28) ) / b_coeff(xi,27) + M**b_coeff(xi,28)
            #R_mHeMHeF = ( b_coeff(xi,24)*M_HeF(xi) + ((b_coeff(xi,25)*M_HeF(xi))**(b_coeff(xi,26))) * (M_HeF(xi)**b_coeff(xi,28)) ) / (b_coeff(xi,27) + M_HeF(xi)**b_coeff(xi,28))
            #R_mHeMHeF = R_mHe(M_HeF(xi),xi,M_c)
            
            R_mHeMHeF = R_mHe(M_HeF(xi),xi,M_cHeI(M_HeF(xi),xi))

            #print(M_c)
            #print(M_cHeI(M_HeF(xi),xi))
            #print(mu_HeB(M_HeF(xi),xi,M_c))
            #print(mu_HeB(M_HeF(xi),xi,M_cHeI(M_HeF(xi),xi)))
            #print(L_ZHe(M_c, xi))
            #print(L_ZHe(M_cHeI(M_HeF(xi),xi),xi))


            #print('old')
            #print(L_ZAHB(M_HeF(xi),xi,M_c))
            #print('new')
            #print(L_ZAHB(M_HeF(xi),xi,M_cHeI(M_HeF(xi),xi)))

            return R_GB(M,xi,L_ZAHB(M,xi,M_c)) * ( R_mHeMHeF / R_GB(M,xi,L_ZAHB(M_HeF(xi),xi,M_c)))**mu
            #return R_GB(M,xi,L_ZAHB(M,xi,M_c)) * ( R_mHeMHeF / R_GB(M,xi,L_ZAHB(M_HeF(xi),xi,M_cHeI(M_HeF(xi),xi))))**mu
        else:
            #print(( b_coeff(xi,24)*M + (b_coeff(xi,25)*M)**(b_coeff(xi,26)) * (M**b_coeff(xi,28)) ) / ( b_coeff(xi,27) + M**b_coeff(xi,28) ))
            #print([( b_coeff(xi,24)*M + (b_coeff(xi,25)*M)**(b_coeff(xi,26)) * (M**b_coeff(xi,28)) ),( b_coeff(xi,27) + M**b_coeff(xi,28) )])
            return ( b_coeff(xi,24)*M + (b_coeff(xi,25)*M)**(b_coeff(xi,26)) * (M**b_coeff(xi,28)) ) / ( b_coeff(xi,27) + M**b_coeff(xi,28) )

def t_He(M,xi,M_c):
        #lifetime of the core helium burning stage, eq. 57

        if M < M_HeF(xi):
            #t_HeMHeF = t_BGB(M,xi) * ( b_coeff(xi,41)*M**b_coeff(xi,42) + b_coeff(xi,43)*M**5) / (b_coeff(xi,44) + M**5)
            t_HeMHeF = t_He(M_HeF(xi),xi,M_c)
            alpha_4 = (t_HeMHeF - b_coeff(xi,39)) / b_coeff(xi,39)

            #print(( b_coeff(xi,39) + (t_HeMS(M_c,xi) - b_coeff(xi,39)) * (1-mu_HeB(M,xi,M_c))**b_coeff(xi,40) ) * (1 + alpha_4*np.exp(15*(M - M_HeF(xi)))))


            return ( b_coeff(xi,39) + (t_HeMS(M_c,xi) - b_coeff(xi,39)) * (1-mu_HeB(M,xi,M_c))**b_coeff(xi,40) ) * (1 + alpha_4*np.exp(15*(M - M_HeF(xi))))
        else:
            return t_BGB(M,xi) * ( b_coeff(xi,41)*M**b_coeff(xi,42) + b_coeff(xi,43)*M**5) / (b_coeff(xi,44) + M**5)

# ------ AGB phase ------- #

def M_cBAGB(M,xi):
            #Core mass at beginning of AGB, Eq. 66
            return ( b_coeff(xi,36)*M**b_coeff(xi,37) + b_coeff(xi,38) )**0.25

def R_AGB(M, xi, L):
        #AGB radius when luminosity is known, Eq. 74

            #A = min(b_coeff(xi,51)*M**(-b_coeff(xi,52)),b_coeff(xi,53)*M**(-b_coeff(xi,54)))
            #R_AGBLHeI = A * ( L_HeI(M,xi)**b_coeff(xi,1) + b_coeff(xi,2)*L_HeI(M,xi)**(b_coeff(xi,55)*b_coeff(xi,3)) )

        if M >= M_HeF(xi):
           b = b_coeff(xi,50)
           A = min(b_coeff(xi,51) * M**(-b_coeff(xi,52)) , b_coeff(xi,53)*M**(-b_coeff(xi,54)))

           return A * ( L**b_coeff(xi,1) + b_coeff(xi,2)*L**b)

        elif M<=(M_HeF(xi)-0.2):
           b = b_coeff(xi,3)
           A = b_coeff(xi,56) + b_coeff(xi,57)*M 

           return A * ( L**b_coeff(xi,1) + b_coeff(xi,2)*L**b)

        else:
            Rlow = R_AGB(M_HeF(xi)-0.2,xi,L)
            Rhigh = R_AGB(M_HeF(xi),xi,L)

            return np.interp( M, [M_HeF(xi)-0.2,M_HeF(xi)], [Rlow, Rhigh] )

# ================= Zero-age Main-sequence (ZAMS) radii and luminosities, from Tout, Pols, Eggleton and Han 1996 (MNRAS 281, 257-262) ==================== #


def lum_coeff_matrix():

        # Coefficients for ZAMS luminosity, Table 1:
        row1 = [ 0.39704170,  -0.32913574,  0.34776688,  0.37470851, 0.09011915 ]
        row2 = [ 8.52762600, -24.41225973, 56.43597107, 37.06152575, 5.45624060 ]
        row3 = [ 0.00025546,  -0.00123461, -0.00023246,  0.00045519, 0.00016176 ]
        row4 = [ 5.43288900,  -8.62157806, 13.44202049, 14.51584135, 3.39793084 ]
        row5 = [ 5.56357900, -10.32345224, 19.44322980, 18.97361347, 4.16903097 ]
        row6 = [ 0.78866060,  -2.90870942,  6.54713531,  4.05606657, 0.53287322 ]
        row7 = [ 0.00586685,  -0.01704237,  0.03872348,  0.02570041, 0.00383376 ]

        return np.matrix( [row1, row2, row3, row4, row5, row6, row7] )

def rad_coeff_matrix():

        # Coefficients for ZAMS Radius, Table 2:
        row1 = [  1.71535900,  0.62246212,  -0.92557761,  -1.16996966, -0.30631491 ]
        row2 = [  6.59778800, -0.42450044, -12.13339427, -10.73509484, -2.51487077 ]
        row3 = [ 10.08855000, -7.11727086, -31.67119479, -24.24848322, -5.33608972 ]
        row4 = [  1.01249500,  0.32699690,  -0.00923418,  -0.03876858, -0.00412750 ]
        row5 = [  0.07490166,  0.02410413,   0.07233664,   0.03040467,  0.00197741 ]
        row6 = [  0.01077422,  0.        ,   0.        ,   0.        ,  0.         ]
        row7 = [  3.08223400,  0.94472050,  -2.15200882,  -2.49219496, -0.63848738 ]
        row8 = [ 17.84778000, -7.45345690, -48.96066856, -40.05386135, -9.09331816 ]
        row9 = [  0.00022582, -0.00186899,   0.00388783,   0.00142402, -0.00007671 ]

        return np.matrix( [row1, row2, row3, row4, row5, row6, row7, row8, row9] )

def coeff(Z, params):

        assert np.size(params) == 5

        a, b, c, d, e = np.squeeze(np.asarray(params))

        # Coefficients, Eq. 3:

        Zsun = 0.02
        x = np.log10(Z/Zsun)

        return a + b*x + c*x**2. + d*x**3. + e*x**4.

def L_ZAMS(M, Z):

        # ZAMS Luminosity, Eq. 1:

        mx = lum_coeff_matrix()

        ms = np.sqrt(M)

        num = coeff(Z, mx[0,:])*M**5.*ms + coeff(Z, mx[1,:])*M**11.
        den = coeff(Z, mx[2,:]) + M**3. + coeff(Z, mx[3,:])*M**5. + coeff(Z, mx[4,:])*M**7. + coeff(Z, mx[5,:])*M**8. + coeff(Z, mx[6,:])*M**9.*ms

        return num / den

def R_ZAMS(M, Z):

        # ZAMS Radius, Eq. 2:

        mx = rad_coeff_matrix()
        ms = np.sqrt(M)

        num = (coeff(Z, mx[0,:])*M**2. + coeff(Z, mx[1,:])*M**6.)*ms + coeff(Z, mx[2,:])*M**11. + (coeff(Z, mx[3,:]) + coeff(Z, mx[4,:])*ms)*M**19.
        den = coeff(Z, mx[5,:]) + coeff(Z, mx[6,:])*M**2. + (coeff(Z, mx[7,:])*M**8. + M**18. + coeff(Z, mx[8,:])*M**19.)*ms

        return num / den

# ============== Luminosity as a function of time in the MS ====================================

def Luminosity(M, xi, t): # Luminosity as a function of time, Eq. 12


        Z = get_Z(xi)

        # Fractional time scale on the MS, Eq. 11
        tau = t / t_MS(M, xi)

        tau1, tau2 = tau_12(M, xi, t)

        LZAMS = L_ZAMS(M, Z)

        LTMS = L_TMS(M, xi)

        eta = eta_exp(M, Z)

        DeltaL = Delta_L(M, xi)

        alphaL = alpha_L(M, xi)
        betaL  =  beta_L(M, xi)

        logratio = alphaL*tau + betaL*tau**eta + ( np.log10(LTMS/LZAMS) - alphaL - betaL )*tau**2. - DeltaL*( tau1**2. - tau2**2. )

        return LZAMS * 10.**logratio

#Luminosity = np.vectorize(Luminosity)


def tau_12(M, xi, t):

        thook = t_hook(M, xi)

        eps = 0.01

        # Eq. 14:
        tau1 = min(1.0, t / thook)

        # Eq. 15:
        tau2 = max( 0.0, min( 1.0, (t - (1.0 - eps)*thook)/(eps*thook) ) )

        return tau1, tau2


def alpha_L(M, xi):

        # Luminosity alpha Coefficient

        # Eq. 19a:
        if M >= 2.0:
                return (a_coeff(xi,45) + a_coeff(xi,46)*M**a_coeff(xi,48)) / (M**0.4 + a_coeff(xi,47)*M**1.9)

        # Eq. 19b:
        if M < 0.5:
                return a_coeff(xi,49)

        if (M >= 0.5) & (M < 0.7):
                return a_coeff(xi,49) + 5.0 * (0.3 - a_coeff(xi,49)) * (M - 0.5)

        if (M >= 0.7) & (M < a_coeff(xi,52)):
                return 0.3 + (a_coeff(xi,50) - 0.3)*(M - 0.7)/(a_coeff(xi,52) - 0.7)

        if (M >= a_coeff(xi,52)) & (M < a_coeff(xi,53)):
                return a_coeff(xi,50) + (a_coeff(xi,51) - a_coeff(xi,50))*(M - a_coeff(xi,52))/(a_coeff(xi,53) - a_coeff(xi,52))

        if (M >= a_coeff(xi,53)) & (M < 2.0):

                B = alpha_L(2.0, xi)

                return a_coeff(xi,51) + (B - a_coeff(xi,51))*(M - a_coeff(xi,53))/(2.0 - a_coeff(xi,53))

def beta_L(M, xi):

        # Luminosity beta Coefficient, Eq. 20:

        beta = max( 0.0, a_coeff(xi,54) - a_coeff(xi,55)*M**a_coeff(xi,56) )

        if (M > a_coeff(xi,57)) & (beta > 0.0):

                B = beta_L(a_coeff(xi, 57), xi)

                beta = max(0.0, B - 10.0*(M - a_coeff(xi,57))*B)

        return beta

def eta_exp(M, Z):

        # Exponent eta, Eq. 18:

        if Z <= 0.0009:

                if M <= 1.0:
                        eta = 10
                elif M >= 1.1:
                        eta = 20
                else:
                        eta = np.interp( M, [1.0, 1.1], [10., 20.])

        else:
                eta = 10

        return eta


def Delta_L(M, xi):

        # Luminosity perturbation, Eq. 16:

        Mhook = M_hook(xi)

        if M <= Mhook:
                return 0.

        elif (M > Mhook) & (M < a_coeff(xi,33)):

                B = Delta_L( a_coeff(xi,33), xi)

                return B * ( (M - Mhook)/(a_coeff(xi,33) - Mhook) )**0.4

        elif M >= a_coeff(xi, 33):

                return min( a_coeff(xi,34) / (M**a_coeff(xi,35)), a_coeff(xi,36) / (M**a_coeff(xi,37)) )


def L_TMS(M, xi):

        # Luminosity at the end of the MS, Eq. 8:

        return (a_coeff(xi,11)*M**3. + a_coeff(xi,12)*M**4. + a_coeff(xi,13)*M**(a_coeff(xi,16)+1.8) ) / (a_coeff(xi,14) + a_coeff(xi,15)*M**5. + M**a_coeff(xi,16))


def L_BGB(M, xi):

        c2 = 9.301992
        c3 = 4.637345

        # Luminosity at the base of the GB, Eq. 10:

        return (a_coeff(xi,27)*M**a_coeff(xi,31) + a_coeff(xi,28)*M**c2) / (a_coeff(xi,29) + a_coeff(xi,30)*M**c3 + M**a_coeff(xi,32))

def L_EHG(M,xi):

    # Luminosity at end of Horizontal Gap, above Eq. 8

    if M<M_FGB(get_Z(xi)):
        return L_BGB(M,xi)
    else:
        return L_HeI(M,xi)

def R_EHG(M,xi):

    # Radius at end of H Gap, above Eq. 8

    if M<M_FGB(get_Z(xi)):
        return R_GB(M,xi,L_BGB(M,xi))
    else:

        M_c = M_cHeI(M,xi)

        return R_HeI(M,xi,M_c)

'''
Z = 0.02
xi = np.log10(Z/0.02)

M = 1.25

t = np.linspace(0., t_MS(M, xi), 5000)
Lum = Luminosity(M, xi, t)

plt.plot( t/t_MS(M, xi), np.log10(Lum) )
plt.xlim([0,1.05])
plt.xlabel('tau', fontsize=18)
plt.ylabel('log L/LSun', fontsize=18)
plt.show()
'''

# ============== Radius as a function of time in the MS ====================================


def Radius(M, xi, t): # Radius as a function of time, Eq. 12

        Z = get_Z(xi)

        # Fractional time scale on the MS, Eq. 11
        tau = t / t_MS(M, xi)

        tau1, tau2 = tau_12(M, xi, t)

        if(M<1.4):
            RZAMS = R_ZAMS(M,get_Z(0.0))
        else:
            RZAMS = R_ZAMS(M, Z)

        RTMS = R_TMS(M, xi)

        alphaR = alpha_R(M, xi)
        betaR  =  beta_R(M, xi)
        gammaR  = gamma_R(M, xi)

        DeltaR = Delta_R(M, xi)

        #print(M)
        #print(RTMS)
        #print(RZAMS)
        logratio = alphaR*tau + betaR*tau**10. + gammaR*tau**40. + (np.log10(RTMS/RZAMS) - alphaR - betaR - gammaR)*tau**3. - DeltaR*( tau1**3. - tau2**3. )
        #print('tau '+str(tau))
        #print('logratio '+str(logratio))
        R = RZAMS * 10.**logratio

        # Low-mass MS stars can be degenerate, Eq.24:

        if M < M_HeF(xi):

                X = 0.76 - 3.0*Z # Initial idrogen abundance

                R = max(R, 0.0258 * (1.0 + X)**(5./3.) * M**(-1./3.))

        return R

#Radius = np.vectorize(Radius)

def alpha_R(M, xi):

        # Radius alpha Coefficient

        a68 = min(a_coeff(xi,68), a_coeff(xi,66))

        # Eq. 21a:
        if (M >= a_coeff(xi,66)) & (M <= a_coeff(xi,67)):
                return (a_coeff(xi,58) * M**a_coeff(xi,60)) / (a_coeff(xi,59) + M**a_coeff(xi,61))

        # Eq. 21b:
        if M < 0.5:
                return a_coeff(xi,62)

        if (M >= 0.5) & (M < 0.65):
                return a_coeff(xi,62) + (a_coeff(xi,63) - a_coeff(xi,62)) * (M - 0.5) / 0.15

        if (M >= 0.65) & (M < a68):

                return a_coeff(xi,63) + (a_coeff(xi,64) - a_coeff(xi,63)) * (M - 0.65) / (a68 - 0.65)

        if (M >= a68) & (M < a_coeff(xi,66)):

                B = alpha_R( a_coeff(xi, 66), xi)

                return a_coeff(xi,64) + (B - a_coeff(xi,64)) * (M - a68) / (a_coeff(xi,66) - a68)

        if M > a_coeff(xi,67):

                C = alpha_R( a_coeff(xi,67), xi)

                return C + a_coeff(xi,65) * (M - a_coeff(xi,67))


def beta_R(M, xi):

        # Radius beta coefficient

        # Eq. 22a:
        if (M >= 2.0) & (M <= 16.0):
                beta1 = (a_coeff(xi,69) * M**3.5) / (a_coeff(xi,70) + M**a_coeff(xi,71))

        # Eq. 22b:
        if M <= 1.0:

                beta1 = 1.06

        if (M > 1.0) & (M < a_coeff(xi,74)):

                beta1 = 1.06 + (a_coeff(xi,72) - 1.06) * (M - 1.0) / (a_coeff(xi,74) - 1.06)

        if (M >= a_coeff(xi,74)) & (M < 2.0):

                B = beta_R(2.0, xi) + 1

                beta1 = a_coeff(xi,72) + (B - a_coeff(xi,72)) * (M - a_coeff(xi,74)) / (2.0 - a_coeff(xi,74))

        if M > 16.0:

                C = beta_R(16.0, xi) + 1

                beta1 = C + a_coeff(xi,73) * (M - 16.0)

        return beta1 - 1


def gamma_R(M, xi):

        # Radius Gamma coefficient, Eq. 23:

        if M > a_coeff(xi,75) + 0.1:
                gamma = 0.0

        elif M <= 1.0:
                gamma = a_coeff(xi,76) + a_coeff(xi,77) * (M - a_coeff(xi,78))**a_coeff(xi,79)

        elif (M > 1.0) & (M <= a_coeff(xi,75)):

                B = gamma_R(1.0, xi)

                gamma = B + (a_coeff(xi,80) - B) * ( (M - 1.0)/(a_coeff(xi,75) - 1.0) )**a_coeff(xi,81)

        elif (M > a_coeff(xi,75)) & (M < a_coeff(xi,75) + 0.1):

                if a_coeff(xi,75) == 1.0:

                        C = gamma_R(1.0, xi)

                else:
                        C = a_coeff(xi,80)

                gamma = C - 10.0 * (M - a_coeff(xi,75)) * C

        #if gamma < 0:
        #       print 'ERROR: gamma < 0'
        #       gamma = 0.0
                #pass

        return gamma


def Delta_R(M, xi):

        # Radius perturbation, Eq. 17:

        Mhook = M_hook(xi)

        if M <= Mhook:

                return 0.

        elif (M > Mhook) & (M <= a_coeff(xi,42)):

                return a_coeff(xi,43) * ( (M - Mhook) / (a_coeff(xi,42) - Mhook) )**0.5

        elif (M > a_coeff(xi,42)) & (M < 2.0):

                B = Delta_R(2.0, xi)

                return a_coeff(xi,43) + (B - a_coeff(xi,43)) * ( (M - a_coeff(xi,42)) / (2.0 - a_coeff(xi,42) ))**a_coeff(xi,44)

        elif M >= 2.0:

                return (a_coeff(xi,38) + a_coeff(xi,39)*M**3.5) / (a_coeff(xi,40)*M**3. + M**a_coeff(xi,41)) - 1.0


def R_TMS(M, xi):

        # Radius at the end of the MS

        Mstar = a_coeff(xi,17) + 0.1
        c1 = -8.672073e-2

        # Eq. 9a:
        if M <= a_coeff(xi,17):

                RTMS =  (a_coeff(xi,18) + a_coeff(xi,19)*M**a_coeff(xi,21)) / (a_coeff(xi,20) + M**a_coeff(xi,22))

        elif M >= Mstar:

                RTMS = (c1*M**3. + a_coeff(xi,23)*M**a_coeff(xi,26) + a_coeff(xi,24)*M**(a_coeff(xi,26) + 1.5)) / (a_coeff(xi,25) + M**5.)

        else:

                M1 = a_coeff(xi,17)
                R1 = R_TMS(M1, xi)

                M2 = Mstar
                R2 = R_TMS(Mstar, xi)

                RTMS = np.interp( M, [M1, M2], [R1, R2]) # straight-line interpolation

                if M < 0.5:

                        Z = get_Z(xi)

                        RTMS = max(RTMS, 1.5 * R_ZAMS(M, Z))

        return RTMS


'''
Rad = Radius(M, xi, t)

plt.plot( t/t_MS(M, xi), np.log10(Rad) )
plt.xlim([0,1.05])
plt.xlabel('tau', fontsize=18)
plt.ylabel('log R/RSun', fontsize=18)
plt.show()
'''

# ============= Giant Branch Evolution ======= $

def GB_params(M,xi):

    def p_param(M,xi):

        if M <= M_HeF(xi):
            p = 6
        elif M>=2.5:
            p = 5
        else:
            p = np.interp(M,[M_HeF(xi),2.5],[6,5])
        return p

    def q_param(M,xi): #Below eq. 38

        if M <= M_HeF(xi):
            q = 3
        elif M>=2.5:
            q = 2
        else:
            q = np.interp(M,[M_HeF(xi),2.5],[3,2])

        return q

    def B_param(M,xi): #Below eq. 38

        return max( 3e4, 500 + 1.75e4 * (M**0.6))
        
    def D_param(M,xi): #below eq. 38

        D_0 = 5.37+0.135*xi

        if M<=M_HeF(xi):
            logD = D_0 
        elif M>2.5:
            logD = max( -1.0, 0.975*D_0 - 0.18*M, 0.5*D_0 - 0.06*M)
        else:

            logDlo = D_0
            logDhi = max( -1.0, 0.975*D_0 - 0.18*M, 0.5*D_0 - 0.06*M)
            
            logD =  np.interp( M, [M_HeF(xi),2.5], [logDlo, logDhi] )

        return 10**logD

    def A_H_param(M,xi):
        #Below eq. 43
        return 10 ** (max( -4.8, min(-5.7 + 0.8*M, -4.1 + 0.14*M) ) )

    p = p_param(M,xi)
    q = q_param(M,xi)
    B = B_param(M,xi)
    D = D_param(M,xi)
    A_H = A_H_param(M,xi)

    #Eq. 38
    M_x = (B / D)**( 1 / (p-q) )

    #eq. 37
    L_x =  min( B * M_x**q, D * M_x**p)

    #Eq. 40
    t_inf1 = t_BGB(M,xi) + (1. / ( (p-1) * A_H * D) ) * ( (D/L_BGB(M,xi))**( (p-1) / p ) )
    
    #Eq. 41
    t_x = t_inf1 - (t_inf1 - t_BGB(M,xi)) *  ( (L_BGB(M,xi)/L_x)**( (p-1) / p ) )

    #Eq. 42
    t_inf2 = t_x + (1. / ( (q-1)*A_H * B) ) * ( (B/L_x)**( (q-1) / q ) )

    return p, q, B, D, A_H, M_x, L_x, t_inf1, t_inf2, t_x

def Luminosity_GB(M, xi, t): # Luminosity as a function of time, Eq. 37

        p, q, B, D, A_H, M_x, L_x, t_inf1, t_inf2, t_x = GB_params(M,xi)

        #print('snark?')
        if M <= 1e5*M_HeF(xi):
            #Eq. 39
            if t <= t_x:
                M_c = ( (p-1) * A_H * D * (t_inf1 - t) )**( 1/(1-p) )
            elif t>t_x:
                M_c = ( (q-1) * A_H * B * (t_inf2 - t) )**( 1/(1-q) )

            #print(M_c)

        elif M >= M_HeF(xi):

            c1 = 9.20925e-5
            c2 = 5.402216

            p, q, B, D, A_H, M_x, L_x, t_inf1, t_inf2, t_x = GB_params(M_HeF(xi),xi)

            if t <= t_x:
                M_c = ( (p-1) * A_H * D * (t_inf1 - t) )**( 1/(1-p) )
            elif t>t_x:
                M_c = ( (q-1) * A_H * B * (t_inf2 - t) )**( 1/(1-q) )

            #C = M_c(L_BGB(M_HeF(xi),xi))**4 - c1*M_HeF(xi)**c2
            C = M_c**4 - c1*M_HeF(xi)**c2

            #Below eq. 44
            #M_cBGB = 0.098*M**1.35

            #print(0.95*M_cBAGB(M,xi))
            #print((C+c1*M**c2)**(1/4))
            M_cBGB = min(0.95*M_cBAGB(M,xi), (C+c1*M**c2)**(1/4))

            #Below eq. 45
            tau = ( t - t_BGB(M,xi) ) / (t_HeI(M,xi) - t_BGB(M,xi))

            #print(tau)
            #Eq. 45
            M_c = M_cBGB + (M_cHeI(M,xi) - M_cBGB) * tau

        #p, q, B, D, A_H, M_x, L_x, t_inf1, t_inf2, t_x = GB_params(M,xi)

        #print([B*M_c**q, D*M_c**p])
        return min(B*M_c**q,D * M_c**p)

#Luminosity = np.vectorize(Luminosity)

def Radius_GB(M, xi, t): # Radius as a function of time, Eq. 46

        L = Luminosity_GB(M,xi,t)

        return R_GB(M,xi,L)

#Luminosity = np.vectorize(Luminosity)

# ================ Core helium burning, Sec. 5.3 ============= #

def Luminosity_Radius_HeB(M, xi, t): # Luminosity as a function of time, Eq. 61

        #p, q, B, D, A_H, M_x, L_x, t_inf1, t_inf2, t_x = GB_params(M,xi)

        Z = get_Z(xi)

        M_c = M_cHeI(M,xi)
        #print(['M_c',M_c])
        #print(L_ZAHB(0.7,xi,M_c))
        #print(L_BAGB(0.7,xi))
        #Above eq. 59
        tau = (t - t_HeI(M,xi)) / t_He(M,xi,M_c)

        #Eq. 67
        #M_c = (1 - tau)*M_cHeI(M,xi) + tau*M_cBAGB(M,xi)
        #print(M_c)

        #Eq. 59
        if M < M_HeF(xi):

            tau_x = 0
            tau_y = 1

            L_x = L_ZAHB(M,xi,M_c)
            #print(L_x)

            R_x = R_ZAHB(M,xi,M_c)

        elif M_HeF(xi) <= M and M < M_FGB(Z):

            tau_x = 1-tau_bl(M,xi,M_c)
            tau_y = 1

            L_x = L_minHe(M,xi)
            #print(L_x)

            R_x = R_GB(M,xi,L_minHe(M,xi))

        elif M>=M_FGB(Z):

            tau_x = 0
            tau_y = tau_bl(M,xi,M_c)
            #print(tau_y)
            L_x = L_HeI(M,xi)

            R_x = R_HeI(M,xi,M_c)

        M_c = (1 - tau)*M_cHeI(M,xi) + tau*M_cBAGB(M,xi)       

        ex = min(2.5, max(0.4, R_mHe(M,xi,M_c) / R_x ))

        #print(['M_c2',ex,L_BAGB(M,xi)/L_x,L_x,M_c])

        #Eq. 62
        lam = ( (tau - tau_x) / (1 - tau_x) )**ex

        #Eq. 63
        lamprime = ( (tau_x - tau) / tau_x )**3

        if tau>1:
            tau = 1
            #print('SNARK')
        #Eq. 61
        if tau >= tau_x and tau <= 1:
            L = L_x * (L_BAGB(M,xi)/L_x)**lam
            #print([L_x,L_BAGB(M,xi),lam])
        elif tau >= 0 and tau < tau_x:
            L = L_x * (L_HeI(M, xi)/L_x)**lamprime
            #print([L_x,L_HeI(M, xi),lamprime])
        else:
            pass
            #print('tau error')
            #print([tau,tau_x])
            #exit()
        
        #if(L>1e5):
            #print("LLLLLL")
            #print([tau,tau_x])
            #print([L,L_x,L_HeI(M, xi),lamprime])
            #print([L,L_x,L_BAGB(M,xi),lam])       
            #exit()

        #if M>M_FGB(Z) and R_mHe(M,xi,M_c)>R_AGB(M,xi,L_HeI(M,xi)):
                #print('RRRRRRR')
                #print([M,R_mHe(M,xi,M_c),R_AGB(M,xi,L_HeI(M,xi))])
                #exit()

        #Eq. 64
        #print([M, lam,ex, tau,tau_x,tau_y])
        #print(tau_bl(M,xi,M_c))
        
        if np.isnan(tau_y):
            R = -99
            #print('tau error')
            #print([tau,tau_x,tau_y])
            #exit()
        elif tau >= 0 and tau<tau_x:
            #print('yes1')
            R = R_GB(M,xi,L)
        elif tau >= tau_y and tau<=1:
            R = R_AGB(M, xi, L)
            #print('yes2')
        elif tau >= tau_x and tau<tau_y:
            #print('yes3')
            if M <= M_FGB(Z):
                L_y = L_BAGB(M,xi)
            else:
                lam = ( (tau_y - tau_x) / (1 - tau_x) )**ex
                L_y = L_x * (L_BAGB(M,xi)/L_x)**lam

            #Below Eq. 63
            R_y = R_AGB(M, xi, L_y)
            R_min = min(R_mHe(M,xi,M_c), R_x)

            #Eq. 65
            rho = (np.log(R_y/R_min))**(1/3) * ( (tau - tau_x) / (tau_y - tau_x) ) - (np.log(R_x/R_min))**(1/3) * ( (tau_y - tau) / (tau_y - tau_x) )

            R = R_min * np.exp( np.abs(rho)**3 )        

        #print([tau,tau_x])
        #print(L,R)
        #if np.isnan(R):
        #    R = 100

        return L, R

#Luminosity = np.vectorize(Luminosity)


#def Radius_GB(M, xi, t): # Radius as a function of time, Eq. 46

#        A = min (b_coeff(xi,4) * M**(-b_coeff(xi,5)) , b_coeff(xi,6)*M**(-b_coeff(xi,7)) )

#        L = Luminosity_GB(M,xi,t)

#        return A * ( L**b_coeff(xi,1) + b_coeff(xi,2)*L**b_coeff(xi,3) )

#Luminosity = np.vectorize(Luminosity)

def Luminosity_HG(M,xi,t):
    
    # Luminosity on Hertzsprung Gap, Eq. 26

    tau = ( t - t_MS(M,xi) ) / (t_BGB(M,xi) - t_MS(M,xi))

    return L_TMS(M,xi) * (L_EHG(M,xi) / L_TMS(M,xi)) ** tau

def Radius_HG(M,xi,t):
    
    # Luminosity on Hertzsprung Gap, Eq. 26

    tau = ( t - t_MS(M,xi) ) / (t_BGB(M,xi) - t_MS(M,xi))

    return R_TMS(M,xi) * (R_EHG(M,xi) / R_TMS(M,xi)) ** tau

# ================= Mass Loss ( Section 7 ) =================== #


def Mdot_NJ(M, L, R, Z):

        # Mass-Loss prescription given by Nieuwenhuijzen & De Jager (1990), Paragraph 7

        return 9.6e-15 * (Z/0.02)**0.5 * R**0.81 * L**1.24 * M**0.16 # [Msun / yr]


def Mdot_LBV(L, R):

        # LBV-like mass-loss for stars beyond the Humphreys-Davidson limit, Paragraph 7

        return 0.1 * (1.e-5 * R * L**0.5  - 1.0)**3. * ( L / 6.e5 - 1.0) # [Msun / yr]


def Mdot_WR(mu, L):

        # Wolf-Rayet-like mass loss, for small hydrogen-evelope mass (mu < 1.0), Paragraph 7

        if mu < 1.0:
                return 1.e-13 * L**1.5 * (1.0 - mu) # [Msun / yr]
        else:
                return 0.

def Mass_Loss(M, L, R, xi, t):

        Z = get_Z(xi)

        Mdot = Mdot_WR(1., L)

        if L > 4000.:
                Mdot = max(Mdot, Mdot_NJ(M, L, R, Z) )

        if (L > 6.e5) & (1.e-5 * R * L**0.5 > 1.0):
                Mdot = Mdot + Mdot_LBV(L, R)

        Mloss = Mdot * t*1.e6 # [MSun]

        return M - Mloss

#Mass_Loss = np.vectorize(Mass_Loss)

# ========================


def T_eff(R, L, t):

        if R==-99:

            return 1

        else:

            R = R * u.solRad
            L = L * u.L_sun

            return ((L / (4.*np.pi*R**2.*const.sigma_sb))**(1./4.)).to(u.K).value # Effective Temperature, [K]
'''
Teff = T_eff(Rad, Lum, t)

plt.plot( t/t_MS(M, xi), Teff )
plt.xlim([0,1.05])
plt.xlabel('tau', fontsize=18)
plt.ylabel('Teff [K]', fontsize=18)
plt.show()
'''

#  ==================== General Function to Get Stellar Parameters =========================


def get_TempRad(M, xi, t):

        L = Luminosity(M, xi, t)
        #print('lum '+str(L))
        R = Radius(M, xi, t)
        #print('R '+str(R))

        T = T_eff(R, L, t)

        #Mt = Mass_Loss(M0, L, R, xi, t)

        #t = t * t_MS(Mt, xi)/t_MS(M0,xi)

        #L = Luminosity(Mt, xi, t)
        #R = Radius(Mt, xi, t)

        return T, R

def get_TempRadL(M, xi, t):

        L = Luminosity(M, xi, t)
        #print('lum '+str(L))
        R = Radius(M, xi, t)
        #print('R '+str(R))

        T = T_eff(R, L, t)

        #Mt = Mass_Loss(M0, L, R, xi, t)

        #t = t * t_MS(Mt, xi)/t_MS(M0,xi)

        #L = Luminosity(Mt, xi, t)
        #R = Radius(Mt, xi, t)

        return T, R, L

def get_TempRadL_HG(M, xi, t):

        L = Luminosity_HG(M, xi, t)
        R = Radius_HG(M, xi, t)

        T = T_eff(R, L, t)

        return T, R, L

def get_TempRad_GB(M, xi, t):

        L = Luminosity_GB(M, xi, t)
        R = Radius_GB(M, xi, t)

        T = T_eff(R, L, t)

        return T, R

def get_TempRadL_GB(M, xi, t):

        L = Luminosity_GB(M, xi, t)
        R = Radius_GB(M, xi, t)

        T = T_eff(R, L, t)

        return T, R, L

def get_Ab1tob7(M, xi, t):

        return min (b_coeff(xi,4) * M**(-b_coeff(xi,5)) , b_coeff(xi,6)*M**(-b_coeff(xi,7)) ), b_coeff(xi,1), b_coeff(xi,2), b_coeff(xi,3), b_coeff(xi,4), b_coeff(xi,5), b_coeff(xi,6), b_coeff(xi,7)

def get_TempRad_CHeB(M, xi, t):

        L, R = Luminosity_Radius_HeB(M, xi, t)

        T = T_eff(R, L, t)

        return T, R

def get_TempRadL_CHeB(M, xi, t):

        L, R = Luminosity_Radius_HeB(M, xi, t)

        T = T_eff(R, L, t)

        return T, R, L

def get_t_MS(m,xi=0):

    M = m.to('Msun').value

    out = np.empty(len(M))

    for i in range(len(m)):
    #for i in tqdm(range(len(m))):
        #out[i] = max( Mu_param(M[i],xi[i])*t_BGB(M[i],xi[i]), x_param(xi[i])*t_BGB(M[i],xi[i]) )#*u.Myr 
        out[i] = t_MS(M[i], xi[i]) 

    return out*u.Myr

def get_t_BGB(m,xi=0):

    M = m.to('Msun').value

    out = np.empty(len(M))

    for i in range(len(m)):
        out[i] = t_BGB(M[i], xi[i])

    return out*u.Myr

def get_t_HeI(m,xi=0):
    M = m.to('Msun').value

    out = np.empty(len(M))

    for i in range(len(m)):
        out[i] = t_HeI(M[i], xi[i])

    return out*u.Myr

def get_t_BAGB(m,xi=0):
    M = m.to('Msun').value

    out = np.empty(len(M))

    #for i in range(len(m)):
    for i in tqdm(range(len(m))):
        #print([i,len(m)])
        M_c = M_cHeI(M[i],xi[i])
        out[i] = t_HeI(M[i], xi[i]) + t_He(M[i],xi[i],M_c)

    return out*u.Myr

'''
path = '/net/vuntus/data2/marchetti/HVS_Gaia_Prediction/'
mass, age, radius = np.loadtxt(path + 'M_tAge_R_grid_Mmin=0.1_allt.txt', unpack = True)
mass, age, temperature = np.loadtxt(path + 'M_tAge_T_grid_Mmin=0.1_allt.txt', unpack = True)
mass, age, mass_t = np.loadtxt(path + 'M_tAge_M_grid_Mmin=0.1_allt.txt', unpack = True)

mytemp = np.empty(len(mass))
myrad  = np.empty(len(mass))
mymass = np.empty(len(mass))

for i in range(0, len(mass)):
        mytemp[i], myrad[i], mymass[i] = get_TRM(mass[i], 0., age[i])
'''
