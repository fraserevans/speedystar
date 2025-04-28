# Welcome to speedystar
Python toolkit for the generation of mock catalogues of high-velocity stars

## Description

`speedystar` allows you to generate, evolve, propagate and perform mock observations of single stars ejected at high velocities. Based on [astropy](https://www.astropy.org/), [scipy](https://scipy.org/), [galpy](https://docs.galpy.org/en/v1.8.0/), [AMUSE](https://www.amusecode.org/), [mwdust](https://github.com/jobovy/mwdust), [pygaia](https://github.com/agabrown/PyGaia), [GaiaUnlimited](https://github.com/gaia-unlimited/gaiaunlimited) and others.  
  
## Setup & Installation  
Download the repository, navigate to the parent directory of `speedystar` and run 
``` 
pip install ./
```
Package requires ~12 MB of space but installation may take a while depending on the number of dependency packages that must be installed (see setup.py or requirements.txt)  

Alternatively, if you do not wish to install globally (e.g. if you want to more easily edit the `speedystar` source code), simply ensure `speedystar/` is in the working directory and make sure required packages are installed:
```
pip install -r requirements.txt
``` 

## Documentation  
You can access every method's docstring by using the help() function in python.  
  
## Getting Started

See `examples` folder for iPython notebooks with working examples. General work flow is as follows:

 1) Define an ejection model, i. e. the ejection mechanisms and associated assumptions. 
	 ```python
	 import speedystar
	ejmodel = speedystar.eject.Hills()
	 ```
2)  Create an ejection sample. Masses, velocities, ages, flight times, luminosities, radii and evolutionary stages are assigned/computed for each ejected star and become attributes to `mysample`:
	```python
	mysample = speedystar.starsample(ejmodel)
	```
  
3. Define a Galactic potential and propagate the ejection star sample through the Galaxy. Default integration timestep is 0.1 Myr. Equatorial (position, proper motion, radial velocity, heliocentric distance, parallax) and Galactocentric Cartesian (x, y, z, v<sub>x</sub>, v<sub>y</sub>, v<sub>z</sub>) are computed and become attributes to `mysample`.
	```python
	from galpy.potential.mwpotentials import MWPotential2014
	mysample.propagate(potential=MWPotential2014)
	```
  
5. Obtain mock observations of each ejected star using the [MIST](https://waps.cfa.harvard.edu/MIST/) isochrones. Apparent magnitudes in the Johnson-Cousins V, I<sub>c</sub> and Gaia G, G<sub>BP</sub>, G<sub>RP</sub> and G<sub>RVS</sub> bands are computed by default and become attributes to `mysample`. Optionally, magnitudes in other photometric systems can be computed as well, see documentation. Computing apparent magnitudes requires a `DustMap` object (see [mwdust](https://github.com/jobovy/mwdust) or the `speedystar.starsample.fetch_dust()` docstring).
  
	  ```python
	  mysample.config_dust('/path/where/large/data/files/are/stored/dust_map.h5')
	  mysample.photometry()
	```

6. Select only the stars in your sample which are of interest.
e.g. if only stars with total velocities >1000 km/s are interesting, try:
	```python
	from astropy import units as u
	idx = (mysample.GCv>1000*u.km/u.s) 
	mysample.subsample(np.where(idx)[0])
	```
	Or, if you only want stars brighter than V=16:
	```python
	import numpy as np
	idx = (mysample.V<16)
	mysample.subsample(np.where(idx)[0])
	```
	Some cuts, most notably those which determine which stars are detectable in different _Gaia_ data releases, are hard-coded in, see the `speedystar.starsample.subsample()` docstring. They can be invoked with the appropriate string argument to `.subsample()`, e.g. 
	```python
	mysample.subsample('Gaia_6D_DR3')
	```

7. Save the final sample and all its attributes to file. Catalogue can also be saved following any of the steps above. Pre-existing catalogues can be loaded with `speedystar.starsample(filename)`. Currently the only available input/output format is as a .fits table
	```python
	mysample.save('./my_catalogue.fits')
	```
  
Have fun!  
  
  ## Tips, Tricks and Troubleshooting

  - Depending on your science case, the distance from the Sun to the Galactic Centre and the circular velocity of the Milky Way at the Solar position may impact results. These are set in the `.galpyrc` configuration file.

  - By default, stars only live up to the end of the main sequence before they are considered 'dead' and removed from the sample. Evolution up until the beginning of the AGB branch can be achieved using [AMUSE](https://www.amusecode.org/) by setting `amuseflag=True` in the ejection model definition. The AMUSE installation process can be very non-trivial, however, see [here](https://amuse.readthedocs.io/en/latest/install/howto-install-AMUSE.html) 

  - A lot of speedup can be gained by calling `speedystar.subsample` immediately after creating the ejection sample. This allows you to not waste time propagating or performing mock observations on stars which are not interesting for your science case. For example, if you are certain that only stars more massive than 1 M<sub>&#9737;</sub> will  be detectable by your survey or instrument of interest, you can call `.subsample()` before `.propagate()` like so:
	```python
	idx = (mysample.m >= 1*u.Msun)
	mysample.subsample(np.where(idx)[0])
	```
- Selecting fast stars detectable by modern-day telescopes/surveys (e.g. _Gaia_) often means selecting only the rarest, brightest stars in the sample. Final samples may therefore be quite small and results will be stochastic. In such cases we recommend averaging results over many iterations of ejections+propagation+observation.
- Recall that `galpy` uses a left-handed Galactocentric coordinate system, meaning the Sun is located on the _positive_ x axis, not negative. This is important if, e.g. you are dealing with `astropy` coordinates as well, which places the Sun on the negative x axis. The best way to avoid this mix-up is to use _only_ `galpy` _or_ `astropy` coordinates.

- Exercise caution when allowing stars with low ejection velocities (v<sub>0</sub> &#8818; 200 km/s) to be propagated. Since stars are ejected directly radially away from the Galactic Centre, slow-ejected stars will quickly return towards Sgr A* on _extremely_ eccentric orbits. Fully integrating these orbits can incur substantial energy error. A timeout condition is implemented to automatically skip an orbit propagation if it takes longer than five seconds.

## Citation  
If you use `speedystar`, please cite [Contigiani et al. 2018](https://ui.adsabs.harvard.edu/abs/2019MNRAS.487.4025C/abstract) and [Evans et al. 2022](https://ui.adsabs.harvard.edu/abs/2022arXiv220514777E/abstract). If discussing the `speedystar.eject.Hills` implementation specifically, please also cite [Rossi et al. 2017](https://ui.adsabs.harvard.edu/abs/2017MNRAS.467.1844R/abstract) and [Marchetti et al. 2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.4697M/abstract).
 
This package makes heavy use of other python packages both widely-used and obscure. When relevant, please credit other works or packages as well: 
- [astropy](https://www.astropy.org/) for units, I/O, conversions
- [scipy](https://scipy.org/) for under-the-hood math stuff
- [galpy](https://docs.galpy.org/en/v1.8.0/) for orbital integration and coordinate transformations
- [mwdust](https://github.com/jobovy/mwdust) for Galactic dust maps
- [pygaia](https://github.com/agabrown/PyGaia) for *Gaia* astrometric/spectroscopic errors
- [scanninglaw](https://github.com/gaiaverse/scanninglaw) for *Gaia* astrometric spread function
- [imf](https://github.com/keflavich/imf) for initial mass function utilities
- [GaiaUnlimited](https://github.com/gaia-unlimited/gaiaunlimited) for *Gaia* selection functions

### Development & Bug Reports 

Development of `speedystar` takes place on GitHub, at [https://github.com/speedystar](https://github.com/speedystar). Bug reports, feature requests, or other issues can be filed there or via email to fraser.evans@utoronto.ca. Contributions to the software are welcome.

### Authors
- Fraser Evans (fraser.evans@utoronto.ca)
- Based on code base originally developed by Omar Contigiani
- Significant contributions from Tommaso Marchetti
- Additional contributions from Josephine Baggen, Sanne Bloot, Amber Remmelzwaal, Isabella Armstrong
- Thanks to Niccolò Veronesi for setup & installation debugging 
