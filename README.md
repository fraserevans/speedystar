# Welcome to speedystar
Python toolkit for the generation of mock catalogues of high-velocity stars

## Description

`speedystar` allows you to generate, evolve, propagate and perform mock observations of single stars ejected at high velocities. Based on [astropy](https://www.astropy.org/), [scipy](https://scipy.org/), [galpy](https://docs.galpy.org/en/v1.8.0/), [AMUSE](https://www.amusecode.org/), [mwdust](https://github.com/jobovy/mwdust), [pygaia](https://github.com/agabrown/PyGaia) and others.  
  
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

NOTE: Installation has not been tested on MacOS or Windows systems. Some troubleshooting may be required. The AMUSE package may seem particularly problematic, see  [https://amuse.readthedocs.io/en/latest/install/howto-install-AMUSE.html](https://amuse.readthedocs.io/en/latest/install/howto-install-AMUSE.html)  
  
## Documentation  
You can access every method's docstring by using the help() function in python.  
  
## Workflow  
 1) Define an ejection model, i. e. the ejection mechanisms and associated assumptions. By default only stars ejected up to 100 Myr in the past are generated. This can be changed in the arguments. Arguments can also change the default initial mass function slope as well as the stellar binary mass ratio and log-period distribution slopes, see documentation.    
	 ```python
	 import speedystar
	ejmodel = speedystar.eject.Hills()
	 ```
2)  Create an ejection sample. Masses, velocities, ages, flight times, luminosities, radii and evolutionary stages are assigned/computed for each ejected star and become attributes to `mysample`:
	```python
	mysample = speedystar.starsample(ejmodel)
	```
  
3. Define a Galactic potential and propagate the fast star sample through the Galaxy. Equatorial (position, proper motion, radial velocity, heliocentric distance, parallax) and Galactocentric Cartesian (x, y, z, v<sub>x</sub>, v<sub>y</sub>, v<sub>z</sub>) are computed and become attributes to `mysample`.  Default orbital integration timestep is 0.1 Myr.  
**Note** `astropy-units` *must* be set to `True` in `~/.galpyrc` and the assumed potential must be either defined in physical units or 'physicalized' with `.turn_physical_on()`, see [galpy](https://docs.galpy.org/en/v1.8.0/getting_started.html) explanation.
	```python
	from galpy.potential.mwpotentials import McMillan17
	mysample.propagate(potential=McMillan17)
	```
  
5. Obtain mock observations of each ejected star. Apparent magnitudes in the Johnson-Cousins V, I<sub>c</sub> and Gaia G, G<sub>BP</sub>, G<sub>RP</sub> and G<sub>RVS</sub> bands are computed by default and become attributes to `mysample`. Optionally, magnitudes in other photometric systems can be computed as well, see documentation. Computing apparent magnitudes requires a `DustMap` object (see [mwdust](https://github.com/jobovy/mwdust) or the `speedystar.starsample.fetch_dust()` docstring). _Gaia_ DR4 astrometric and  radial velocity errors are computed by default as well.
  
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
 
## Example

`myexample.py` shows the basic workflow of generating a mock HVS sample, following more or less the steps outlined above.

The class EjectionModel within `speedystar.eject` is the basic structure every ejection model class should be based on. Custom ejection models should be subclasses and follow the same structure.  
  
  ## Tips, Tricks and Troubleshooting

- See the note above regarding `.galpyrc` and implemented `galpy` potentials
  - Note as well that the distance from the Sun to the Galactic Centre and the circular velocity of the Galaxy at the Solar position are also set in `.galpyrc`.
- A lot of speedup can be gained by calling `speedystar.subsample` immediately after creating the ejection sample. This allows you to not waste time propagating or performing mock observations on stars which are not interesting for your science case. For example, if you are certain that only stars more massive than 1 M<sub>&#9737;</sub> will  be detectable by your survey or instrument of interest, you can call `.subsample()` before `.propagate()` like so:
	```python
	idx = (mysample.m >= 1*u.Msun)
	mysample.subsample(np.where(idx)[0])
	```
- Selecting fast stars detectable by modern-day telescopes/surveys (e.g. _Gaia_) often means selecting only the rarest, brightest stars in the sample. Final samples may therefore be quite small and results will be stochastic. In such cases we recommend averaging results over many iterations of ejections+propagation+observation.
- Recall that `galpy` uses a left-handed Galactocentric coordinate system, meaning the Sun is located on the _positive_ x axis, not negative. This is important if, e.g. you are dealing with `astropy` coordinates as well, which places the Sun on the negative x axis. The best way to avoid this mix-up is to use _only_ `galpy` _or_ `astropy` coordinates.
- `python` may run out of available memory if dealing with large samples, particularly during `speedystar.photometry()`. This is still being debugged, however, the best way to avoid this at present is to eject, propagate and save a sample with one .py script, and load in the sample and perform `.photometry()` in  a separate script.
- Exercise caution when allowing stars with low ejection velocities (v<sub>0</sub> &#8818; 200 km/s) to be propagated. Since stars are ejected directly radially away from the Galactic Centre, slow-ejected stars will quickly return towards Sgr A* on _extremely_ eccentric orbits. Fully integrating these orbits can incur substantial energy error and may make `galpy` hang indefinitely.

## Citation  
If you use `speedystar`, please cite [Contigiani et al. 2018](https://ui.adsabs.harvard.edu/abs/2019MNRAS.487.4025C/abstract) and [Evans et al. 2022](https://ui.adsabs.harvard.edu/abs/2022arXiv220514777E/abstract). If discussing the `speedystar.eject.Hills` implementation specifically, please also cite [Rossi et al. 2017](https://ui.adsabs.harvard.edu/abs/2017MNRAS.467.1844R/abstract) and [Marchetti et al. 2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.4697M/abstract).
 
This package makes heavy use of other python packages both widely-used and obscure. When relevant, please credit other works or packages as well: 
- [astropy](https://www.astropy.org/) for units, I/O and conversions
- [scipy](https://scipy.org/) for under-the-hood math-y stuff
- [galpy](https://docs.galpy.org/en/v1.8.0/) for orbital integration
- [AMUSE](https://www.amusecode.org/) for stellar evolution
- [mwdust](https://github.com/jobovy/mwdust) for Galactic dust maps
- [pygaia](https://github.com/agabrown/PyGaia) for *Gaia* astrometric/spectroscopic errors
- [selectionfunctions](https://github.com/gaiaverse/selectionfunctions) for *Gaia* spectroscopic selection functions
- [scanninglaw](https://github.com/gaiaverse/scanninglaw) for *Gaia* astrometric spread function
- [imf](https://github.com/keflavich/imf) for initial mass function utilities

### Development & Bug Reports 

Development of `speedystar` takes place on GitHub, at [https://github.com/speedystar](https://github.com/speedystar). Bug reports, feature requests, or other issues can be filed there or via email to evans@strw.leidenuniv.nl. Contributions to the software are welcome.

Please see TODO.txt for upcoming and planned future features.

### Authors
- Fraser Evans (evans@strw.leidenuniv.nl)
- Based on code base originally developed by Omar Contigiani
- Significant contributions from Tommaso Marchetti
- Additional contributions from Josephine Baggen, Sanne Bloot, Amber Remmelzwaal
- Thanks to Niccolò Veronesi and Claudia Dai for setup & installation debugging 
