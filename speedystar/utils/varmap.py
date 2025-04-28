import h5py
import pandas as pd
import healpy as hp
import numpy as np
import xarray as xr
from scipy.special import expit, logit

from gaiaunlimited import fetch_utils, utils

__all__ = [
    "SelectionFunctionBase",
    "DR3RVSSelectionFunction",
    "EDR3RVSSelectionFunction",
]


def _validate_ds(ds):
    """Validate if xarray.Dataset contains the expected selection function data."""
    if not isinstance(ds, xr.Dataset):
        raise ValueError("ds must be an xarray.Dataset.")
    #for required_variable in ["logitq"]:
    #    if required_variable not in ds:
    #        raise ValueError(f"missing required variable {required_variable}.")
    diff = set(ds["logitq"].dims) - set(["g", "c", "ipix"])
    if diff:
        raise ValueError(f"Unrecognized dims of probability array: {diff}")

# TODO: spec out base class and op for combining SFs
class SelectionFunctionBase:
    """Base class for Gaia selection functions.
    Selection function is defined as the selection probability as a function of
    Gaia G magnitude and healpix location, and optionally also on G-G_PR color.
    We use xarray.Dataset as the main data structure to contain this
    multi-dimensional map of probabilities. This Dataset instance should be
    attach as `.ds` and have the following schema:
        - must contain data variable `p` and `logitp` for selection probability
          and logit of that selection probability.
        - must have coodinates
            - ipix for healpix id in int64 dtype
            - g for Gaia G magnitude
            - c for Gaia G - G_RP color
    It is assumed that ipix is the full index array for the given healpix order
    with no missing index.
    """

    def __init__(self, ds):
        #_validate_ds(ds)
        self.ds = ds

    @property
    def order(self):
        """Order of the HEALPix."""
        return hp.npix2order(self.ds["ipix"].size)

    @property
    def nside(self):
        """Nside of the HEALPix."""
        return hp.npix2nside(self.ds["ipix"].size)

    @property
    def factors(self):
        """Variables other than HEALPix id that define the selection function."""
        return set(self.ds["logitq"].dims) - set({"ipix"})

    def __mul__(self, other):
        # if not isinstance(other, SelectionFunctionBase):
        #     raise TypeError()
        # TODO
        raise NotImplementedError()

    @classmethod
    def from_conditions(cls, conditions):
        # potential idea
        # TODO: query for conditions and make Dataset from result
        raise NotImplementedError()

    def query(self, coords, **kwargs):
        """Query the selection function at the given coordinates.
        Args:
            coords: sky coordinates as an astropy coordinates instance.
        Other factors that determine this selection function should be given
        as keyword arguments of the same shape as coords.
        Returns:
            np.array: array of internal selection probabilities.
        """
        # NOTE: make input atleast_1d for .interp keyword consistency.
        ipix = utils.coord2healpix(coords, "icrs", self.nside, nest=True)
        ipix = xr.DataArray(np.atleast_1d(ipix))
        d = {}
        for k in self.factors:
            if k not in kwargs:
                raise ValueError(f"{k} values are missing.")
            d[k] = xr.DataArray(np.atleast_1d(kwargs[k]))
        d["method"] = "nearest"
        d["kwargs"] = dict(fill_value=None)  # extrapolates
        out = self.ds["logitp"].interp(ipix=ipix, **d).to_numpy()
        if len(coords.shape) == 0:
            out = out.squeeze()
        return expit(out)

class DR3RVSSelectionFunctionVar(SelectionFunctionBase, fetch_utils.DownloadMixin):
    """Internal selection function for the RVS sample in DR3.
    This function gives the probability
        P(has RV | has G and G_RP)
    as a function of G magnitude and G-RP color.
    """

    datafiles = {
        "dr3-rvs-nk.h5": "https://dataverse.harvard.edu/api/access/datafile/6424746"
    }

    def __init__(self):
        with h5py.File(self._get_data("dr3-rvs-nk.h5")) as f:
            df = pd.DataFrame.from_records(f["data"][()])
            df['q'] = (df['k'] + 1) * (df['n'] - df['k'] + 1) / ( (df['n']+3)*(df['n']+2)*(df['n']+2))
            df["logitq"] = logit(df["q"])
            dset_dr3 = xr.Dataset.from_dataframe(df.set_index(["ipix", "i_g", "i_c"]))
            gcenters, ccenters = f["g_mid"][()], f["c_mid"][()]
            dset_dr3 = dset_dr3.assign_coords(i_g=gcenters, i_c=ccenters)
            dset_dr3 = dset_dr3.rename({"i_g": "g", "i_c": "c"})
            super().__init__(dset_dr3)

    def queryvar(self, coords, **kwargs):
        """Query the selection function at the given coordinates.
        Args:
            coords: sky coordinates as an astropy coordinates instance.
        Other factors that determine this selection function should be given
        as keyword arguments of the same shape as coords.
        Returns:
            np.array: array of internal selection probabilities.
        """
        # NOTE: make input atleast_1d for .interp keyword consistency.
        ipix = utils.coord2healpix(coords, "icrs", self.nside, nest=True)
        ipix = xr.DataArray(np.atleast_1d(ipix))
        d = {}
        for k in self.factors:
            if k not in kwargs:
                raise ValueError(f"{k} values are missing.")
            d[k] = xr.DataArray(np.atleast_1d(kwargs[k]))
        d["method"] = "nearest"
        d["kwargs"] = dict(fill_value=None)  # extrapolates
        out = self.ds["logitq"].interp(ipix=ipix, **d).to_numpy()
        if len(coords.shape) == 0:
            out = out.squeeze()
        return expit(out)

    def queryk(self, coords, **kwargs):
        """Query the selection function at the given coordinates.
        Args:
            coords: sky coordinates as an astropy coordinates instance.
        Other factors that determine this selection function should be given
        as keyword arguments of the same shape as coords.
        Returns:
            np.array: array of internal selection probabilities.
        """
        # NOTE: make input atleast_1d for .interp keyword consistency.
        ipix = utils.coord2healpix(coords, "icrs", self.nside, nest=True)
        ipix = xr.DataArray(np.atleast_1d(ipix))
        d = {}
        for k in self.factors:
            if k not in kwargs:
                raise ValueError(f"{k} values are missing.")
            d[k] = xr.DataArray(np.atleast_1d(kwargs[k]))
        d["method"] = "nearest"
        d["kwargs"] = dict(fill_value=None)  # extrapolates
        out = self.ds["k"].interp(ipix=ipix, **d).to_numpy()
        if len(coords.shape) == 0:
            out = out.squeeze()
        return out

    def queryn(self, coords, **kwargs):
        """Query the selection function at the given coordinates.
        Args:
            coords: sky coordinates as an astropy coordinates instance.
        Other factors that determine this selection function should be given
        as keyword arguments of the same shape as coords.
        Returns:
            np.array: array of internal selection probabilities.
        """
        # NOTE: make input atleast_1d for .interp keyword consistency.
        ipix = utils.coord2healpix(coords, "icrs", self.nside, nest=True)
        ipix = xr.DataArray(np.atleast_1d(ipix))
        d = {}
        for k in self.factors:
            if k not in kwargs:
                raise ValueError(f"{k} values are missing.")
            d[k] = xr.DataArray(np.atleast_1d(kwargs[k]))
        d["method"] = "nearest"
        d["kwargs"] = dict(fill_value=None)  # extrapolates
        out = self.ds["n"].interp(ipix=ipix, **d).to_numpy()
        if len(coords.shape) == 0:
            out = out.squeeze()
        return out

# Selection functions ported from gaiaverse's work ----------------
class EDR3RVSSelectionFunctionVar(SelectionFunctionBase, fetch_utils.DownloadMixin):
    """Internal selection function for the RVS sample in EDR3.
    This has been ported from the selectionfunctions by Gaiaverse team.
    NOTE: The definition of the RVS sample is not the same as DR3RVSSelectionFunction.
    """

    __bibtex__ = """
@ARTICLE{2022MNRAS.509.6205E,
       author = {{Everall}, Andrew and {Boubert}, Douglas},
        title = "{Completeness of the Gaia verse - V. Astrometry and radial velocity sample selection functions in Gaia EDR3}",
      journal = {\mnras},
     keywords = {methods: data analysis, methods: statistical, stars: statistics, Galaxy: kinematics and dynamics, Galaxy: stellar content, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2022,
        month = feb,
       volume = {509},
       number = {4},
        pages = {6205-6224},
          doi = {10.1093/mnras/stab3262},
archivePrefix = {arXiv},
       eprint = {2111.04127},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.6205E},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
"""

    datafiles = {
        "rvs_cogv.h5": "https://dataverse.harvard.edu/api/access/datafile/5203267"
    }

    # frame = ''

    def __init__(self):
        with h5py.File(self._get_data("rvs_cogv.h5")) as f:
            # NOTE: HEAL pixelisation is in ICRS in these files!
            # x is the logit probability evaluated at each (mag, color, ipix)
            x = f["x"][()]  # dims: (n_mag_bins, n_color_bins, n_healpix_order6)
            # these are model parameters
            # b = f['b'][()]
            # z = f['z'][()]
            attrs = dict(f["x"].attrs.items())
            n_gbins, n_cbins, n_pix = x.shape
            gbins = np.linspace(*attrs["Mlim"], n_gbins + 1)
            cbins = np.linspace(*attrs["Clim"], n_cbins + 1)
            edges2centers = lambda x: (x[1:] + x[:-1]) * 0.5
            gcenters, ccenters = edges2centers(gbins), edges2centers(cbins)
            ds = xr.Dataset(
                data_vars=dict(logitp=(["g", "c", "ipix"], x)),
                coords=dict(g=gcenters, c=ccenters),
            )
        super().__init__(ds)

    def queryvar(self, coords, **kwargs):
        """Query the selection function at the given coordinates.
        Args:
            coords: sky coordinates as an astropy coordinates instance.
        Other factors that determine this selection function should be given
        as keyword arguments of the same shape as coords.
        Returns:
            np.array: array of internal selection probabilities.
        """
        # NOTE: make input atleast_1d for .interp keyword consistency.
        ipix = utils.coord2healpix(coords, "icrs", self.nside, nest=True)
        ipix = xr.DataArray(np.atleast_1d(ipix))
        d = {}
        for k in self.factors:
            if k not in kwargs:
                raise ValueError(f"{k} values are missing.")
            d[k] = xr.DataArray(np.atleast_1d(kwargs[k]))
        d["method"] = "nearest"
        d["kwargs"] = dict(fill_value=None)  # extrapolates
        out = self.ds["logitq"].interp(ipix=ipix, **d).to_numpy()
        if len(coords.shape) == 0:
            out = out.squeeze()
        return expit(out)

    def queryk(self, coords, **kwargs):
        """Query the selection function at the given coordinates.
        Args:
            coords: sky coordinates as an astropy coordinates instance.
        Other factors that determine this selection function should be given
        as keyword arguments of the same shape as coords.
        Returns:
            np.array: array of internal selection probabilities.
        """
        # NOTE: make input atleast_1d for .interp keyword consistency.
        ipix = utils.coord2healpix(coords, "icrs", self.nside, nest=True)
        ipix = xr.DataArray(np.atleast_1d(ipix))
        d = {}
        for k in self.factors:
            if k not in kwargs:
                raise ValueError(f"{k} values are missing.")
            d[k] = xr.DataArray(np.atleast_1d(kwargs[k]))
        d["method"] = "nearest"
        d["kwargs"] = dict(fill_value=None)  # extrapolates
        out = self.ds["k"].interp(ipix=ipix, **d).to_numpy()
        if len(coords.shape) == 0:
            out = out.squeeze()
        return out

    def queryn(self, coords, **kwargs):
        """Query the selection function at the given coordinates.
        Args:
            coords: sky coordinates as an astropy coordinates instance.
        Other factors that determine this selection function should be given
        as keyword arguments of the same shape as coords.
        Returns:
            np.array: array of internal selection probabilities.
        """
        # NOTE: make input atleast_1d for .interp keyword consistency.
        ipix = utils.coord2healpix(coords, "icrs", self.nside, nest=True)
        ipix = xr.DataArray(np.atleast_1d(ipix))
        d = {}
        for k in self.factors:
            if k not in kwargs:
                raise ValueError(f"{k} values are missing.")
            d[k] = xr.DataArray(np.atleast_1d(kwargs[k]))
        d["method"] = "nearest"
        d["kwargs"] = dict(fill_value=None)  # extrapolates
        
        out = self.ds["n"].interp(ipix=ipix, **d).to_numpy()
        if len(coords.shape) == 0:
            out = out.squeeze()
        return out
