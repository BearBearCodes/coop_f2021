"""
radial_profile.py

Radial profiles of 2D data.

Isaac Cheng - October 2021
"""

import copy

import astropy.coordinates as coord
import astropy.units as u
import numpy as np

import radial_profile_utils as rpu

# TODO: add support for highly inclined galaxies (beyond i_threshold)
# TODO: make a function to find the ellipse enclosing x% of the data

class RadialProfile:
    """
    For radial profiles of 2D data.

    Methods: (see individual docstrings for more information)
      calc_radial_profile()
        Calculates the radial profile. Main workhorse of this class
      calc_area()
        Calculates the area of the radial profile annuli
      correct_for_i()
        Corrects the radial profile data/noise for inclination
      copy()
        Convenience function so the user does not have to import the copy module

    Attributes:
      data :: 2D array
        The data used for generating a radial profile. If using snr_cutoff, the data
        should be background-subtracted
      center :: 2-tuple of ints/floats or `astropy.coordinates.SkyCoord` object
        The centre of the galaxy. If not a SkyCoord object, the center should be in
        pixel units. If this is a SkyCoord object, the wcs or header must also be
        provided
      i :: float
        The inclination of the galaxy. The cosine of the inclination is defined as the
        ratio of the semi-minor axis, b, to the semi-major axis, a. That is,
        cos(i) := b/a
      pa :: float
        The position angle of the galaxy. The position angle is defined as the angle
        starting from north and increasing toward the east counter-clockwise
      noise :: 2D array
        The noise (uncertainty) array associated with the data. Must have the same
        shape as the data array. If using snr_cutoff, this parameter is required and
        this array should be background-subtracted
      avg_data :: 1D array
        The average (i.e., median or arithmetic mean) of the data in each ellipse/annulus.
        Not corrected for inclination
      avg_noise :: 1D array
        If noise is not None, this is the corresponding average uncertainty in each
        ellipse/annulus. If func is "median", these are the median uncertainties. If
        func is "mean", these are the sums of the noise values in each annulus in
        quadrature, divided by the area of the annulus and accounting for the
        inclusion/exclusion of NaN/inf values. Not corrected for inclination.
        If noise is None, then this is None
      avg_data_err, avg_noise_err :: 1D arrays or None
        If bootstrap_errs is True, these are the uncertainties in avg_data and avg_noise
        estimated by bootstrapping. If bootstrap_errs is False, these are None
      std_signal, std_noise :: 1D arrays or None
        The standard deviation of the signal and noise in each ellipse/annulus. If noise
        is None, then std_noise is None. N.B. if there are many "bad" points in an
        ellipse/annulus (i.e., NaNs or infs) and include_bad is True, then this may not be
        an accurate representation of the standard deviation in that given ellipse/annulus
      radii :: 1D array
        The radii of the ellipses/annuli. The radii are defined to be the artihmetic means
        of the ellipses/annuli's circularized radii
      a_ins, a_outs :: 1D arrays
        The inner and outer semi-major axes of the ellipses/annuli. N.B. an ellipse's
        inner semi-major axis length is 0
      b_ins, b_outs :: 1D arrays
        The inner and outer semi-minor axes of the ellipses/annuli. N.B. an ellipse's
        inner semi-minor axis length is 0
      rp_options :: dict
        The parameters used to generate the radial profile (e.g., n_annuli, min_width,
        etc.). Irrelevant attributes are set to None
    """

    def __init__(self, data, center, i, pa, noise=None):
        """
        Initialize a radial profile object.

        Parameters:
          data :: 2D array
            The data used for generating a radial profile. If using snr_cutoff, the data
            should be background-subtracted
          center :: 2-tuple of ints/floats or `astropy.coordinates.SkyCoord` object
            The centre of the galaxy. If not a SkyCoord object, the center should be in
            pixel units. If this is a SkyCoord object, the wcs or header must also be
            provided
          i :: float
            The inclination of the galaxy. The cosine of the inclination is defined as the
            ratio of the semi-minor axis, b, to the semi-major axis, a. That is,
            cos(i) := b/a
          pa :: float
            The position angle of the galaxy. The position angle is defined as the angle
            starting from north and increasing toward the east counter-clockwise
          noise :: 2D array (optional)
            The noise (uncertainty) array associated with the data. Must have the same
            shape as the data array. If using snr_cutoff, this parameter is required and
            this array should be background-subtracted

        Returns: None
        """
        # Check inputs
        if data.ndim != 2:
            raise ValueError("data must be a 2D array")
        if not isinstance(center, (list, tuple, np.ndarray, coord.SkyCoord)):
            raise ValueError(
                "center must be a list, tuple, or array in pixel coordinates "
                + "OR an astropy.coordinates.SkyCoord object"
            )
        if not isinstance(center, coord.SkyCoord) and np.shape(center) != (2,):
            raise ValueError("center must be 1D with 2 elements")
        if not isinstance(i, (int, float)):
            raise ValueError("i (the inclination) must be a float or int")
        if not isinstance(pa, (int, float)):
            raise ValueError("pa (the position angle) must be a float or int")
        if noise is not None and np.shape(noise) != np.shape(data):
            raise ValueError("noise must be an array of the same shape as data")
        # Assign inputs
        self.data = data
        self.center = center
        self.i = i
        self.pa = pa
        self.noise = noise
        # Attributes to be set by calc_radial_profile()
        self.avg_data = None
        self.avg_noise = None
        self.avg_data_err = None
        self.avg_noise_err = None
        self.std_data = None
        self.std_noise = None
        self.radii = None
        self.annuli = None
        self.a_ins = None
        self.a_outs = None
        self.b_ins = None
        self.b_outs = None
        self.rp_options = None
        # self._generated_rp = False  # bool to track if radial profile has been generated

    def copy(self):
        """
        Convenience function so the user does not have to import the copy module.

        Parameters: None

        Returns: self_copied
          self_copied :: `RadialProfile` object
            A copy of self
        """
        return copy.deepcopy(self)

    def calc_radial_profile(
        self,
        i_threshold=None,
        n_annuli=None,
        snr_cutoff=None,
        max_snr_annuli=50,
        min_width=None,
        min_width_ax="minor",
        header=None,
        wcs=None,
        include_bad=True,
        method="exact",
        func="median",
        debug_plot=False,
        is_radio=True,
        header_min_width_key="IQMAX",
        header_min_width_unit=u.arcsec,
        bootstrap_errs=False,
        n_bootstraps=100,
        n_samples=None,
        bootstrap_seed=None,
    ):
        """
        Convenience function for calculating the radial profile of a galaxy from radio
        or other (e.g., optical) data. Data are azimuthally averaged (median or
        arithmetic mean) in ellipses/annuli and the radii are defined to be the
        artihmetic means of the ellipses/annuli's circularized radii.

        Note that these radial profile results (i.e., avg_data, avg_noise,
        avg_data_err, avg_noise_err, std_data, std_noise) are not corrected for
        inclination.

        Also, will set irrelevant parameters to None.

        Parameters:
          i_threshold :: float (optional)
            TODO: implement this
            If i >= i_threshold, use squares/rectangles that have a thickness of min_width
            (e.g., the radio beam width) aligned with the major axis instead of
            ellipses/elliptical annuli. The initial square will be centred on the galactic
            centre (provided via the center parameter). The next "rectangle" region will
            be composed of two squares appended on the sides of the initial square along
            the major axis of the galaxy. In other words, the heights of all the
            rectangles will be min_width and the widths will be 1, 3, 5, ... times
            min_width. N.B. for widths greater than 1 min_width, only the outer 2 squares
            will be used for the average calculation. Lastly, n_annuli = max number of
            rectangles (e.g., n_annuli=3 => rectangles of widths 1, 3, 5), a_in & b_in are
            both zero, and a_out & b_out are the widths and heights of the rectangles.
            ! FIXME: This is super poorly explained.
          n_annuli :: int (optional)
            The number of ellipses/annuli to create. If n_annuli==1, the function will
            generate an ellipse. If n_annuli>1, the function will generate a central
            ellipse surrounded by (n_annuli-1) annuli. N.B. the user should specify
            exactly one parameter: n_annuli or snr_cutoff
          snr_cutoff :: float (optional)
            The signal-to-noise ratio (SNR) cutoff for the ellipses/annuli. If the SNR of
            the central ellipse or surrounding annuli drop below this value, the function
            will stop fitting annuli. Must also pass in the background-subtracted noise
            array. N.B. the user should specify exactly one parameter: n_annuli or
            snr_cutoff
          max_snr_annuli :: int (optional)
            The maximum number of ellipses/annuli to fit if using the snr_cutoff
            parameter. Ignored for n_annuli. Once this value is reached, the function will
            stop generating ellipses/annuli regardless of other parameters
          min_width :: float or `astropy.units.quantity.Quantity` object (optional)
            The minimum width and separation of all ellipses/annuli. If a float, min_width
            should be in pixel units. If an astropy Quantity, the header or wcs must also
            be provided. This width will be the minimum length of the semi-major/minor
            axis of an ellipse or the minimum distance between the inner & outer rings of
            an annulus along the semi-major/minor axes. For example, this is typically the
            beam size of the radio telescope or the size of the worst (largest) PSF of an
            optical telescope.
            If min_width not provided, the FITS header must be provided instead
          min_width_ax :: "minor" or "major" (optional)
            The axis along which the minimum width is defined. If "minor", min_width is
            the minimum width and separation of any ellipse/annulus along the minor axis.
            If "major", min_width is the minimum width and separation of any
            ellipse/annulus along the major axis
          header :: `astropy.io.fits.Header` object (optional)
            The header of the data's FITS file. Required if min_width is not provided. If
            is_radio is True, will set the min_width equal to the radio beam size.
            Otherwise, will set the min_width according to the header_min_width_key and
            header_min_width_unit
          wcs :: `astropy.wcs.WCS` object (optional)
            The WCS object corresponding to the data. Required if center is a SkyCoord
            object and header is not provided
          include_bad :: bool (optional)
            If True, includes NaNs and infs in signal & noise arrays by setting these
            values to zero. If False, exclude NaNs and infs entirely from all calculations
          func :: "median" or "mean" (optional)
            Specifies which "average" function to use for the radial profile: median or
            arithmetic mean
          method :: "exact" or "center" or "subpixel" (optional)
            The `photutils.aperture.aperture_photometry` method used to create the
            ellipses/annuli
          debug_plot :: bool (optional)
            If True, plots intermediate steps of the radial profile averaging calculations
            (i.e., the ellipses/annuli, masked signal, and if calculating the mean, the
            area over which the mean is calculated). Warning: lots of plots will be made
            for each ellipse/annulus!
          is_radio :: bool (optional)
            If True, the inputs correspond to radio data. Otherwise, the inputs are not
            radio data (e.g., optical, IR, etc.). The only difference is if the user does
            not input a minimum width, in which case radio data default to the beam size
            while other data default to the worst image quality specified by the header
            (also see header_min_width_key and header_min_width_unit)
          header_min_width_key :: str (optional)
            The FITS header keyword corresponding to the value you wish to use for the
            min_width. Only relevant if min_width is None and is_radio is False; ignored
            if is_radio is True. Also see header_min_width_unit
          header_min_width_unit :: `astropy.units.quantity.Quantity` object (optional)
            The unit of the min_width parameter from the FITS header (i.e., the unit
            corresponding to header_min_width_key). Only relevant if min_width is None and
            is_radio is False; ignored if is_radio is True. Also see header_min_width_key
          bootstrap_errs :: bool (optional)
            If True, estimate the uncertainty in the radial profile results (i.e.,
            avg_data & avg_noise) using bootstrapping
          n_bootstraps :: int (optional)
            The number of bootstrap iterations to use to estimate errors. Ignored if
            bootstrap_errs is False
          n_samples :: int (optional)
            The number of samples to use in each bootstrap iteration. If None, the number
            of samples per bootstrap iteration is the number of data points enclosed in
            the ellipse/annulus (usually this is what we want). Ignored if bootstrap_errs
            is False
          bootstrap_seed :: int (optional)
            The seed to use for bootstrapping (per ellipse/annulus); does not affect
            global seed. Ignored if bootstrap_errs is False

        Returns: new_RadialProfile
          new_RadialProfile :: `RadialProfile` object
            The new radial profile object that contains the radial profile results
        """
        #
        # Set irrelevant parameters to None
        #
        if n_annuli is not None and snr_cutoff is None:
            max_snr_annuli = None
        if is_radio:
            header_min_width_key, header_min_width_unit = None, None
        if not bootstrap_errs:
            n_bootstraps, n_samples, bootstrap_seed = None, None, None
        #
        # Radial profile parameters/options (exlcuding debug_plot)
        #
        rp_options = {
            "i_threshold": i_threshold,
            "n_annuli": n_annuli,
            "snr_cutoff": snr_cutoff,
            "max_snr_annuli": max_snr_annuli,
            "min_width": min_width,
            "min_width_ax": min_width_ax,
            "header": header,
            "wcs": wcs,
            "include_bad": include_bad,
            "func": func,
            "method": method,
            "is_radio": is_radio,
            "header_min_width_key": header_min_width_key,
            "header_min_width_unit": header_min_width_unit,
            "bootstrap_errs": bootstrap_errs,
            "n_bootstraps": n_bootstraps,
            "n_samples": n_samples,
            "bootstrap_seed": bootstrap_seed,
        }
        #
        # Generate radial profile
        #
        new_RadialProfile = copy.deepcopy(self)
        (
            new_RadialProfile.avg_data,
            new_RadialProfile.avg_noise,
            new_RadialProfile.avg_data_err,
            new_RadialProfile.avg_noise_err,
            new_RadialProfile.std_data,
            new_RadialProfile.std_noise,
            new_RadialProfile.radii,
            new_RadialProfile.annuli,
            new_RadialProfile.a_ins,
            new_RadialProfile.a_outs,
            new_RadialProfile.b_ins,
            new_RadialProfile.b_outs,
        ) = rpu.calc_radial_profile(
            self.data,
            self.center,
            self.i,
            self.pa,
            noise=self.noise,
            debug_plot=debug_plot,
            **rp_options,
        )
        new_RadialProfile.rp_options = rp_options
        # self._generated_rp = True  # radial profile has been created
        return new_RadialProfile

    def calc_area(self):
        """
        Calculates the pixel area of the radial profile annuli. This gives the number of
        pixels that the data/noise are calculated over including the effects from the
        include_bad parameter and any edge effects from the method parameter.

        Parameters: None

        Returns: data_areas, noise_areas
          data_areas :: 1D array of floats
            The areas over which the average data values are calculated
          noise_areas :: 1D array of floats
            If noise is not None, this contains the areas over which the average noise
            values area calculated. If noise is None, this is None
        """

        def _make_area_arr(arr, aper):
            aper_mask, padded_arr = rpu.create_aper_mask(
                arr,
                aper,
                include_bad=self.rp_options["include_bad"],
                method=self.rp_options["method"],
            )
            padded_arr[~np.isnan(padded_arr)] = 1.0
            area_arr = padded_arr * aper_mask
            return area_arr

        if self.avg_data is None:
            raise ValueError("Radial profile must be generated first")
        masked_data, masked_noise = rpu.mask_bad(
            self.data,
            self.rp_options["include_bad"],
            noise=self.noise,
            bad_fill_value=0.0,
        )
        data_areas = []
        # Slightly verbose, but I'm guessing this is faster than checking if noise is None
        # in a loop
        if self.noise is None:
            noise_areas = None
            for annulus in self.annuli:
                data_area = np.nansum(_make_area_arr(masked_data, annulus))
                data_areas.append(data_area)
        else:
            noise_areas = []
            for annulus in self.annuli:
                data_area = np.nansum(_make_area_arr(masked_data, annulus))
                data_areas.append(data_area)
                noise_area = np.nansum(_make_area_arr(masked_noise, annulus))
                noise_areas.append(noise_area)
        data_areas = np.asarray(data_areas)
        noise_areas = np.asarray(noise_areas) if noise_areas is not None else None
        if noise_areas is not None:
            if not np.all(data_areas == noise_areas):
                print(
                    "WARNING: not all data areas equal noise areas. Please double check "
                    + "inputs and notify me (Isaac Cheng) with a minimal working example"
                )
        return data_areas, noise_areas

    def correct_for_i(self, i_replacement=None):
        """
        Corrects for the inclination of the galaxy. Typically used if the data or their
        precursors are affected by optically thick properties of the galaxy (e.g., finding
        the gas surface density from CO).

        The inclination, i, is defined as:
                                        cos(i) := b/a
        where b is the length of the semi-minor axis and a is the length of the semi-major
        axis.

        Parameters:
          i_replacement :: scalar or array of scalars
            The value to replace i when i >= i_threshold. Must be able to broadcast with i

        Returns: new_RadialProfile
          new_RadialProfile :: `RadialProfile` object
            The `RadialProfile` object with quantities (i.e., data, noise, avg_data,
            avg_noise) corrected for inclination
        """
        if self.avg_data is None:
            raise ValueError("Radial profile must be generated first")
        new_RadialProfile = copy.deepcopy(self)
        new_RadialProfile.data = rpu.correct_for_i(
            self.data, self.i, self.rp_options["i_threshold"], i_replacement
        )
        new_RadialProfile.avg_data = rpu.correct_for_i(
            self.avg_data, self.i, self.rp_options["i_threshold"], i_replacement
        )
        if self.noise is not None:
            new_RadialProfile.noise = rpu.correct_for_i(
                self.noise, self.i, self.rp_options["i_threshold"], i_replacement
            )
            new_RadialProfile.avg_noise = rpu.correct_for_i(
                self.avg_noise, self.i, self.rp_options["i_threshold"], i_replacement
            )
        # N.B. constants do not change standard deviation or uncertainties on averages
        return new_RadialProfile
