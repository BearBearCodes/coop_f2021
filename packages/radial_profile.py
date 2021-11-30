"""
radial_profile.py

Radial profiles of 2D data.

Isaac Cheng - October 2021
"""

import copy

import astropy.coordinates as coord
import astropy.units as u
import numpy as np

# My packages
import radial_profile_utils as rpu

# TODO: make a function to find the ellipse enclosing x% of the data


class RadialProfile:
    """
    For radial profiles of 2D data.

    Methods: (see individual docstrings for more information)
      calc_radial_profile()
        Calculates the radial profile. Main workhorse of this class
      calc_area()
        Calculates the area of the radial profile annuli. Must run calc_radial_profile()
        first
      correct_for_i()
        Corrects the radial profile data/noise for inclination. Must run
        calc_radial_profile() first
      copy()
        Convenience function so the user does not have to import the copy module
      directional_radial_profile()
        Calculates the radial profile along a given direction. Must run
        calc_radial_profile() first

    Attributes: (data, center, i, pa, noise, avg_data, avg_noise, avg_data_err,
                avg_noise_err, std_data, std_noise, data_area_mask, noise_area_mask,
                radii, annuli, a_ins, a_outs, b_ins, b_outs, rp_options, drp_results)
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
      data_area_mask, noise_area_mask :: 1D arrays or None
        1D arrays containing the included regions of the data and noise arrays per
        ellipse/annulus. 1 means the pixel is wholly included while 0 and NaNs mean the
        pixel is wholly excluded. Numbers between 0 and 1 indicate the relative
        contribution of that pixel (e.g., 0.5 means the pixel is "half-included"). If
        noise is None, then noise_area is also None
      radii :: 1D array
        The radii of the ellipses/annuli. The radii are defined to be the artihmetic means
        of the ellipses/annuli's circularized radii. If i >= i_threshold, then the radii
        are the midpoints of the rectangles/rectangular annuli along the galaxy's major
        axis
      a_ins, a_outs :: 1D arrays
        The inner and outer semi-major axes of the ellipses/annuli in pixel units. N.B. an
        ellipse's inner semi-major axis length is 0. If i >= i_threshold, these are the
        heights of the rectangles in the galaxy's minor axis direction
      b_ins, b_outs :: 1D arrays
        The inner and outer semi-minor axes of the ellipses/annuli in pixel units. N.B. an
        ellipse's inner semi-minor axis length is 0. If >= i_threshold, these are the
        widths of the rectangles in the galaxy's major axis direction
      rp_options :: dict
        The parameters used to generate the radial profile (e.g., n_annuli, min_width,
        etc.). Irrelevant attributes are set to None
      drp_results :: dict
        The directional_radial_profile() results with keys: "radii", "avg_data",
        "avg_noise", "avg_data_err", "avg_noise_err", "std_data", "std_noise",
        "split_masks"
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
        if not isinstance(i, (int, float, np.int64, np.float64)):
            raise ValueError("i (the inclination) must be a float or int")
        if not isinstance(pa, (int, float, np.int64, np.float64)):
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
        self.data_area_masks = None
        self.noise_area_masks = None
        self.radii = None
        self.annuli = None
        self.a_ins = None
        self.a_outs = None
        self.b_ins = None
        self.b_outs = None
        self.rp_options = None
        # Attributes to be set by directional_radial_profile()
        self.drp_results = None

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
        high_i_height=None,
        bootstrap_errs=False,
        n_bootstraps=100,
        n_samples=None,
        bootstrap_seed=None,
    ):
        """
        Calculates the radial profile of a galaxy from radio or other (e.g., optical)
        data. Data are azimuthally averaged (median or arithmetic mean) in ellipses/annuli
        and the radii are defined to be the artihmetic means of the ellipses/annuli's
        circularized radii. If it is a high-inclination galaxy, we fit rectangles to the
        data instead of ellipses/annuli. In this case, each radius is defined to be the
        midpoint of the rectangle/rectangular cutout along the galaxy's major axis.

        Note that these radial profile results (i.e., avg_data, avg_noise, avg_data_err,
        avg_noise_err, std_data, std_noise) are not corrected for inclination.

        Finally, if i >= i_threshold, the radial profile is calculated using rectangles
        instead of ellipses/annuli. All the keywords, parameters, and return values remain
        the same, except mentally replace "ellipse/annulus" with "rectangle/rectangular
        annulus". Regarding return values, a_out will represent the heights of the
        rectangles (perpendicular to the galaxy's major axis) while b_out will represent
        the widths (along the galaxy's major axis). b_in will represent the start of the
        rectangular annulus (looks more like a rectangular sandwich) while a_in will be
        equal to a_out except for the first element, which will be zero (i.e., an ordinary
        rectangle). N.B. the i_threshold and high_i_height parameters. Also see the
        documentation for fit_rectangles().

        Also, will set irrelevant parameters to None.

        Parameters:
          i_threshold :: float (optional)
            If i >= i_threshold, use rectangles/rectangular annuli that have a thickness
            of min_width (e.g., the radio beam width) aligned with the major axis instead
            of ellipses/elliptical annuli. The initial rectangle will be centred on the
            galactic centre (provided via the center parameter). The next "rectangular
            sandwich" region will be composed of two rectangles identitcal to the first
            appended on the sides of the initial rectangle along the major axis of the
            galaxy. In other words, the heights of all the rectangles will be equal and
            the widths will be 1, 3, 5, ... times min_width. N.B. for widths greater than
            1 min_width, only the outer 2 rectangles will be used for the average
            calculation. Lastly, here is how to (mentally) map the variable names
            n_annuli = max number of rectangles (e.g., n_annuli=3 => rectangles of widths
            1, 3, 5), a = heights, b = widths. Also see the high_i_height parameter and
            the documentation for fit_rectangles() in radial_profile_utils.py
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
          high_i_height :: "min_width", float, or `astropy.units.quantity.Quantity` object
                           or None (optional)
            If i >= i_threshold, this is the size of the rectangles/rectangular annuli
            along the galaxy's minor axis. If "min_width", set the heights of the
            rectangles/annuli equal to min_width. If a float, it should be in pixel units.
            If an astropy Quantity, either header or wcs must also be provided. If None,
            automatically extend the rectangle/annulus to the edges of the image (at which
            point it will likely no longer be a perfect rectangle. N.B. the include_bad
            parameter)
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
            "high_i_height": high_i_height,
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
            new_RadialProfile.data_area_masks,
            new_RadialProfile.noise_area_masks,
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
        # pylint: disable=not-an-iterable
        if self.avg_data is None:
            raise ValueError("Radial profile must be generated first")
        data_areas = []
        for data_area_mask in self.data_area_masks:
            data_areas.append(np.nansum(data_area_mask))
        data_areas = np.asarray(data_areas)
        if self.noise_area_masks is not None:
            noise_areas = []
            for noise_area_mask in self.noise_area_masks:
                noise_areas.append(np.nansum(noise_area_mask))
            noise_areas = np.asarray(noise_areas)
            if not np.all(data_areas == noise_areas):
                print(
                    "WARNING: not all data areas equal noise areas. Please double check "
                    + "inputs and notify me (Isaac Cheng) with a minimal working example"
                )
        else:
            noise_areas = None
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
            avg_noise, and if applicable, the directional radial profile's
            avg_data and avg_noise) corrected for inclination
        """
        # pylint: disable=unsubscriptable-object
        #
        # Perfectly valid to correct data before radial profile fitting
        #
        # if self.avg_data is None:
        #    raise ValueError("Radial profile must be generated first")
        #
        new_RadialProfile = copy.deepcopy(self)
        #
        # N.B. constants do not change standard deviation or uncertainties on averages,
        # hence we do not need to correct them
        #
        new_RadialProfile.data = rpu.correct_for_i(
            self.data, self.i, self.rp_options["i_threshold"], i_replacement
        )
        if self.avg_data is not None:
            new_RadialProfile.avg_data = rpu.correct_for_i(
                self.avg_data, self.i, self.rp_options["i_threshold"], i_replacement
            )
        if self.noise is not None:
            new_RadialProfile.noise = rpu.correct_for_i(
                self.noise, self.i, self.rp_options["i_threshold"], i_replacement
            )
            if self.avg_noise is not None:
                new_RadialProfile.avg_noise = rpu.correct_for_i(
                    self.avg_noise, self.i, self.rp_options["i_threshold"], i_replacement
                )
        if self.drp_results is not None:
            # Correct directional profile results for inclination
            new_RadialProfile.drp_results["avg_data"] = rpu.correct_for_i(
                self.drp_results["avg_data"],
                self.rp_options["i_threshold"],
                i_replacement,
            )
            if self.drp_results["avg_noise"] is not None:
                new_RadialProfile.drp_results["avg_noise"] = rpu.correct_for_i(
                    self.drp_results["avg_noise"],
                    self.rp_options["i_threshold"],
                    i_replacement,
                )
        return new_RadialProfile

    def directional_radial_profile(
        self,
        angle=None,
        include_bad=None,
        func=None,
        debug_plot=False,
        bootstrap_errs=False,
        n_bootstraps=100,
        n_samples=None,
        bootstrap_seed=None,
        results_to_1d=True,
    ):
        """
        Calculates the radial profile along a given direction specified by the angle
        parameter. This is accomplished by splitting the radial profile annulus/rectangle
        perpendicular to the specified angle and averaging these two areas individually.
        That is, the split line is the line perpendicular to the supplied angle. Note that
        the angle is defined to be zero on the +x-axis and increases counter-clockwise
        (i.e., the traditional definition of the angle in mathematics).

        If results_to_1d is False, the results are all either 2D arrays with shapes
        (len(n_annuli), 2) or None (except for split_masks, which is a 4D array: a 2D
        array with 2D entries). The rows (1st index) correspond to the same annulus. The
        columns (2nd index) correspond to the same half (i.e., mask1 or mask2) over all
        the annuli.

        For example, if we want to get the two halves of the first annulus, we would use:
                                    mask1, mask2 = masks[0]
        If we want to get the top half of all the annuli, we would use:
                                    all_masks1 = masks[:, 0]
        This slicing convention applies to all the returned arrays.

        Note that mask1 is the part of the mask above the split line and mask2 is the
        part of the mask below the split line. If the split line is a vertical line (i.e.,
        angle is 0 or 180 degrees), then mask1 is the part of the mask to the right of the
        split line and mask2 is the part of the mask to the left of the split line.

        TODO: finish docstring

        Parameters:
          angle :: float (optional)
            The direction in degrees along which to calculate the radial profile. If None,
            use the position angle of the galaxy as the direction. Note that the angle is
            defined to be zero on the +x-axis and increases counter-clockwise (i.e., the
            traditional definition of the angle in mathematics)
          results_to_1d :: bool (optional)
            If True, return the results as 1D arrays, each ordered from the most negative
            radius to the most positive radius. Note that the "split_masks" entry of the
            results will be 3D since it is 1D along the first axis, but each entry along
            this first axis is a 2D array

        Returns: new_RadialProfile
          new_RadialProfile :: `RadialProfile` object
            The `RadialProfile` object with directional radial profile results stored in
            the "drp_results" attribute. The "drp_results" attribute is a dictionary with
            keys: "radii", "avg_data", "avg_noise", "avg_data_err", "avg_noise_err",
            "std_data", "std_noise", "split_masks"
        """
        if self.avg_data is None:
            raise ValueError("Radial profile must be generated first")
        new_RadialProfile = copy.deepcopy(self)
        #
        # Determine the angle of the slice (perpendicular to the direction of the desired
        # radial profile)
        #
        split_angle = self.pa if angle is None else (angle + 90) % 360.0
        #
        # Other parameters
        #
        include_bad = (
            self.rp_options["include_bad"] if include_bad is None else include_bad
        )
        func = self.rp_options["func"] if func is None else func
        #
        # Generate directional radial profile values
        #
        (
            avg_signals,
            avg_noises,
            avg_signal_errs,
            avg_noise_errs,
            std_signals,
            std_noises,
            split_masks,
        ) = rpu.calc_avg_sn_split_mask_multi(
            self.data,
            self.data_area_masks,
            self.center,
            split_angle,
            noise=self.noise,
            header=self.rp_options["header"],
            wcs=self.rp_options["wcs"],
            include_bad=include_bad,
            func=func,
            plot=debug_plot,
            bootstrap_errs=bootstrap_errs,
            _n_bootstraps=n_bootstraps,
            _n_samples=n_samples,
            _seed=bootstrap_seed,
        )
        #
        # Fix radius for high-inclination galaxies (so radius is still centre of rectangles)
        #
        if self.rp_options["i_threshold"] is not None and self.i >= self.rp_options["i_threshold"]:
            tmp_b_ins = self.b_ins
            tmp_b_ins[0] = 0
            tmp_radii = rpu.calc_radius(tmp_b_ins, self.b_outs, is_rectangle=True)
        else:
            tmp_radii = self.radii
        #
        # Process results and save to drp_results attribute
        #
        if results_to_1d:
            # Reshape to 1D arrays for ease of use when plotting
            radii = np.hstack((-tmp_radii[::-1], tmp_radii))
            avg_signals = np.hstack((avg_signals[:, 1][::-1], avg_signals[:, 0]))
            if avg_noises is not None:
                avg_noises = np.hstack((avg_noises[:, 1][::-1], avg_noises[:, 0]))
            if avg_signal_errs is not None:
                avg_signal_errs = np.hstack(
                    (avg_signal_errs[:, 1][::-1], avg_signal_errs[:, 0])
                )
            if avg_noise_errs is not None:
                avg_noise_errs = np.hstack(
                    (avg_noise_errs[:, 1][::-1], avg_noise_errs[:, 0])
                )
            std_signals = np.hstack((std_signals[:, 1][::-1], std_signals[:, 0]))
            if std_noises is not None:
                std_noises = np.hstack((std_noises[:, 1][::-1], std_noises[:, 0]))
            split_masks = np.concatenate(
                (split_masks[:, 1, :, :][::-1], split_masks[:, 0, :, :]), axis=0
            )
        else:
            radii = np.column_stack((tmp_radii, -tmp_radii))
        new_RadialProfile.drp_results = {
            "radii": radii,
            "avg_data": avg_signals,
            "avg_noise": avg_noises,
            "avg_data_err": avg_signal_errs,
            "avg_noise_err": avg_noise_errs,
            "std_data": std_signals,
            "std_noise": std_noises,
            "split_masks": split_masks,
        }
        return new_RadialProfile
