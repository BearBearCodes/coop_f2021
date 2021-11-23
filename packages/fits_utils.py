"""
fits_utils.py

Utilities for handling .FITS files. Many of these are for convenience.
Includes functions for easy binning of data (both regular and adaptive binning).

Isaac Cheng - October 2021
"""

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pandas as pd
import reproject

# from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


def calc_pc_per_px(imgwcs, dist, dist_err=None):
    """
    Calculates the physical size of each pixel in parsecs per pixel dimension.

    Parameters:
      imgwcs :: `astropy.wcs.wcs.WCS`
        The WCS coordinates of the .fits file
      dist :: `astropy.units.quantity.Quantity` object
        The distance to the object
      dist_err :: `astropy.units.quantity.Quantity` object (optional, default: None)
        The uncertainty in the distance to the object

    Returns: pc_per_px, pc_per_px_err
      pc_per_px :: 1D array
        The spatial resolution of the image in parsecs per pixel (along each axis)
      pc_per_px_err :: 1D array or None
        The uncertainty in the spatial resolution of the image (along each axis)
    """
    arcsec_per_px = (proj_plane_pixel_scales(imgwcs.celestial) * u.deg).to(u.arcsec)
    # Still deciding whether to use arctan for the next line
    # pylint: disable=no-member
    arcsec_per_pc = np.rad2deg(1 * u.pc / dist.to(u.pc) * u.rad).to(u.arcsec)
    pc_per_px = arcsec_per_px / arcsec_per_pc
    #
    if dist_err is not None:
        # Uncertainty transforms linearly
        pc_per_px_err = pc_per_px * dist_err.to(u.pc) / dist.to(u.pc)
        return pc_per_px.value, pc_per_px_err.value
    return pc_per_px.value, None


def cutout(data, center, shape, wcs=None, header=None, mode="trim"):
    """
    Convenience wrapper for astropy's Cutout2D class. The cutout array should be wholly
    contained inside the extent of the data array if using trim="strict".

    Parameters:
      data :: 2D array
        The data from which to extract the cutout
      center :: 2-tuple of int or `astropy.coordinates.SkyCoord`
        The (x, y) center of the cutout
      shape :: 2-tuple of int or `astropy.units.Quantity`
        The (vertical, horizontal) dimensions of the cutout
      wcs :: `astropy.wcs.WCS` (optional)
        The WCS of the data. If not None, will return a copy of the updated WCS for the
        cutout data array. Only one of either wcs or header should be provided, if any.
      header :: `astropy.io.fits.header.Header` (optional)
        The header to the .fits file used to extract WCS data. Only one of either wcs or
        header should be provided, if any.
      mode :: str (optional)
        The Cutout2D mode ("trim", "partial", or "strict"). The cutout array should be
        wholly contained inside the extent of the data array if using trim="strict".

    Returns: data_cut, wcs_cut
      data_cut :: 2D array
        The array cutout from the original data array
      wcs_cut :: `astropy.wcs.WCS` or None
        If WCS initially provided, this is the updated WCS for the cutout data array.
        If WCS initially not provided, this is None.
    """
    if wcs is not None and header is not None:
        raise ValueError("Only provide either wcs or header to prevent ambiguity")
    elif header is not None:
        wcs = WCS(header)
    cutout_obj = Cutout2D(data, center, shape, wcs=wcs, mode=mode)
    data_cut = cutout_obj.data
    wcs_cut = cutout_obj.wcs
    return data_cut, wcs_cut


def cutout_to_target(input_arr, input_wcs, target_arr, target_wcs, mode="trim"):
    """
    Uses Cutout2D to cut out a subsction of input_arr to the extent of target_arr
    (provided via the target_wcs). The extent of the target_arr should be wholly contained
    inside the extent of input_arr if using trim="strict".

    Parameters:
      input_arr :: 2D array
        The array from which the cutout will be extracted
      input_wcs :: `astropy.wcs.WCS`
        The WCS of the input_arr
      target_arr :: 2D array
        The array whose extent you want to copy
      target_wcs :: `astropy.wcs.WCS`
        The WCS of the target_arr
      mode :: str (optional)
        The Cutout2D mode ("trim", "partial", or "strict"). The extent of the target_arr
        should be wholly contained inside the extent of input_arr if using trim="strict".

    Returns: input_arr_cut, input_wcs_cut
      input_data_cut :: 2D array
        The array cut out from the original input_arr
      input_wcs_cut :: `astropy.wcs.WCS` or None
        The updated WCS for the cutout array
    """
    target_bottomleft = target_wcs.pixel_to_world(0, 0)
    target_topright = target_wcs.pixel_to_world(target_arr.shape[1], target_arr.shape[0])
    target_centre = target_wcs.pixel_to_world(
        target_arr.shape[1] / 2, target_arr.shape[0] / 2  # (x, y). This is correct
    )
    # Map the pixels above to their corresponding pixels in the input array
    input_bottomleft = input_wcs.world_to_pixel(target_bottomleft)
    input_topright = input_wcs.world_to_pixel(target_topright)
    # Determine shape of cutout in input array
    cutout_shape = abs(np.subtract(input_topright, input_bottomleft))[::-1]
    return cutout(input_arr, target_centre, cutout_shape, wcs=input_wcs, mode=mode)


def line_profile(data, start, end, wcs=None, extend=False):
    """
    Returns the profile of some 2D data along a line specified by the start and end
    points. Also returns the x- and y-pixel indices of the line. Uses very rough
    nearest-neighbour sampling.

    Tip: a good way to ensure the line is aligned with the position angle (PA) of a
    galaxy is by specifying a start point, finding the direction vector using the galaxy's
    PA, then calculate the end point using end = start + direction_vector * length.
    For example:
        start = np.array([123, 456])
        pa = 338  # position angle, degrees. N.B. pa starts at North and increases CCW
        # May need to flip direction (i.e., multiply by by -1) depending on start coords
        direction = np.array([np.cos(np.deg2rad(pa + 90)), np.sin(np.deg2rad(pa + 90))])
        length_px = 900
        end = np.array(start + direction * length_px).astype(int)

    Parameters:
      data :: 2D array
        The data to be profiled
      start :: 2-element array-like of int or `astropy.coordinates.SkyCoord` object
        If tuple, (x, y) pixel coordinates of the start point. If SkyCoord, the WCS (e.g.,
        from the header using astropy.wcs.WCS(header)) must also be provided using the wcs
        parameter. N.B. converting pixel coordinates to SkyCoords and back to pixel
        coordinates (e.g., using astropy functions) may not give exactly the same
        coordinates
      end :: 2-element array-like of int or `astropy.coordinates.SkyCoord` object
        If tuple, (x, y) pixel coordinates of the end point. If SkyCoord, the WCS (e.g.,
        from the header using astropy.wcs.WCS(header)) must also be provided using the wcs
        parameter. N.B. converting pixel coordinates to SkyCoords and back to pixel
        coordinates (e.g., using astropy functions) may not give exactly the same
        coordinates because skycoord_to_pixel() returns floats
      wcs :: `astropy.wcs.WCS` object (optional)
        The world coordinate system to transform SkyCoord objects to pixel coordinates
      extend :: bool
        Not implemented yet!
        If True, extend the line defined by the start and end points to the edges of the
        image. If False, the line profile is only evaluated between the start and end
        points. Note that, because the coordinates are all integers, extending the line to
        the edge may not be exactly coincident with the line segment defined by the start
        and end points and may not go exactly to the edges. The final start and end points
        can be found using the returned idx arrays. That is, start = (x_idx[0], y_idx[0])
        and end = (x_idx[-1], y_idx[-1])

    Returns: profile
      profile :: 1D array
        The line profile of the data
      x_idx, y_idx :: 1D arrays
        The pixel coordinates of the line
    """
    from astropy.wcs.utils import skycoord_to_pixel

    #
    # Check inputs
    #
    if isinstance(start, coord.SkyCoord):
        start = skycoord_to_pixel(start, wcs=wcs)
        start = (int(start[0]), int(start[1]))
    if isinstance(end, coord.SkyCoord):
        end = skycoord_to_pixel(end, wcs=wcs)
        end = (int(end[0]), int(end[1]))
    if np.shape(start) != (2,) or np.shape(end) != (2,):
        raise ValueError("start and end must have exactly 2 elements")
    if not all(isinstance(var, (int, np.int64)) for var in start) or not all(
        isinstance(var, (int, np.int64)) for var in end
    ):
        raise ValueError(
            "start and end must be 2-element array-like of ints or SkyCoord objects"
        )
    if start[0] == end[0] and start[1] == end[1]:
        raise ValueError("start and end points must be different")
    if any([num < 0 for num in start]) or any([num < 0 for num in end]):
        raise ValueError("start and end points cannot correspond to negative pixels")
    # Too lazy to do more checks. User should be able to debug index out of bounds errors
    x0, y0 = start
    x1, y1 = end
    length = int(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
    if extend:
        raise NotImplementedError("extend=True not implemented yet. Sorry!")
    x_idx = np.linspace(x0, x1, length).astype(int)
    y_idx = np.linspace(y0, y1, length).astype(int)
    profile = data[y_idx, x_idx]  # since array indexing is [row, col]
    return profile, x_idx, y_idx


def calc_mag(flux, flux_err=0.0, zpt=30.0, calc_abs=False, dist=None, dist_err=0.0):
    """
    Calculates the relative or absolute magnitude of an object given its flux.

    Parameters:
      flux :: array
        The flux of the pixel
      flux_err :: array (optional)
        The uncertainty in the flux. Must be able to broadcast with flux array
      zpt :: float (optional)
        The zero point of the magnitude system
      calc_abs :: bool (optional)
        If True, returns the absolute magnitude, otherwise returns the relative magnitude.
        Requires that dist is also provided.
      dist :: scalar or array (optional)
        The distance to the object/pixel in parsecs. Must be able to broadcast with flux
        array
      dist_err :: float or array (optional)
        The uncertainty in the distance. Must be able to broadcast with flux array

    Returns: mag, mag_err
      mag :: array
        The magnitude of the pixel
      mag_err :: array
        The uncertainty in the magnitude
    """
    rel_mag = -2.5 * np.log10(flux) + zpt
    rel_mag_err = 2.5 / np.log(10) * abs(flux_err / flux)
    if calc_abs:
        if dist is None:
            raise ValueError("dist must be provided if calc_abs is True")
        abs_mag = rel_mag - 5 * (np.log10(dist) - 1)
        abs_mag_err = np.sqrt(rel_mag_err ** 2 + (5 / np.log(10) * dist_err / dist) ** 2)
        return abs_mag, abs_mag_err
    return rel_mag, rel_mag_err


def calc_colour(blue, red):
    """
    Calculates the colour (aka colour index) of an object.

    The colour, or colour index, is defined as:
                        colour = -2.5 * log10(blue / red)
    where blue is the flux in the shorter wavelength band and red is the flux in the
    longer wavelength band.

    Parameters:
      blue :: array
        The flux in the shorter wavelength band. Must be able to broadcast with red array
      red :: array
        The flux in the longer wavelength band. Must be able to broadcast with blue array

    Returns: colour
      colour :: array
        The "blue-red" colour of the object
    """
    return -2.5 * np.log10(blue / red)


def calc_colour_err(blue, red, blue_err, red_err):
    """
    Calculates the uncertainty in the colour index using basic uncertainty propagation.

    The colour, or colour index, is defined as:
                        colour = -2.5 * log10(blue / red)
    where blue is the flux in the shorter wavelength band and red is the flux in the
    longer wavelength band.

    The uncertainty, assuming independent errors and to a first-order approximation, is
    given by:
            colour_err^2 = (-2.5/ln(10) * 1/blue)^2 * blue_err^2 +
                           (+2.5/ln(10) * 1/red)^2 * red_err^2
                         = (2.5/ln(10))^2 * [(blue_err/blue)^2 + (red_err/red)^2]
    Thus:
            colour_err = sqrt(colour_err^2)
                       = 2.5 / ln(10) * sqrt((blue_err/blue)^2 + (red_err/red)^2)

    Note that all the parameters MUST be able to broadcast together.

    Parameters:
      blue :: array
        The flux in the shorter wavelength band
      red :: array
        The flux in the longer wavelength band
      blue_err :: array
        The uncertainty in the flux in the shorter wavelength band
      red_err :: array
        The uncertainty in the flux in the longer wavelength band

    Returns: colour_err
      colour_err :: array
        The uncertainty in the colour index.
    """
    prefactor = 2.5 / np.log(10)
    errs = np.sqrt((blue_err / blue) ** 2 + (red_err / red) ** 2)
    return prefactor * errs


def calc_tot_sn(signal, noise, func=np.nansum):
    """
    Calculates the total signal and noise in the input arrays.

    Parameters:
      signal, noise :: arrays
        The signal and noise arrays
      func :: `np.sum` or `np.nansum` (optional)
        The function used to calculate the total signal and noise

    Returns: tot_signal, tot_noise
      tot_signal :: float
        The total signal in the input signal array (straight sum)
      tot_noise :: float
        The total noise in the input noise array (summed in quadrature)
    """
    # pylint: disable=comparison-with-callable
    if func != np.nansum and func != np.sum:
        raise ValueError("func must be numpy.sum or numpy.nansum")
    tot_signal = func(signal)
    # Add noise in quadrature
    tot_noise = noise * noise
    tot_noise = func(tot_noise)
    tot_noise = np.sqrt(tot_noise)
    return tot_signal, tot_noise


def optimal_sn(index, signal, noise):
    """
    Signal-to-noise ratio (S/N) approximation using optimal weighing of pixels. Intended
    to be used with Voronoi binning (i.e., voronoi_2d_binning(sn_func=optimal_sn)).

    See Eq. (3) of Cappellari & Copin (2003):
    https://ui.adsabs.harvard.edu/abs/2003MNRAS.342..345C/abstract

    Parameters: (nearly verbatim from Cappellari & Copin's voronoi_2d_binning.py)
      index :: 1D array
        Integer vector of length N containing the indices of the spaxels for which the
        combined S/N has to be returned. The indices refer to elements of the vectors'
        signal and noise.
      signal :: 1D array
        Vector of length M>N with the signal of all spaxels.
      noise :: 1D array
        Vector of length M>N with the noise of all spaxels.

    Returns: sn
      sn :: 1D array
        Scalar S/N or another quantity that needs to be equalized.
    """
    return np.sqrt(
        np.sum((signal[index] / noise[index]) * (signal[index] / noise[index]))
    )


# ----------------------------- FUNCTIONS FOR REGULAR BINNING ----------------------------


def reg_bin_sn(signal, noise, block_size=(4, 4), print_info=True, func=np.sum):
    """
    Regular preliminary binning of 2D signal & noise data (e.g., for Voronoi binning).

    (From astropy's block_reduce() documentation): If the data are not perfectly divisible
    by block_size along a given axis, then the data will be trimmed (from the end) along
    that axis.

    TODO: finish docstring
    """
    from astropy.nddata import block_reduce

    # Add signal array using arithmetic sum
    signal_binned = block_reduce(signal, block_size=block_size, func=func)
    # Add noise array in quadrature
    noise_binned = noise * noise
    noise_binned = block_reduce(noise_binned, block_size=block_size, func=func)
    noise_binned = np.sqrt(noise_binned)
    #
    # Generate pixel coordinates
    #
    y_coords, x_coords = np.meshgrid(
        np.arange(signal_binned.shape[0]), np.arange(signal_binned.shape[1])
    )
    x_coords, y_coords = x_coords.T, y_coords.T
    #
    # Ensure no infs or NaNs in binned data (for Voronoi binning)
    #
    is_good = np.isfinite(signal_binned) & np.isfinite(noise_binned)
    #
    if print_info:
        print(
            "x_coords, y_coords, signal_binned, noise_binned shapes:",
            x_coords.shape,
            y_coords.shape,
            signal_binned.shape,
            noise_binned.shape,
        )
        print("total bad elements (infs/NaNs):", np.sum(~is_good))
    #
    return signal_binned, noise_binned, x_coords, y_coords, is_good


# * USE THESE IF INPUT ARRAYS ARE NOT YET CUT OUT TO TARGET


def get_reproject_shape_factor(target_arr, input_wcs, target_wcs):
    """
    Determine reprojection shape and binning factor.

    N.B. reprojection shape should be as close as possible to the shape of a regular "cut
    out" if the input array was cut at the boundaries of the target.

    TODO: finish docstring
    """
    #
    # Find the coordinates of the target_data's edges, assuming the data are rectangular
    #
    target_bottomleft = target_wcs.pixel_to_world(0, 0)
    target_topright = target_wcs.pixel_to_world(*target_arr.shape)
    #
    # Map the pixels above to their corresponding pixels in the input array
    #
    input_bottomleft = input_wcs.world_to_pixel(target_bottomleft)
    input_topright = input_wcs.world_to_pixel(target_topright)
    #
    # Determine binning/transformation factor
    #
    input_to_target_factor = np.subtract(input_topright, input_bottomleft)
    input_to_target_factor = np.round(
        np.divide(input_to_target_factor, target_arr.shape)
    ).astype(int)
    input_reproject_shape = input_to_target_factor * target_arr.shape
    #
    return input_to_target_factor, input_reproject_shape


def reproj_arr(
    input_arr,
    input_wcs,
    target_wcs,
    input_to_target_factor,
    input_reproject_shape,
    reproject_func=reproject.reproject_exact,
):
    """
    Reproject input array to target array.

    Requires the reproject package: https://reproject.readthedocs.io/en/stable/

    TODO: finish docstring
    """
    #
    # Reproject input array to target array
    #
    wcs_reproject = target_wcs.deepcopy()
    wcs_reproject.wcs.crpix = target_wcs.wcs.crpix * input_to_target_factor
    if wcs_reproject.wcs.has_cd():
        wcs_reproject.wcs.cd = target_wcs.wcs.cd / input_to_target_factor
    else:
        wcs_reproject.wcs.cdelt = target_wcs.wcs.cdelt / input_to_target_factor
    wcs_reproject.array_shape = input_reproject_shape
    arr_reproject = reproject_func(
        (input_arr, input_wcs),
        wcs_reproject,
        shape_out=input_reproject_shape,
        return_footprint=False,
    )
    return arr_reproject, wcs_reproject


def bin_sn_arrs_to_target(
    signal_arr,
    signal_wcs,
    noise_arr,
    noise_wcs,
    target_arr,
    target_wcs,
    reproject_func=reproject.reproject_exact,
    bin_func=np.nansum,
    print_debug=False,
    return_bin_dimen=False,  # for backwards compatibility...
):
    """
    Bin a signal & noise array to the exact physical dimensions and resolution of a target
    (provided via the target_wcs). The input arrays should already be masked (i.e.,
    invalid values should be set to np.nan) and the input arrays should entirely contain
    the target_wcs (i.e., the extent of the target_data).

    Note that the signal_wcs and noise_wcs should be identical! The returned WCS object
    will be based on the signal_wcs.

    Requires the reproject package: https://reproject.readthedocs.io/en/stable/

    Parameters:
      bin_func :: `numpy.sum` or `numpy.nansum` (optional)

    Returns: (x_coords, y_coords, signal_binned, noise_binned, is_good, wcs_binned,
              bin_dimensions <- maybe)

    TODO: finish docstring
    """
    if print_debug:
        print("target_wcs:", target_wcs)
        print()
    #
    # Determine reprojection shape and binning factor
    #   N.B. reprojection shape should be as close as possible to the shape of a regular
    #   "cut out" if the input array was cut at the boundaries of the target.
    #
    signal_to_target_factor, signal_reproject_shape = get_reproject_shape_factor(
        target_arr, signal_wcs, target_wcs
    )
    if print_debug:
        print("signal_to_target_factor:", signal_to_target_factor)
        print("signal_reproject_shape:", signal_reproject_shape)
        print()
    noise_to_target_factor, noise_reproject_shape = get_reproject_shape_factor(
        target_arr, noise_wcs, target_wcs
    )
    if np.any(signal_reproject_shape != noise_reproject_shape) or np.any(
        signal_to_target_factor != noise_to_target_factor
    ):
        raise ValueError("Signal and noise arrays must have the same shape and wcs.")
    #
    # Reproject data
    #
    # N.B. signal_wcs_reproject and noise_wcs_reproject should be the same
    signal_reproject, signal_wcs_reproject = reproj_arr(
        signal_arr,
        signal_wcs,
        target_wcs,
        signal_to_target_factor,
        signal_reproject_shape,
        reproject_func=reproject_func,
    )
    if print_debug:
        print("signal_wcs_reproject:", signal_wcs_reproject)
        print()
    noise_reproject, noise_wcs_reproject = reproj_arr(  # pylint: disable=unused-variable
        noise_arr,
        noise_wcs,
        target_wcs,
        noise_to_target_factor,
        noise_reproject_shape,
        reproject_func=reproject_func,
    )
    #
    # Bin to target resolution
    #
    signal_binned, noise_binned, x_coords, y_coords, is_good = reg_bin_sn(
        signal_reproject,
        noise_reproject,
        block_size=signal_to_target_factor,
        func=bin_func,
        print_info=print_debug,
    )
    # Modify WCS object
    wcs_binned = signal_wcs_reproject.slice(
        (np.s_[:: signal_to_target_factor[0]], np.s_[:: signal_to_target_factor[1]])
    )  # slicing order is correct.
    wcs_binned.wcs.crpix = signal_wcs_reproject.wcs.crpix / signal_to_target_factor
    if signal_wcs_reproject.wcs.has_cd():
        wcs_binned.wcs.cd = (
            signal_wcs_reproject.wcs.cd * signal_to_target_factor
        )
    else:
        wcs_binned.wcs.cdelt = (
            signal_wcs_reproject.wcs.cdelt * signal_to_target_factor
        )
    if print_debug:
        print("wcs_binned:", wcs_binned)
    #
    if return_bin_dimen:
        return (
            x_coords,
            y_coords,
            signal_binned,
            noise_binned,
            is_good,
            wcs_binned,
            signal_to_target_factor,
        )
    else:
        return x_coords, y_coords, signal_binned, noise_binned, is_good, wcs_binned


# * END OF "USE THESE IF INPUT ARRAYS ARE NOT YET CUT OUT TO TARGET"


# * USE THESE IF INPUT ARRAYS ARE ALREADY CUT OUT TO TARGET. MAY BE BUGGY


def get_reproject_shape_factor_cut(input_arr, target_arr, input_wcs, target_wcs):
    """
    Determine reprojection shape and binning factor. Assumes input array has already been
    cut to desired extent.

    N.B. reprojection shape should be as close as possible to the shape of a regular "cut
    out" if the input array was cut at the boundaries of the target.

    Note the orders of each returned tuple:
    - reproject_shape is (y, x)
    - input_to_reproject_factor is (x, y)
    - reproject_to_target_factor is (x, y)
    Blame numpy vs. WCS for this confusing mess

    TODO: finish docstring
    """
    #
    # Find the coordinates of the target_data's edges, assuming the data are rectangular
    #
    target_bottomleft = target_wcs.pixel_to_world(0, 0)
    target_topright = target_wcs.pixel_to_world(*target_arr.shape)  # don't reverse
    #
    # Map the pixels above to their corresponding pixels in the input array
    #
    input_bottomleft = input_wcs.world_to_pixel(target_bottomleft)
    input_topright = input_wcs.world_to_pixel(target_topright)
    # input_centre = input_wcs.world_to_pixel(target_centre)  # don't need this
    #
    # Determine binning/transformation factor
    #
    reproject_to_target_factor = np.subtract(input_topright, input_bottomleft)
    reproject_to_target_factor = np.round(
        np.divide(reproject_to_target_factor, target_arr.shape)
    ).astype(int)[
        ::-1
    ]  # Reverse order to match WCS attributes convention
    reproject_shape = reproject_to_target_factor * target_arr.shape
    input_to_reproject_factor = input_arr.shape / reproject_shape
    input_to_reproject_factor = input_to_reproject_factor[::-1]
    #
    return reproject_shape, input_to_reproject_factor, reproject_to_target_factor


def reproject_cut_arr(
    input_arr,
    input_wcs,
    input_to_reproject_factor,
    input_reproject_shape,
    reproject_func=reproject.reproject_exact,
):
    """
    Reproject input array to target array and update input array's WCS object. Meant to be
    used for bin_cut_sn_arrs_to_target(). Assumes input array has already been cut to
    desired extent.

    Requires the reproject package: https://reproject.readthedocs.io/en/stable/

    input_to_reproject_factor should be (x, y) while input_reproject_shape should be (y,
    x). Blame the WCS vs. numpy convention.

    TODO: finish docstring
    """
    #
    # Reproject input array to target array
    #
    wcs_reproject = input_wcs.deepcopy()
    wcs_reproject.wcs.crpix = input_wcs.wcs.crpix * input_to_reproject_factor
    if input_wcs.wcs.has_cd():
        wcs_reproject.wcs.cd = input_wcs.wcs.cd / input_to_reproject_factor
    else:
        wcs_reproject.wcs.cdelt = input_wcs.wcs.cdelt / input_to_reproject_factor
    wcs_reproject.array_shape = input_reproject_shape
    arr_reproject = reproject_func(
        (input_arr, input_wcs),
        wcs_reproject,
        shape_out=input_reproject_shape,
        return_footprint=False,
    )
    return arr_reproject, wcs_reproject


def bin_cut_sn_arrs_to_target(
    signal_arr,
    signal_wcs,
    noise_arr,
    noise_wcs,
    target_arr,
    target_wcs,
    reproject_func=reproject.reproject_exact,
    bin_func=np.sum,
    print_debug=False,
    return_bin_dimen=False,  # for backwards compatibility...
):
    """
    Bin a signal & noise array to the exact physical dimensions and resolution of a target
    (provided via the target_wcs). The input arrays should already be masked (i.e.,
    invalid values should be set to np.nan) and the input arrays should entirely contain
    the target_wcs (i.e., the extent of the target_data).

    Importantly, the signal & noise arrays + wcs should already be cut to the extent you
    wish to bin. This function will not cutout the arrays! You should use cutout() or
    cutout_to_target() to cut out the signal_arr and noise_arr to the extent of the
    target_arr first.

    Note that the signal_wcs and noise_wcs should be identical! The returned WCS object
    will be based on the signal_wcs.

    Requires the reproject package: https://reproject.readthedocs.io/en/stable/

    Parameters:
      bin_func :: `numpy.sum` or `numpy.nansum` (optional)

    Returns: (x_coords, y_coords, signal_binned, noise_binned, is_good, wcs_binned,
              bin_dimensions <- maybe)

    TODO: finish docstring
    """
    if print_debug:
        print("signal_arr.shape:", signal_arr.shape)
        print("signal_wcs:", signal_wcs)
        print()
        print("target_wcs:", target_wcs)
        print()
    # #
    # # Do not cut out arrays because of propagation of floating point errors (+/- 1 pixel)
    # # using world_to_pixel() and pixel_to_world()
    # #
    # signal_arr, noise_arr = np.copy(signal_arr), np.copy(noise_arr)
    # signal_arr, signal_wcs = cutout_to_target(
    #     signal_arr, signal_wcs, target_arr, target_wcs
    # )
    # noise_arr, noise_wcs = cutout_to_target(noise_arr, noise_wcs, target_arr, target_wcs)
    #
    # Determine reprojection shape and binning factor
    #   N.B. reprojection shape should be as close as possible to the shape of a regular
    #   "cut out" if the input array was cut at the boundaries of the target.
    #
    (
        signal_reproject_shape,
        signal_to_reproject_factor,
        signal_reproject_to_target_factor,
    ) = get_reproject_shape_factor_cut(signal_arr, target_arr, signal_wcs, target_wcs)
    if print_debug:
        print("signal_reproject_shape (y, x):", signal_reproject_shape)
        print("signal_to_reproject_factor (y, x):", signal_to_reproject_factor)
        print(
            "signal_reproject_to_target_factor (x, y):", signal_reproject_to_target_factor
        )
        print()
    (
        noise_reproject_shape,
        noise_to_reproject_factor,
        noise_reproject_to_target_factor,
    ) = get_reproject_shape_factor_cut(noise_arr, target_arr, noise_wcs, target_wcs)
    if (
        np.any(signal_reproject_shape != noise_reproject_shape)
        or np.any(signal_to_reproject_factor != noise_to_reproject_factor)
        or np.any(signal_reproject_to_target_factor != noise_reproject_to_target_factor)
    ):
        raise ValueError("Signal and noise arrays must have the same shape and wcs.")
    #
    # Reproject data
    #
    # N.B. signal_wcs_reproject and noise_wcs_reproject should be the same
    signal_reproject, signal_wcs_reproject = reproject_cut_arr(
        signal_arr,
        signal_wcs,
        # target_wcs,
        signal_to_reproject_factor,
        signal_reproject_shape,
        reproject_func=reproject_func,
    )
    if print_debug:
        print("signal_reproject.shape:", signal_reproject.shape)
        print("signal_wcs_reproject:", signal_wcs_reproject)
        print()
    # pylint: disable=unused-variable
    noise_reproject, noise_wcs_reproject = reproject_cut_arr(
        noise_arr,
        noise_wcs,
        # target_wcs,
        noise_to_reproject_factor,
        noise_reproject_shape,
        reproject_func=reproject_func,
    )
    #
    # Bin to target resolution
    #
    signal_binned, noise_binned, x_coords, y_coords, is_good = reg_bin_sn(
        signal_reproject,
        noise_reproject,
        block_size=signal_reproject_to_target_factor,
        func=bin_func,
        print_info=print_debug,
    )
    # Modify WCS object
    wcs_binned = signal_wcs_reproject.slice(
        (
            np.s_[:: signal_reproject_to_target_factor[0]],
            np.s_[:: signal_reproject_to_target_factor[1]],
        )
    )  # slicing order is correct
    signal_reproject_to_target_factor = signal_reproject_to_target_factor[::-1]
    wcs_binned.wcs.crpix = (
        signal_wcs_reproject.wcs.crpix / signal_reproject_to_target_factor
    )
    if signal_wcs_reproject.wcs.has_cd():
        wcs_binned.wcs.cd = (
            signal_wcs_reproject.wcs.cd * signal_reproject_to_target_factor
        )
    else:
        wcs_binned.wcs.cdelt = (
            signal_wcs_reproject.wcs.cdelt * signal_reproject_to_target_factor
        )
    if print_debug:
        print("wcs_binned:", wcs_binned)
    #
    if return_bin_dimen:
        return (
            x_coords,
            y_coords,
            signal_binned,
            noise_binned,
            is_good,
            wcs_binned,
            signal_reproject_to_target_factor,
        )
    else:
        return x_coords, y_coords, signal_binned, noise_binned, is_good, wcs_binned


# * END OF "USE THESE IF INPUT ARRAYS ARE ALREADY CUT OUT TO TARGET. MAY BE BUGGY"


# -------------------------- END FUNCTIONS FOR REGULAR BINNING ---------------------------


def MLi_taylor2011(gband, iband, dist, wcs, gband_err=0, iband_err=0, px_per_bin=1):
    """
    Estimates the mass-to-light ratio, i-band luminosities, and stellar mass densities
    using g-i colours + i-band fluxes and following the prescription from Taylor et al.
    (2011). Also calculates the random errors in the estimates. Note thet this function
    does not account for systematic (i.e., distance) uncertainties.

    Parameters:
      gband, iband :: arrays
        The g- and i-band fluxes, respectively
      dist :: `astropy.units.quantity.Quantity` object
        The distance to the target
      wcs :: `astropy.wcs.WCS` object
        The WCS of the gband and iband data. Used to calculate the physical area (to get
        densities)
      gband_err, iband_err :: arrays or scalar
        The uncertainties in the g- and i-band fluxes, respectively. If a scalar, the same
        error will be used for the entire array
      px_per_bin :: array or int
        The number of pixels per bin where 1 pixel is the pixel size defined by the WCS
        object (e.g., for regular binning, px_per_bin=1 vs. px_per_bin varies in Voronoi
        binning). Must be able to broadcast with gband & iband arrays

    Returns: results, errors
      results :: list of [avg_iband_abs_mag, gi_colour, Li, MLi_ratio, M, M_density]
        avg_iband_abs_mag :: array
          The average i-band absolute AB magnitude per bin
      errors :: list of [avg_iband_abs_mag_err, gi_colour_err, Li_err, MLi_ratio_err,
                         M_err, M_density_err]

    Note that avg_iband_abs_mag and M_density are corrected (i.e. normalized) using
    px_per_bin (and in the case of M_density, scaled to units of solar masses per square
    parsec). gi_colour and MLi_ratio do not need to be corrected (areas cancel out). Li
    and M have not been corrected using px_per_bin (but this can easily be done by the
    user to get the average Li & M per bin).

    TODO: finish docstring
    """
    #
    # Check inputs
    #
    if np.shape(gband) != np.shape(iband):
        raise ValueError("gband and iband must have the same shape.")
    if (
        np.shape(gband) != np.shape(gband_err) or np.shape(iband) != np.shape(iband_err)
    ) and (
        not isinstance(gband_err, (int, float, np.int64, np.float64))
        or not isinstance(iband_err, (int, float, np.int64, np.float64))
    ):
        raise ValueError(
            "gband_err & iband_err must have the same shape as their respective data "
            + "arrays or be scalars."
        )
    if np.shape(gband) != np.shape(px_per_bin) and not isinstance(
        px_per_bin, (int, np.int64)
    ):
        raise ValueError(
            "px_per_bin must have the same shape as gband and iband or be a scalar."
        )
    #
    # Calculate i-band absolute AB magnitudes
    #
    iband_abs_mag, iband_abs_mag_err = calc_mag(
        iband, flux_err=iband_err, zpt=30, dist=dist.to(u.pc).value, calc_abs=True
    )
    # Average the i-band absolute AB magnitudes per bin
    avg_iband_abs_mag = iband_abs_mag / px_per_bin
    avg_iband_abs_mag_err = iband_abs_mag_err / px_per_bin
    #
    # Calculate mass-to-light ratio
    #
    # 1. Calculate g-i colour
    gi_colour = calc_colour(gband, iband)
    gi_colour_err = calc_colour_err(gband, iband, gband_err, iband_err)
    # 2. Apply Eq. (7) from Taylor et al. (2011)
    log_MLi_ratio = 0.7 * gi_colour - 0.68
    log_MLi_ratio_err = 0.7 * gi_colour_err
    MLi_ratio = 10 ** log_MLi_ratio  # mass-to-light ratio
    MLi_ratio_err = MLi_ratio * np.log(10) * log_MLi_ratio_err
    #
    # Calculate i-band luminosity and stellar mass
    #
    I_ABS_MAG_SUN = 4.58  # absolute i-band AB magnitude of Sun. From Taylor et al. (2011)
    Li = 10 ** (-0.4 * (iband_abs_mag - I_ABS_MAG_SUN))  # i-band luminosity, solar units
    Li_err = Li * np.log(10) * 0.4 * iband_abs_mag_err
    M = MLi_ratio * Li  # stellar mass in solar masses
    M_err = M * np.sqrt((MLi_ratio_err / MLi_ratio) ** 2 + (Li_err / Li) ** 2)
    #
    # Calculate stellar mass density
    #
    px_dimensions, _ = calc_pc_per_px(wcs, dist, dist_err=None)  # parsecs
    px_area = px_dimensions[0] * px_dimensions[1]  # square parsecs
    M_density = M / (px_area * px_per_bin)  # stellar mass density (M_sun per pc^2)
    M_density_err = M_err / (px_area * px_per_bin)  # assume no distance uncertainty
    #
    results = [avg_iband_abs_mag, gi_colour, Li, MLi_ratio, M, M_density]
    errors = [
        avg_iband_abs_mag_err,
        gi_colour_err,
        Li_err,
        MLi_ratio_err,
        M_err,
        M_density_err,
    ]
    return results, errors


def ML_from_mlcr(
    colour,
    flux,
    dist,
    wcs,
    colour_err=0,
    flux_err=0,
    px_per_bin=1,
    slope=0.7,
    yint=-0.68,
    sun_abs_mag_in_flux_band=4.58,
    zpt=30.0,
):
    """
    Estimates the mass-to-light ratio and various related quantities using an arbitrary
    mass-to-light vs. colour relation (MLCR). Also calculates the random errors in the
    estimates. Note thet this function does not account for systematic (i.e., distance)
    uncertainties.

    For example, to replicate the MLCR from Taylor et al. (2011), use:
    ```
    results, errors = ML_from_mlcr(
        gi_colour, iband_flux, dist, wcs,
        slope=0.7, yint=-0.68, sun_abs_mag_in_flux_band=4.58, zpt=30.0
    )
    ```

    Parameters:
      dist :: `astropy.units.quantity.Quantity` object
        The distance to the target
      wcs :: `astropy.wcs.WCS` object
        The WCS of the colour and flux data. Used to calculate the physical area (to get
        densities)
      px_per_bin :: array or int
        The number of pixels per bin where 1 pixel is the pixel size defined by the WCS
        object (e.g., for regular binning, px_per_bin=1 vs. px_per_bin varies in Voronoi
        binning). Must be able to broadcast with gband & iband arrays
      slope, y-int :: scalar (optional)
        The slope and y-intercept of the MLCR: log(ML) = slope * colour + yint. Default is
        the MLCR from Taylor et al. (2011)
      sun_abs_mag_in_flux_band :: scalar (optional)
        The absolute magnitude of the Sun in the same band as the flux data. Default is
        the absolute AB magnitude of the Sun in the i-band
      zpt :: scalar (optional)
        The zero-point of the magnitude system. For example, the AB magnitude system uses
        zpt=30.0

    Returns: results, errors
      results :: list of [avg_flux_abs_mag, L, ML_ratio, M, M_density]
        avg_flux_abs_mag :: array
          The absolute AB magnitudes of the flux data averaged per bin
      errors :: list of [avg_flux_abs_mag_err, L_err, ML_ratio_err, M_err, M_density_err]

    Note that avg_flux_abs_mag and M_density are corrected (i.e. normalized) using
    px_per_bin (and in the case of M_density, scaled to units of solar masses per square
    parsec). ML_ratio does not need to be corrected (areas cancel out). L and M have not
    been corrected using px_per_bin (but this can easily be done by the user to get the
    average L & M per bin).

    TODO: finish docstring
    """
    #
    # Check inputs
    #
    if np.shape(colour) != np.shape(colour_err) and not isinstance(
        colour_err, (int, float, np.int64, np.float64)
    ):
        raise ValueError("colour_err must have the same shape as colour or be a scalar.")
    if np.shape(flux) != np.shape(colour):
        raise ValueError("flux and colour and colour_err must all have the same shape.")
    if np.shape(colour) != np.shape(px_per_bin) and not isinstance(
        px_per_bin, (int, np.int64)
    ):
        raise ValueError(
            "px_per_bin must have the same shape as the input arrays or be a scalar."
        )
    if np.shape(flux) != np.shape(flux_err) and not isinstance(
        flux_err, (int, float, np.int64, np.float64)
    ):
        raise ValueError(
            "flux_err must have the same shape as the input arrays or be a scalar."
        )
    #
    # Calculate absolute AB magnitudes
    #
    flux_abs_mag, flux_abs_mag_err = calc_mag(
        flux, flux_err=flux_err, zpt=zpt, dist=dist.to(u.pc).value, calc_abs=True
    )
    # Average the absolute AB magnitudes per bin
    avg_flux_abs_mag = flux_abs_mag / px_per_bin
    avg_flux_abs_mag_err = flux_abs_mag_err / px_per_bin
    #
    # Calculate mass-to-light ratio using MLCR
    #
    log_ML_ratio = slope * colour + yint
    log_ML_ratio_err = slope * colour_err
    ML_ratio = 10 ** log_ML_ratio  # mass-to-light ratio
    ML_ratio_err = ML_ratio * np.log(10) * log_ML_ratio_err
    #
    # Calculate luminosity and stellar mass
    #
    L = 10 ** (-0.4 * (flux_abs_mag - sun_abs_mag_in_flux_band))  # luminosity, L_sun unit
    L_err = L * np.log(10) * 0.4 * flux_abs_mag_err
    M = ML_ratio * L  # stellar mass in solar masses
    M_err = M * np.sqrt((ML_ratio_err / ML_ratio) ** 2 + (L_err / L) ** 2)
    #
    # Calculate stellar mass density
    #
    px_dimensions, _ = calc_pc_per_px(wcs, dist, dist_err=None)  # parsecs
    px_area = px_dimensions[0] * px_dimensions[1]  # square parsecs
    M_density = M / (px_area * px_per_bin)  # stellar mass density (M_sun per pc^2)
    M_density_err = M_err / (px_area * px_per_bin)  # assume no distance uncertainty
    #
    results = [avg_flux_abs_mag, L, ML_ratio, M, M_density]
    errors = [avg_flux_abs_mag_err, L_err, ML_ratio_err, M_err, M_density_err]
    return results, errors


def load_sed_results(
    path,
    nmax,
    nmin=0,
    infile="prospectorFit_emcee_<NUMS>_results.txt",
    replace="<NUMS>",
    skip=None,
):
    """
    Takes the individual .txt files containing the SED fitting results and compiles the
    data into an array. Optionally outputs a .txt file containing the amalgamated results
    (not implemented yet). N.B. nmax is inclusive. Assumes the shape of each file is the
    same.

    Parameters:
      path :: str
        The directory containing the individual .txt files
      nmax :: int
        The last file number to load (based on the filename), inclusive
      nmin :: int (optional)
        The first file number to load (based on the filename), inclusive
      infile :: str (optional)
        An example of the filename (including the .txt extension) to load
      replace :: str (optional)
        The substring to in the infile str to replace with numbers between [nmin, nmax]
      skip :: array-like of ints (optional)
        The files to skip and assign to NaN.

    Returns: results
      results :: 2D array
        The SED fitting results. Each row contains data from one file. The columns contain
        the same data (i.e., parameter) across all files.
    """
    # Check directory and filename
    if path[-1] != "/":
        path += "/"
    if infile[-4:] != ".txt":
        raise ValueError("infile must be a .txt file.")
    #
    start_idx = infile.index(replace)
    end_idx = start_idx + len(replace)
    pre_str = infile[:start_idx]
    post_str = infile[end_idx:]
    len_digits = len(str(nmax))
    results = []
    if skip is None:
        skip = []
    for i in range(nmin, nmax + 1):
        if i in skip:
            if i > nmin:
                results.append(np.zeros_like(results[-1]) * np.nan)
            else:
                raise ValueError("skip must be greater than nmin.")
        else:
            results.append(
                # pd.read_csv() much faster than np.loadtxt()
                pd.read_csv(
                    path + pre_str + str(i).zfill(len_digits) + post_str,
                    sep=" ",
                    header=None,
                ).values
            )
            # results.append(
            #     np.loadtxt(path + pre_str + str(i).zfill(len_digits) + post_str)
            # )
    results = np.vstack(results)
    return results


def calc_gas_fraction(gas_density, mass_density, gas_density_err=0, mass_density_err=0):
    """
    Calculates the gas fraction its uncertainty.

    The gas fraction is defined as
                        gas fraction = log10(gas_density / mass_density)

    Parameters:
      gas_density :: array-like
        The gas density. Must have same units as mass_density
      mass_density :: array-like
        The mass density. Must have same units as gas_density
      gas_density_err :: array-like (optional)
        The uncertainty in gas_density
      mass_density_err :: array-like (optional)
        The uncertainty in mass_density

    Returns:
      gas_fraction :: array-like
        The gas fraction
      gas_fraction_err :: array-like
        The uncertainty in the gas fraction
    """
    # Process inputs
    gas_density = np.asarray(gas_density)
    gas_density_err = np.asarray(gas_density_err)
    mass_density = np.asarray(mass_density)
    mass_density_err = np.asarray(mass_density_err)
    # Calculate gas fraction
    gas_fraction = np.log10(gas_density / mass_density)
    gas_fraction_err = np.sqrt(
        (gas_density_err / gas_density) ** 2 + (mass_density_err / mass_density) ** 2
    ) / np.log(10)
    return gas_fraction, gas_fraction_err


def get_worst_img_qual(header_lst, header_key="IQMAX", header_unit=u.arcsec):
    """
    Gets the worst input image quality given a list of headers.

    Parameters:
      header_lst :: list of astropy.io.fits.header.Header
        The headers of the images
      header_key :: str (optional)
        The header keyword containing the image quality
      header_unit :: `astropy.units.quantity.Quantity` object (optional)
        The unit of the header keyword corresponding to the image quality

    Returns: worst_img_qual, worst_img_qual_idx
      worst_img_qual :: `astropy.units.quantity.Quantity`
        The worst image quality
      worst_img_qual_idx :: int
        The index of header_lst containing the header with the worst image quality
    """
    # worst_img_qual = np.max([header[header_key] for header in header_lst])
    worst_img_qual_idx = np.argmax([header[header_key] for header in header_lst])
    worst_img_qual = header_lst[worst_img_qual_idx][header_key] * header_unit
    return worst_img_qual, worst_img_qual_idx


# ------------------------ MISCELLANEOUS FUNCTIONS FOR MY OWN USE ------------------------


def _dill_regBin_results(
    outfile,
    reproject_method,
    dist_pc,
    x_coords,
    y_coords,
    signal_binned,
    noise_binned,
    abs_mag,
    abs_mag_err,
    is_good,
    wcs_binned,
):
    """
    Misc. function to help save results of binning optical data to VERTICO's Nyquist or 2"
    pixel resolution. For my own use, so no documentation.
    """
    import dill

    with open(outfile, "wb") as f:
        dill.dump(
            {
                "note": "Remember to set `wcs_binned.array_shape = wcs_binned_array_shape`",
                "reproject_method": reproject_method,
                "dist_pc": dist_pc,  # distance in parsecs
                "x_coords": x_coords,  # x-value of pixel coordinates
                "y_coords": y_coords,  # y-value of pixel coordinates
                "signal_binned": signal_binned,
                "noise_binned": noise_binned,
                "abs_mag": abs_mag,
                "abs_mag_err": abs_mag_err,  # should be excluding any systematic errors
                "is_good": is_good,
                "wcs_binned": wcs_binned,  # dill has trouble saving the "NAXIS" keyword
                "wcs_binned_array_shape": wcs_binned.array_shape,  # "NAXIS" keyword
            },
            f,
        )
    print(f"Pickled {outfile}")


def _txt_mags(
    outfile,
    xs,
    ys,
    u_mag,
    u_mag_err,
    g_mag,
    g_mag_err,
    i_mag,
    i_mag_err,
    z_mag,
    z_mag_err,
):
    """
    Misc. function to help me save the absolute magnitudes of the optical data so it can
    be processed for SED fitting by Dr. Roediger.
    """
    len_digits = len(str(np.max((np.max(xs), np.max(ys)))))
    id_arr = np.array(
        [
            str(x).zfill(len_digits) + str(y).zfill(len_digits)
            for x, y in zip(xs.flatten(), ys.flatten())
        ]
    )
    df = pd.DataFrame(
        {
            "id": id_arr,
            "z": np.zeros(xs.size),
            "u_mag": u_mag.flatten(),
            "u_mag_err": u_mag_err.flatten(),
            "g_mag": g_mag.flatten(),
            "g_mag_err": g_mag_err.flatten(),
            "i_mag": i_mag.flatten(),
            "i_mag_err": i_mag_err.flatten(),
            "z_mag": z_mag.flatten(),
            "z_mag_err": z_mag_err.flatten(),
        }
    )
    df.to_csv(path_or_buf=outfile, sep=" ", index=False, header=True)
    print(f"Saved {outfile}")


def _txt_mags_from_pkl(galpath, galaxy, bin_resolution, outfile):
    """
    Misc. function to help me save the absolute magnitudes of the optical data so it can
    be processed for SED fitting by Dr. Roediger.

    Wrapper for _txt_mags() when data are from pickle files.

    Parameters: all strings
    """
    # Load pickled data
    import dill

    uband_outfile = galpath + f"{galaxy}_regBin_uband_{bin_resolution}.pkl"
    with open(uband_outfile, "rb") as f:
        file = dill.load(f)
        uband_xs_binned = file["x_coords"]
        uband_ys_binned = file["y_coords"]
        uband_abs_mag_binned = file["abs_mag"]
        uband_abs_mag_err_binned = file["abs_mag_err"]
    gband_outfile = galpath + f"{galaxy}_regBin_gband_{bin_resolution}.pkl"
    with open(gband_outfile, "rb") as f:
        file = dill.load(f)
        gband_abs_mag_binned = file["abs_mag"]
        gband_abs_mag_err_binned = file["abs_mag_err"]
    iband_outfile = galpath + f"{galaxy}_regBin_iband_{bin_resolution}.pkl"
    with open(iband_outfile, "rb") as f:
        file = dill.load(f)
        iband_abs_mag_binned = file["abs_mag"]
        iband_abs_mag_err_binned = file["abs_mag_err"]
    zband_outfile = galpath + f"{galaxy}_regBin_zband_{bin_resolution}.pkl"
    with open(zband_outfile, "rb") as f:
        file = dill.load(f)
        zband_abs_mag_binned = file["abs_mag"]
        zband_abs_mag_err_binned = file["abs_mag_err"]
    _txt_mags(
        outfile,
        uband_xs_binned,
        uband_ys_binned,
        uband_abs_mag_binned,
        uband_abs_mag_err_binned,
        gband_abs_mag_binned,
        gband_abs_mag_err_binned,
        iband_abs_mag_binned,
        iband_abs_mag_err_binned,
        zband_abs_mag_binned,
        zband_abs_mag_err_binned,
    )


# ---------------------- END MISCELLANEOUS FUNCTIONS FOR MY OWN USE ----------------------
