"""
fits_plot_utils.py

Utilities for generating plots (e.g., imshow, contour, contourf) from FITS files.
Originally developed for NGVS and VERTICO data processing.

Isaac Cheng - September 2021
"""

import copy
import warnings

import astropy
import astropy.coordinates as coord
import astropy.units as u
import dill
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import reproject
import seaborn as sns
from astropy.io import fits
from astropy.nddata import Cutout2D, block_reduce
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales, skycoord_to_pixel
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from radio_beam import Beam


#
# Colour bars with custom midpoints. NOT MY CODE. See docstrings
# I have not checked edge cases (i.e., be careful with MidPointLogNorm)
#
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    From https://stackoverflow.com/a/20528097.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


class MidPointLogNorm(mpl.colors.LogNorm):
    """
    LogNorm with adjustable midpoint. From https://stackoverflow.com/a/48632237.

    ! WARNING: Be careful with parameter edge cases. !
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        mpl.colors.LogNorm.__init__(self, vmin=vmin, vmax=vmax, clip=clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        x, y = [np.log(self.vmin), np.log(self.midpoint), np.log(self.vmax)], [0, 0.5, 1]
        return np.ma.array(np.interp(np.log(value), x, y), mask=result.mask, copy=False)


def load_img(imgpath, idx=0):
    """
    Retrieves and returns the image data and header from a .fits file
    given a filepath.

    Parameters:
      imgpath :: str
        The path to the .fits file
      idx :: int (optional, default: 0)
        The index of the block from which to extract the data

    Returns: imgdata, imgheader, imgwcs
      imgdata :: 2D array
        An array with the pixel values from the .fits file
      imgheader :: `astropy.io.fits.header.Header`
        The header to the .fits file
    """
    with fits.open(imgpath) as hdu_list:
        hdu_list.info()
        # Get image data (typically in PRIMARY block)
        imgdata = hdu_list[idx].data  # 2D array
        # Get image header and WCS coordinates
        imgheader = hdu_list[idx].header
    return imgdata, imgheader


def add_scalebar(
    ax,
    imgwcs,
    dist,
    scalebar_factor=1,
    label="1 kpc",
    color="k",
    loc="lower right",
    size_vertical=0.5,
    pad=1,
    fontsize=12,
    **kwargs
):
    """
    Adds a 1 kpc scale bar (by default) to a plot.

    Parameters:
      ax :: `matplotlib.axes._subplots.AxesSubplot`
        The matplotlib axis object on which to add the scalebar
      imgwcs :: `astropy.wcs.wcs.WCS`
        The WCS coordinates of the .fits file
      dist :: `astropy.units.quantity.Quantity` scalar
        The distance to the object
      scalebar_factor :: float
        Factor by which to multiply the 1 kpc scale bar.
      label :: str (optional, default: "1 kpc")
        The scale bar label
      color :: str (optional, default: "k")
        Colour of the scale bar and label
      loc :: str or int (optional, default: "lower right")
        The location of the scale bar and label
      size_vertical :: float (optional, default: 0.5)
        Vertical length of the scale bar (in ax units)
      pad :: float or int (optional, default: 1)
        Padding around scale bar and label (in fractions of the font size)
      fontsize :: str or int (optional, default: 12)
        The font size for the label
      **kwargs :: dict (optional)
        Keyworded arguments to pass to `matplotlib.offsetbox.AnchoredOffsetbox`

    Returns: ax.add_artist(scalebar)
      ax.add_artist(scalebar) :: `mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar`
        Adds the scale har to the given ax
    """
    arcsec_per_px = (proj_plane_pixel_scales(imgwcs.celestial)[0] * u.deg).to(u.arcsec)
    # Still deciding whether to use arctan for the next line
    arcsec_per_kpc = np.rad2deg(1 * u.kpc / dist.to(u.kpc) * u.rad).to(u.arcsec)
    px_per_kpc = arcsec_per_kpc / arcsec_per_px
    # Set font properties
    fontproperties = mpl.font_manager.FontProperties(size=fontsize)
    # Add scale bar
    scalebar = AnchoredSizeBar(
        ax.transData,
        px_per_kpc * scalebar_factor,
        label=label,
        loc=loc,
        fontproperties=fontproperties,
        pad=pad,
        color=color,
        frameon=False,
        size_vertical=size_vertical,
        **kwargs
    )
    return ax.add_artist(scalebar)


def add_cbar(fig, img, label=None, tick_params=None):
    """
    Adds a colour bar to the given figure.

    Parameters:
      fig :: `matplotlib.figure.Figure`
        The figure object on which to add the colour bar
      img :: `matplotlib.image.AxesImage`
        The image object to which the colour bar is attached
      label :: str (optional, default: None)
        The label for the colour bar
      tick_params :: dict (optional, default: None)
        Keyworded arguments to pass to `matplotlib.colorbar.ColorbarBase`. If None,
        will set colour bar ticks to point outside with minor ticks on

    Returns: None
    """
    if tick_params is None:
        tick_params = {"which": "both", "direction": "out"}
    cbar = fig.colorbar(img)
    cbar.ax.tick_params(**tick_params)
    cbar.set_label(label) if label is not None else None


def add_plot_labels(
    ax,
    xax=None,
    yax=None,
    title=None,
    aspect="equal",
    grid=False,
    fig=None,
    figname=None,
    dpi=300,
    tick_params=None,
):
    """
    Convenience function to add labels to a given plot, set the axis aspect, turn on/off
    the grid, and adjust the tick parameters. If fig and figname given, will save plot to
    figname.

    Parameters:
      ax :: `matplotlib.axes._subplots.AxesSubplot`
        The matplotlib axis object on which to add the labels
      xax :: str (optional, default: None)
        The label for the x-axis
      yax :: str (optional, default: None)
        The label for the y-axis
      title :: str (optional, default: None)
        The title for the plot
      aspect :: str (optional, default: "equal")
        The aspect of the axis (i.e., the ratio of the y-unit to x-unit)
      grid :: bool (optional, default: False)
        If True, overlay a grid on the plot
      fig :: `matplotlib.figure.Figure` (optional, default: None)
        The figure object to save to figname
      figname :: str (optional, default: None)
        The name/path to the file to save the plot
      dpi :: int (optional, default: 300)
        The resolution of the saved plot. Ignored for vector-based
        graphics (e.g., PDFs)
      tick_params :: dict (optional, default: None)
        Keyworded arguments to adjust the tickl parameters. If None, will set ticks
        to point outside with minor ticks on.

    Returns: None
    """
    if tick_params is None:
        tick_params = {"which": "both", "direction": "out"}
    ax.set_aspect(aspect)
    ax.grid(grid)
    ax.tick_params(**tick_params)
    ax.set_xlabel(xax) if xax is not None else None
    ax.set_ylabel(yax) if yax is not None else None
    ax.set_title(title) if title is not None else None
    if figname is not None:
        if fig is not None:
            fig.savefig(figname, bbox_inches="tight", dpi=dpi)
        else:
            raise AttributeError("fig must be given if figname is given")


def plot_img_log(
    imgdata,
    imgheader,
    dist,
    cmap="magma",
    vmin=None,
    vmax=None,
    midpoint=None,
    interp="none",
    scale_lbl="1 kpc",
    scale_loc="lower right",
    scale_color="k",
    scalebar_factor=1,
    scale_size_vertical=0.5,
    scale_pad=1,
    scale_fontsize=12,
    fig_title=None,
    fig_yax=None,
    fig_xax=None,
    fig_cbar=None,
    figname=None,
    dpi=300,
    nan_color="lowest",
    **imshow_kwargs
):
    """
    Really clunky wrapper for convenience. Plots an image with a spatial scale bar using
    `imshow` + a log_10 colour bar scale. Also optionally saves the figure if given a
    figname. May not use in the future since it is so clunky.

    FIXME: add **scale_kwargs

    Parameters:
      imgdata :: 2D array
        An array with the pixel values from the .fits file
      imgheader :: `astropy.io.fits.header.Header`
        The header to the .fits file
      dist :: `astropy.units.quantity.Quantity` scalar
        The distance to the object
      cmap :: str (optional, default: "magma")
        Colour map
      vmin, vmax, midpoint :: float (optional, defaults: None)
        The minimum, maximum, and midpoint of the colour bar. If None,
        use the default values determined by matplotlib
      interp :: str (optional, default: "none")
        Interpolation for `imshow` (e.g., "none", "antiasliased", "bicubic")
      scale_lbl, scale_loc, scale_color :: str (optional,
                                                defaults: "1 kpc", "lower right", "k")
        The label, location, and colour of the scale bar
      scalebar_factor :: float (optional, default: 1)
        Factor by which to multiply the 1 kpc scale bar.
      scale_size_vertical :: float (optional, default: 0.5)
        Vertical length of the scale bar (in axis units)
      scale_pad :: float or int (optional, default: 1)
        Padding around scale bar and label (in fractions of the font size)
      scale_fontsize :: str or int (optional, default: 12)
        The font size for the scale bar label
      fig_title, fig_yax, fig_xax, fig_cbar :: str (optional, defaults: None)
        The title of the plot, y-axis label, x-axis label, and colourbar label.
        If None, do not write labels (though astropy may automatically add coordinate labels)
      figname :: str (optional, default: None)
        Name of the file to save (e.g. "img.pdf", "img_2.png"). By default, it will save the file
        in the directory from which the Python file is called or where the iPython notebook is located.
        If None, do not save the image.
      dpi :: int (optional, default: 300)
        The dpi of the saved image. Relevant for raster graphics only (i.e., not PDFs).
      nan_color :: str (optional, default: "lowest")
        The colour to use for NaN pixels. If "lowest", map NaNs to the lowest colour in
        the colour bar. If None, use default behaviour of imshow (nan_color="white").
      **imshow_kwargs :: dict (optional)
          Keyworded arguments to pass to `matplotlib.image.AxesImage`

      Returns: None
    """
    imgwcs = WCS(imgheader)
    #
    # Remove background pixels
    #
    # imgdata[imgdata <= 0] = np.nan  # or not. May look bad
    # imgdata[imgdata <= 0] = np.min(imgdata[imgdata > 0])
    imgdata[imgdata <= 0] = 1e-15
    #
    # Set colour map
    #
    # cmap = mpl.cm.get_cmap(cmap, levels)
    cmap = copy.copy(mpl.cm.get_cmap(cmap))
    #
    # Set NaN pixel colour
    #
    if nan_color == "lowest":
        nan_color = cmap(0)
    elif nan_color is None:
        nan_color = "white"
    cmap.set_bad(nan_color)
    #
    # Plot
    #
    fig, ax = plt.subplots(subplot_kw={"projection": imgwcs})
    if midpoint is None:
        img = ax.imshow(
            imgdata,
            cmap=cmap,
            norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
            interpolation=interp,
            **imshow_kwargs
        )
    elif midpoint > 0:
        img = ax.imshow(
            imgdata,
            cmap=cmap,
            norm=MidPointLogNorm(vmin=vmin, vmax=vmax, midpoint=midpoint),
            interpolation=interp,
            **imshow_kwargs
        )
    else:
        raise ValueError("midpoint should be > 0 and between vmin & vmax")
    add_scalebar(
        ax,
        imgwcs,
        dist,
        scalebar_factor=scalebar_factor,
        label=scale_lbl,
        color=scale_color,
        loc=scale_loc,
        size_vertical=scale_size_vertical,
        pad=scale_pad,
        fontsize=scale_fontsize,
    )
    add_cbar(fig, img, label=fig_cbar)
    add_plot_labels(
        ax, xax=fig_xax, yax=fig_yax, title=fig_title, fig=fig, figname=figname, dpi=dpi
    )
    plt.show()


def plot_img_linear(
    imgdata,
    imgheader,
    dist,
    cmap="magma",
    vmin=None,
    vmax=None,
    midpoint=0.5,
    interp="none",
    scale_lbl="1 kpc",
    scale_loc="lower right",
    scale_color="k",
    scalebar_factor=1,
    scale_size_vertical=0.5,
    scale_pad=1,
    scale_fontsize=12,
    fig_title=None,
    fig_yax=None,
    fig_xax=None,
    fig_cbar=None,
    figname=None,
    dpi=300,
    nan_color="lowest",
    **imshow_kwargs
):
    """
    Really clunky wrapper for convenience. Plots an image with a spatial scale bar using
    `imshow` + a linear colour bar scale. Also optionally saves the figure if given a
    figname. May not use in the future since it is so clunky.

    FIXME: add **scale_kwargs

    Parameters:
      imgdata :: 2D array
        An array with the pixel values from the .fits file
      imgheader :: `astropy.io.fits.header.Header`
        The header to the .fits file
      dist :: `astropy.units.quantity.Quantity` scalar
        The distance to the object
      cmap :: str (optional, default: "magma")
        Colour map
      vmin, vmax, midpoint :: float (optional, defaults: None)
        The minimum, maximum of the colour bar. If None,
        use the default values determined by matplotlib
      midpoint :: float (optional, default: 0.5)
        The midpoint of the colourbar. Must be between 0.0 and 1.0.
      interp :: str (optional, default: "none")
        Interpolation for `imshow` (e.g., "none", "antiasliased", "bicubic")
      scale_lbl, scale_loc, scale_color :: str (optional,
                                                defaults: "1 kpc", "lower right", "k")
        The label, location, and colour of the scale bar
      scalebar_factor :: float (optional, default: 1)
        Factor by which to multiply the 1 kpc scale bar.
      scale_size_vertical :: float (optional, default: 0.5)
        Vertical length of the scale bar (in axis units)
      scale_pad :: float or int (optional, default: 1)
        Padding around scale bar and label (in fractions of the font size)
      scale_fontsize :: str or int (optional, default: 12)
        The font size for the scale bar label
      fig_title, fig_yax, fig_xax, fig_cbar :: str (optional, defaults: None)
        The title of the plot, y-axis label, x-axis label, and colourbar label.
        If None, do not write labels (though astropy may automatically add coordinate labels)
      figname :: str (optional, default: None)
        Name of the file to save (e.g. "img.pdf", "img_2.png"). By default, it will save the file
        in the directory from which the Python file is called or where the iPython notebook is located.
        If None, do not save the image.
      dpi :: int (optional, default: 300)
        The dpi of the saved image. Relevant for raster graphics only (i.e., not PDFs).
      nan_color :: str (optional, default: "lowest")
        The colour to use for NaN pixels. If "lowest", map NaNs to the lowest colour in
        the colour bar. If None, use default behaviour of imshow (nan_color="white").
      **imshow_kwargs :: dict (optional)
          Keyworded arguments to pass to `matplotlib.image.AxesImage`

      Returns: None
    """
    imgwcs = WCS(imgheader)
    #
    # Set colour map
    #
    cmap = mpl.cm.get_cmap(cmap)
    cmap = copy.copy(shiftedColorMap(cmap, midpoint=midpoint))
    #
    # Set NaN pixel colour
    #
    if nan_color == "lowest":
        nan_color = cmap(0)
    elif nan_color is None:
        nan_color = "white"
    cmap.set_bad(nan_color)
    #
    # Plot
    #
    fig, ax = plt.subplots(subplot_kw={"projection": imgwcs})
    img = ax.imshow(
        imgdata, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interp, **imshow_kwargs
    )
    add_scalebar(
        ax,
        imgwcs,
        dist,
        scalebar_factor=scalebar_factor,
        label=scale_lbl,
        color=scale_color,
        loc=scale_loc,
        size_vertical=scale_size_vertical,
        pad=scale_pad,
        fontsize=scale_fontsize,
    )
    add_cbar(fig, img, label=fig_cbar)
    add_plot_labels(
        ax, xax=fig_xax, yax=fig_yax, title=fig_title, fig=fig, figname=figname, dpi=dpi
    )
    plt.show()


def cutout(data, center, shape, wcs=None, header=None, mode="trim"):
    """
    Convenience wrapper for astropy's Cutout2D class.

    Parameters:
      data :: 2D array
        The data from which to extract the cutout
      center :: 2-tuple of int or `astropy.coordinates.SkyCoord`
        The (x,y) center of the cutout
      shape :: 2-tuple of int or `astropy.units.Quantity`
        The (vertical, horizontal) dimensions of the cutout
      wcs :: `astropy.wcs.WCS` (optional, default: None)
        The WCS of the data. If not None, will return a copy of the updated WCS for the
        cutout data array
      header :: `astropy.io.fits.header.Header` (optional, default: None)
        The header to the .fits file used to extract WCS data. If wcs and header provided,
        wcs parameter will take precedence
      mode :: str (optional, default: "trim")
        The Cutout2D mode ("trim", "partial", or "strict")

    Returns: data_cut, wcs_cut
      data_cut :: 2D array
        The array cutout from the original data array
      wcs_cut :: `astropy.wcs.WCS` or None
        If WCS initially provided, this is the updated WCS for the cutout data array.
        If WCS initially not provided, this is None.
    """
    if wcs is not None and header is not None:
        warnings.warn("wcs and header both provided. Will use wcs for Cutout2D")
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
    inside the extent of input_arr.

    Parameters:
      input_arr :: 2D array
        The array from which the cutout will be extracted
      input_wcs :: `astropy.wcs.WCS`
        The WCS of the input_arr
      target_arr :: 2D array
        The array whose extent you want to copy
      target_wcs :: `astropy.wcs.WCS`
        The WCS of the target_arr

    Returns: input_arr_cut, input_wcs_cut
      input_data_cut :: 2D array
        The array cut out from the original input_arr
      input_wcs_cut :: `astropy.wcs.WCS` or None
        The updated WCS for the cutout array
    """
    target_bottomleft = target_wcs.pixel_to_world(0, 0)
    target_topright = target_wcs.pixel_to_world(*target_arr.shape)
    # ? Don't know if I am passing the arguments in the right order.
    target_centre = target_wcs.pixel_to_world(
        target_arr.shape[1] / 2, target_arr.shape[0] / 2
    )
    # Map the pixels above to their corresponding pixels in the input array
    input_bottomleft = input_wcs.world_to_pixel(target_bottomleft)
    input_topright = input_wcs.world_to_pixel(target_topright)
    # Determine shape of cutout in input array
    cutout_shape = abs(np.subtract(input_topright, input_bottomleft))
    return cutout(input_arr, target_centre, cutout_shape, wcs=input_wcs, mode=mode)


def line_profile(data, start, end, wcs=None):
    """
    Returns the profile of some 2D data along a line specified by the start and end
    points. Uses very rough nearest-neighbour sampling.

    Parameters:
      data :: 2D array
        The data to be profiled
      start :: 2-tuple of int or `astropy.coordinates.SkyCoord` object
        If tuple, (x, y) pixel coordinates of the start point. If SkyCoord, the WCS (e.g.,
        from the header using astropy.wcs.WCS(header)) must also be provided using the wcs
        parameter. N.B. converting pixel coordinates to SkyCoords and back to pixel
        coordinates (e.g., using astropy functions) may not give exactly the same
        coordinates
      end :: 2-tuple of int or `astropy.coordinates.SkyCoord` object
        If tuple, (x, y) pixel coordinates of the end point. If SkyCoord, the WCS (e.g.,
        from the header using astropy.wcs.WCS(header)) must also be provided using the wcs
        parameter. N.B. converting pixel coordinates to SkyCoords and back to pixel
        coordinates (e.g., using astropy functions) may not give exactly the same
        coordinates
      wcs :: `astropy.wcs.WCS` object (optional, default: None)
        The world coordinate system to transform SkyCoord objects to pixel coordinates

    Returns: profile
      profile :: 1D array
        The line profile of the data
    """
    if isinstance(start, coord.SkyCoord):
        start = skycoord_to_pixel(start, wcs=wcs)
    if isinstance(end, coord.SkyCoord):
        end = skycoord_to_pixel(end, wcs=wcs)
    x0, y0 = start
    x1, y1 = end
    length = int(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
    x_idx = np.linspace(x0, x1, length).astype(int)
    y_idx = np.linspace(y0, y1, length).astype(int)
    profile = data[y_idx, x_idx]  # array indexing is [row, col]
    return profile


def line_profile_idx(data, start, end, wcs=None, extend=False):
    """
    Returns the profile of some 2D data along a line specified by the start and end
    points. Also returns the x and y pixel indices of the line. Uses
    very rough nearest-neighbour sampling.

    Parameters:
      data :: 2D array
        The data to be profiled
      start :: 2-tuple of int or `astropy.coordinates.SkyCoord` object
        If tuple, (x, y) pixel coordinates of the start point. If SkyCoord, the WCS (e.g.,
        from the header using astropy.wcs.WCS(header)) must also be provided using the wcs
        parameter. N.B. converting pixel coordinates to SkyCoords and back to pixel
        coordinates (e.g., using astropy functions) may not give exactly the same
        coordinates
      end :: 2-tuple of int or `astropy.coordinates.SkyCoord` object
        If tuple, (x, y) pixel coordinates of the end point. If SkyCoord, the WCS (e.g.,
        from the header using astropy.wcs.WCS(header)) must also be provided using the wcs
        parameter. N.B. converting pixel coordinates to SkyCoords and back to pixel
        coordinates (e.g., using astropy functions) may not give exactly the same
        coordinates because skycoord_to_pixel() returns floats
      wcs :: `astropy.wcs.WCS` object (optional, default: None)
        The world coordinate system to transform SkyCoord objects to pixel coordinates
      extend :: bool
        If True, extend the line defined by the start and end points to the edges of the
        image. If False, the line profile is only evaluated between the start and end
        points. Note that, because the coordinates are all integers, extending the line to
        the edge may not be exactly coincident with the line segment defined by the start
        and end points and may not go exactly to the edges. The final start and end points
        can be found using the returned idx arrays. That is, start = (x_idx[0], y_idx[0])
        and end = (x_idx[-1], y_idx[-1])

    ! FIXME: the extend parameter is not working as intended

    Returns: profile
      profile :: 1D array
        The line profile of the data
      x_idx, y_idx :: 1D arrays
        The pixel coordinates of the line
    """
    if isinstance(start, coord.SkyCoord):
        start = skycoord_to_pixel(start, wcs=wcs)
        start = (int(start[0]), int(start[1]))
    if isinstance(end, coord.SkyCoord):
        end = skycoord_to_pixel(end, wcs=wcs)
        end = (int(end[0]), int(end[1]))
    if list(map(type, start)) != [int, int] or list(map(type, end)) != [int, int]:
        raise ValueError("start and end must be 2-tuples of ints or SkyCoord objects")
    if start == end:
        raise ValueError("start and end points must be different")
    x0, y0 = start
    x1, y1 = end
    length = int(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
    if extend:
        unit_x = (x1 - x0) / length
        unit_y = (y1 - y0) / length
        # Find number of subtractions to get to edges of image
        # FIXME: bug somewhere here
        # (probably the -1 + factor combination for negative slope)
        if unit_x != 0:  # ! Just a quick band-aid for now. Will clean up
            start_to_left = x0 / unit_x
            start_to_right = (np.shape(data)[1] - 1 - x0) / unit_x  # data indices: [row, col]
            end_to_left = x1 / unit_x
            end_to_right = (np.shape(data)[1] - 1 - x1) / unit_x
        else:
            start_to_left = start_to_right = end_to_left = end_to_right = np.nan
        if unit_y != 0:
            start_to_bottom = y0 / unit_y
            start_to_top = (np.shape(data)[0] - 1 - y0) / unit_y  # data indices: [row, col]
            end_to_bottom = y1 / unit_y
            end_to_top = (np.shape(data)[0] - 1 - y1) / unit_y
        else:
            start_to_bottom = start_to_top = end_to_bottom = end_to_top = np.nan
            #
            dist_to_bottom = np.nanmin([start_to_bottom, end_to_bottom])
            dist_to_left = np.nanmin([start_to_left, end_to_left])
            dist_to_top = np.nanmin([start_to_top, end_to_top])
            dist_to_right = np.nanmin([start_to_right, end_to_right])
        #
        slope = unit_y / unit_x if unit_x != 0 else np.nan
        if slope > 0:  # positive slope
            print("Positive slope")
            # Find distance to bottom or left edge, whichever is closer
            dist_to_bottomleft = np.min([dist_to_bottom, dist_to_left])
            if dist_to_bottomleft == dist_to_bottom:
                diffy = int(dist_to_bottomleft * unit_y)
                diffx = diffy * unit_x
                # if dist_to_bottom == start_to_bottom:
                #     y0 -= diff  # should now be 0
                #     x0 -= diff
                # else:
                #     y1 -= diff  # should now be 0
                #     x1 -= diff
            else:
                diffx = int(dist_to_bottomleft * unit_x)
                diffy = diffx * unit_y
                # if dist_to_left == start_to_left:
                #     x0 -= diff  # should now be 0
                #     y0 -= diff
                # else:
                #     x1 -= diff  # should now be 0
                #     y1 -= diff
            # Move point to bottom or left adge
            if dist_to_bottomleft in [start_to_left, start_to_bottom]:
                    x0 -= diffx
                    y0 -= diffy
                    moved_start = True
            else:
                    x1 -= diffx
                    y1 -= diffy
                    moved_start = False
            # Find distance to top or right edge, whichever is closer
            dist_to_topright = np.min([dist_to_top, dist_to_right])
            if dist_to_topright == dist_to_top:
                diffy = int(dist_to_topright * unit_y)
                diffx = diffy * unit_x
            else:
                diffx = int(dist_to_topright * unit_x)
                diffy = diffx * unit_y
            # Move the remaining point to the top or right edge
            if moved_start:
                x1 += diffx
                y1 += diffy
            else:
                x0 += diffx
                y0 += diffy
        elif slope < 0:  # negative slope
            print("Negative slope")
            print("STILL VERY BUGGY!")
            # Find distance to top or left edge, whichever is closer
            dist_to_topleft = np.min([dist_to_top, dist_to_left])
            if dist_to_topleft == dist_to_top:
                diffy = int(dist_to_topleft * unit_y)
                diffx = diffy * unit_x
                factor = -1  # to convert - to + later
            else:
                diffx = int(dist_to_topleft * unit_x)
                diffy = diffx * unit_y
                factor = 1
            # Move point to top or left edge
            if dist_to_topleft in [start_to_left, start_to_top]:
                x0 -= diffx * factor
                y0 -= diffy * factor
                moved_start = True
            else:
                x1 -= diffx * factor
                y1 -= diffy * factor
                moved_start = False
            # Find distance to bottom or right edge, whichever is closer
            dist_to_bottomright = np.min([dist_to_bottom, dist_to_right])
            if dist_to_bottomright == dist_to_bottom:
                diffy = int(dist_to_bottomright * unit_y)
                diffx = diffy * unit_x
                factor = -1  # to convert + to - later
            else:
                diffx = int(dist_to_bottomright * unit_x)
                diffy = diffx * unit_y
                factor = 1
            # Move the remaining point to the bottom or right edge
            if moved_start:
                x1 += diffx * factor
                y1 += diffy * factor
            else:
                x0 += diffx * factor
                y0 += diffy * factor
        elif slope == 0:  # zero (horizontal) slope
            print("Zero slope")
            # Extend to left/right edges
            x0 = 0 if start_to_left < start_to_right else np.shape(data)[1] - 1
            x1 = np.shape(data)[1] - 1 if end_to_right < end_to_left else 0
        else:  # vertical slope
            # FIXME: sometimes returns size 0 (almost definitely the inequalities & NaNs)
            print("Vertical slope")
            # Extend to bottom/top edges
            y0 = 0 if start_to_bottom < start_to_top else np.shape(data)[0] - 1
            y1 = np.shape(data)[0] - 1 if end_to_top < end_to_bottom else 0
        print(x0, y0, x1, y1)
        # Re-evaluate length
        length = int(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
    x_idx = np.linspace(x0, x1, length).astype(int)
    y_idx = np.linspace(y0, y1, length).astype(int)
    profile = data[y_idx, x_idx]  # array indexing is [row, col]
    return profile, x_idx, y_idx


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


def optimal_sn(index, signal, noise):
    """
    Signal-to-noise ratio approximation using optimal weighing of pixels.

    See Eq. (3) of Cappellari & Copin (2003):
    https://ui.adsabs.harvard.edu/abs/2003MNRAS.342..345C/abstract

    Parameters: (nearly verbatim from Cappellari & Copin's voronoi_2d_binning.py)
      index :: 1D array
        Integer vector of length N containing the indices of
        the spaxels for which the combined S/N has to be returned.
        The indices refer to elements of the vectors signal and noise.
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


def prelim_bin(signal, noise, block_size=(4, 4), print_info=True, func=np.nansum):
    """
    Regular preliminary binning of 2D data (e.g., for Voronoi binning). Ignores NaNs.

    (From astropy's block_reduce() documentation): If the data are not perfectly divisible
    by block_size along a given axis, then the data will be trimmed (from the end) along
    that axis.

    TODO: finish docstring
    """
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
        print("x_coords, y_coords, signal_binned, noise_binned shapes:",
              x_coords.shape, y_coords.shape, signal_binned.shape, noise_binned.shape)
        print("total bad elements (infs/NaNs):", np.sum(~is_good))
    #
    return signal_binned, noise_binned, x_coords, y_coords, is_good


def get_reproject_shape_factor(target_arr, input_wcs, target_wcs):
    #
    # Determine reprojection shape and binning factor
    #   N.B. reprojection shape should be as close as possible to the shape of a regular
    #   "cut out" if the input array was cut at the boundaries of the target.
    #
    # Find the coordinates of the target_data's edges, assuming the data are rectangular
    target_bottomleft = target_wcs.pixel_to_world(0, 0)
    target_topright = target_wcs.pixel_to_world(*target_arr.shape)
    # ? Don't know if I am passing the arguments in the right order.
    target_centre = target_wcs.pixel_to_world(
        target_arr.shape[1] / 2, target_arr.shape[0] / 2
    )
    # Map the pixels above to their corresponding pixels in the input array
    input_bottomleft = input_wcs.world_to_pixel(target_bottomleft)
    input_topright = input_wcs.world_to_pixel(target_topright)
    input_centre = input_wcs.world_to_pixel(target_centre)
    # Determine binning/transformation factor
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

    target_wcs must support CDELT keyword.
    TODO: add support for CD[#]_[#] keywords.

    Requires the reproject package: https://reproject.readthedocs.io/en/stable/

    TODO: finish docstring
    """
    #
    # Reproject input array to target array
    #
    wcs_reproject = target_wcs.deepcopy()
    wcs_reproject.wcs.crpix = target_wcs.wcs.crpix * input_to_target_factor
    wcs_reproject.wcs.cdelt = target_wcs.wcs.cdelt / input_to_target_factor
    wcs_reproject.array_shape = input_reproject_shape
    arr_reproject = reproject_func(
        (input_arr, input_wcs),
        wcs_reproject,
        shape_out=input_reproject_shape,
        return_footprint=False,
    )
    return arr_reproject, wcs_reproject


def bin_snr_to_target(
    signal_arr,
    signal_wcs,
    noise_arr,
    noise_wcs,
    target_arr,
    target_wcs,
    reproject_func=reproject.reproject_exact,
    bin_func=np.nansum,
    print_debug=False,
):
    """
    Wrapper for binning an signal & noise arrays to the exact physical dimensions and
    resolution of a target (provided via the target_wcs). The input arrays should already
    be masked (i.e., invalid values should be set to np.nan) and the input arrays should
    entirely contain the target_wcs (i.e., the extent of the target_data).

    Requires the reproject package: https://reproject.readthedocs.io/en/stable/

    target_wcs should support the CDELT keyword.
    TODO: add support for CD[#]_[#] keywords.

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
    noise_reproject, noise_wcs_reproject = reproj_arr(
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
    signal_binned, noise_binned, x_coords, y_coords, is_good = prelim_bin(
        signal_reproject,
        noise_reproject,
        block_size=signal_to_target_factor,
        func=bin_func,
        print_info=print_debug,
    )
    # Modify WCS object
    # ? Not sure if slice argument order is correct
    wcs_binned = signal_wcs_reproject.slice(
        (np.s_[:: signal_to_target_factor[0]], np.s_[:: signal_to_target_factor[1]])
    )
    wcs_binned.wcs.crpix = signal_wcs_reproject.wcs.crpix / signal_to_target_factor
    wcs_binned.wcs.cdelt = signal_wcs_reproject.wcs.cdelt * signal_to_target_factor
    if print_debug:
        print("wcs_binned:", wcs_binned)
    #
    return x_coords, y_coords, signal_binned, noise_binned, is_good, wcs_binned


def plot_snr(
    signal,
    noise,
    wcs,
    contour_arr,
    countour_wcs,
    contour_levels=[0, 5, 10],
    vmin=150,
    vmax=1250,
    band="u-band",
):
    snr = abs(signal / noise)
    fig, ax = plt.subplots(subplot_kw={"projection": wcs})
    img = ax.imshow(snr, cmap="plasma", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(img)
    cbar.ax.tick_params(which="both", direction="out")
    cbar.set_label(f"{band} SNR")
    ax.contour(
        contour_arr,
        transform=ax.get_transform(countour_wcs),
        levels=contour_levels,
        colors="w",
    )
    ax.set_title(f"VCC 792 / NGC 4380: {band} Data Binned")
    ax.set_xlabel("RA (J2000)")
    ax.set_ylabel("Dec (J2000)")
    ax.grid(False)
    ax.set_aspect("equal")
    plt.show()


def dill_data(
    outfile,
    reproject_method,
    dist_pc,
    dist_pc_err,
    x_coords,
    y_coords,
    signal_binned,
    noise_binned,
    abs_mag,
    abs_mag_err_withDistErr,
    abs_mag_err_noDistErr,
    is_good,
    wcs_binned,
    wcs_binned_array_shape,
):
    with open(outfile, "wb") as f:
        dill.dump(
            {
                "note": "Remember to set `wcs_binned.array_shape = wcs_binned_array_shape`",
                "reproject_method": reproject_method,
                "dist_pc": dist_pc,
                "dist_pc_err": dist_pc_err,
                "x_coords": x_coords,
                "y_coords": y_coords,
                "signal_binned": signal_binned,
                "noise_binned": noise_binned,
                "abs_mag": abs_mag,
                "abs_mag_err_withDistErr": abs_mag_err_withDistErr,
                "abs_mag_err_noDistErr": abs_mag_err_noDistErr,
                "is_good": is_good,
                "wcs_binned": wcs_binned,
                "wcs_binned_array_shape": wcs_binned_array_shape,  # this is the "NAXIS" keyword
            },
            f,
        )
    print(f"Pickled {outfile}")


def txt_snr(
    outfile,
    xs,
    ys,
    u_sig,
    u_noise,
    g_sig,
    g_noise,
    i_sig,
    i_noise,
    z_sig,
    z_noise,
):
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
            "counts_u": u_sig.flatten(),
            "err_u": u_noise.flatten(),
            "counts_g": g_sig.flatten(),
            "err_g": g_noise.flatten(),
            "counts_i": i_sig.flatten(),
            "err_i": i_noise.flatten(),
            "counts_z": z_sig.flatten(),
            "err_z": z_noise.flatten(),
        }
    )
    df.to_csv(path_or_buf=outfile, sep=" ", index=False, header=True)
    print(f"Saved {outfile}")


def txt_mags(
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


def calc_mag(flux, flux_err=0.0, zpt=30.0, calc_abs=False, dist=None, dist_err=0.0):
    """
    Calculates the relative or absolute magnitude of an object given its flux.

    Parameters:
      flux :: array
        The flux of the pixel
      flux_err :: array (optional, default: 0.0)
        The uncertainty in the flux. Must be able to broadcast with flux array
      zpt :: float (optional, default: 30.0)
        The zero point of the magnitude system
      calc_abs :: bool (optional, default: False)
        If True, returns the absolute magnitude, otherwise returns the relative magnitude.
        Requires that dist is also provided.
      dist :: scalar or array (optional, default: None)
        The distance to the object/pixel in parsecs. Must be able to broadcast with flux
        array
      dist_err :: float or array (optional, default: 0.0)
        The uncertainty in the distance. Must be able to broadcast with flux array

    Returns: mag, mag_err
      mag :: array
        The magnitude of the pixel
      mag_err :: array
        The uncertainty in the magnitude
    """
    rel_mag = -2.5 * np.log10(flux) + zpt
    rel_mag_err = 2.5 / np.log(10) * abs(flux_err / flux)
    #
    if calc_abs:
        if dist is None:
            raise ValueError("dist must be provided if calc_abs is True")
        abs_mag = rel_mag - 5 * (np.log10(dist) - 1)
        abs_mag_err = np.sqrt(rel_mag_err ** 2 + (5 / np.log(10) * dist_err / dist) ** 2)
        return abs_mag, abs_mag_err
    #
    return rel_mag, rel_mag_err


def plot_mags(
    mag,
    mag_err,
    wcs,
    contour_arr,
    countour_wcs,
    contour_levels=[0, 5, 10],
    vmin=None,
    vmax=None,
    band="u-band",
):
    fig, ax = plt.subplots(subplot_kw={"projection": wcs})
    img = ax.imshow(mag, cmap="plasma", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(img)
    cbar.ax.tick_params(which="both", direction="out")
    cbar.set_label(f"{band} Magnitudes")
    ax.contour(
        contour_arr,
        transform=ax.get_transform(countour_wcs),
        levels=contour_levels,
        colors="w",
    )
    ax.set_title(f"VCC 792 / NGC 4380: {band} Data Binned")
    ax.set_xlabel("RA (J2000)")
    ax.set_ylabel("Dec (J2000)")
    ax.grid(False)
    ax.set_aspect("equal")
    plt.show()
    #
    fig, ax = plt.subplots(subplot_kw={"projection": wcs})
    img = ax.imshow(mag_err, cmap="plasma", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(img)
    cbar.ax.tick_params(which="both", direction="out")
    cbar.set_label(f"{band} Magnitude Uncertainties")
    ax.contour(
        contour_arr,
        transform=ax.get_transform(countour_wcs),
        levels=contour_levels,
        colors="w",
    )
    ax.set_title(f"VCC 792 / NGC 4380: {band} Data Binned")
    ax.set_xlabel("RA (J2000)")
    ax.set_ylabel("Dec (J2000)")
    ax.grid(False)
    ax.set_aspect("equal")
    plt.show()


def calc_pixel_size(imgwcs, dist, dist_err=None):
    """
    Calculates the physical size of each pixel in parsecs per pixel dimension.

    Parameters:
      imgwcs :: `astropy.wcs.wcs.WCS`
        The WCS coordinates of the .fits file
      dist :: `astropy.units.quantity.Quantity` scalar
        The distance to the object
      dist_err :: `astropy.units.quantity.Quantity` scalar (optional, default: None)
        The uncertainty in the distance to the object

    Returns: pc_per_px, pc_per_px_err
      pc_per_px :: 1D array
        The spatial resolution of the image in parsecs per pixel (along each axis)
      pc_per_px_err :: 1D array or None
        The uncertainty in the spatial resolution of the image (along each axis)
    """
    arcsec_per_px = (proj_plane_pixel_scales(imgwcs.celestial) * u.deg).to(u.arcsec)
    # Still deciding whether to use arctan for the next line
    arcsec_per_pc = np.rad2deg(1 * u.pc / dist.to(u.pc) * u.rad).to(u.arcsec)
    pc_per_px = arcsec_per_px / arcsec_per_pc
    #
    if dist_err is not None:
        # arcsec_per_pc_err = np.rad2deg(1 * u.pc / dist_err.to(u.pc) * u.rad).to(u.arcsec)
        # pc_per_px_err = arcsec_per_px / arcsec_per_pc_err
        # Uncertainty transforms linearly
        pc_per_px_err = pc_per_px * dist_err.to(u.pc) / dist.to(u.pc)
        return pc_per_px.value, pc_per_px_err.value
    return pc_per_px.value, None


def lognorm_median(r_data, g_data, b_data, a=1000, norm_factor=1000):
    """
    Normalize an image using median values on a (natural) log scale. Requires
    astropy.visualization.LogStretch().

    The data are scaled using the following formula:
                y = ln(a * x + 1) / ln(a + 1)
    where a is a scalar parameter and x are the data to be transformed.

    To create an RGB image, simply pass the returned array to imshow like:
                ax.imshow(rgb_data, interpolation="none")

    Parameters:
      r_data :: 2D array or `astropy.io.fits.ImageHDU` object
        The data of shape (N, M) to be mapped to red
      g_data :: 2D array or `astropy.io.fits.ImageHDU` object
        The data of shape (N, M) to be mapped to green
      b_data :: 2D array or `astropy.io.fits.ImageHDU` object
        The data of shape (N, M) to be mapped to blue
      a :: float (optional, default: 1000)
        The scaling factor for the log transform. Must be greater than 0.
      norm_factor :: float (optional: default: 1000)
        The normalization factor for the median

    Returns: rgb_data
      rgb_data :: 3D array of shape (N, M, 3)
        The transformed data to be mapped to red (rgb_data[:,:,0]),
        green (rgb_data[:,:,1]), and blue (rgb_data[:,:,2]).
    """
    rgb_data = []
    for image in (r_data, g_data, b_data):
        median = np.median(image.data)
        # Define transformation
        T = astropy.visualization.LogStretch(a=a)
        # Normalize by median and apply transformation
        image = image / median / norm_factor
        rgb_data.append(T(image))
    return np.dstack(rgb_data)


def calc_snr(signal, noise, func=np.nansum):
    """
    Calculates the overall signal-to-noise ratio of the input arrays.

    TODO: finish docstring

    Returns: snr, bin_signal, bin_noise
      snr :: float
        The total signal-to-noise ratio of the input data
    """
    bin_signal = func(signal)
    # Add noise in quadrature
    bin_noise = noise * noise
    bin_noise = func(bin_noise)
    bin_noise = np.sqrt(bin_noise)
    return bin_signal / bin_noise, bin_signal, bin_noise


# def load_sed_results(
#     path,
#     xmax,
#     ymax,
#     outfile=None,
#     xmin=0,
#     ymin=0,
#     infile="prospectorFit_emcee_<NUMS>_results.txt",
#     replace="<NUMS>",
# ):
#     """
#     Takes the individual .txt files containing the SED fitting results and compiles the
#     data into an array. Optionally outputs a .txt file containing the amalgamated results
#     (not implemented yet).
#     """
#     # Check directory and filename
#     if path[-1] != "/":
#         path += "/"
#     if infile[-4:] != ".txt":
#         raise ValueError("infile must be a .txt file.")
#     #
#     start_idx = infile.index(replace)
#     end_idx = start_idx + len(replace)
#     pre_str = infile[:start_idx]
#     post_str = infile[end_idx:]
#     len_digits = len(str(np.max((np.max(xmax), np.max(ymax)))))
#     results = []
#     for y in range(ymin, ymax + 1):
#         for x in range(xmin, xmax + 1):
#             results.append(
#                 np.loadtxt(
#                     path
#                     + pre_str
#                     + str(x).zfill(len_digits)
#                     + str(y).zfill(len_digits)
#                     + post_str
#                 )
#             )
#     results = np.vstack(results)
#     return results


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

    TODO: finish docstring

    Parameters:
      skip :: array-like of ints (optional)
        The files to skip and assign to NaN.
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
                np.loadtxt(path + pre_str + str(i).zfill(len_digits) + post_str)
            )
    results = np.vstack(results)
    return results


def joint_contour_plot(
    xdata,
    ydata,
    plot_lts=True,
    lts_slope=None,
    lts_yint=None,
    lts_pivot=None,
    lts_rms=None,
    lts_clip=None,
    lts_xlim=None,
    lts_color=sns.color_palette("colorblind")[5],
    fig_xlim=None,
    fig_ylim=None,
    fig_xlabel=None,
    fig_ylabel=None,
    fig_suptitle=None,
    fig_savename=None,
    contour_cmap=sns.color_palette("ch:start=0.5, rot=-0.5", as_cmap=True),
    margin_color="#66c2a5",
    margin_bins=100,
    plot_scatter=False,
    scatter_kwargs=None,
):
    """
    Creates a joint contour plot of the data with marginal plots showing the histogram and
    KDE of each axis' data. Optionally plots a line showing the LTS fit and/or a scatter
    plot superimposed over the contours.

    Parameters:
      xdata, ydata :: 1D array
        The x- and y-axis data.
      plot_lts :: bool (optional)
        If True, plots the LTS best-fit line and requires at least the lts_slope,
        lts_yint, and lts_pivot. If False, do not plot the LTS line.
      lts_slope, lts_yint, lts_pivot :: float (optional)
        The slope, y-intercept, and pivot of the LTS best-fit line. Required if plot_lts
        is True.
      lts_rms, lts_clip :: float (optional)
        The RMS uncertainty and clipping value of the LTS fit.
      lts_xlim :: 2-tuple of floats (optional)
        The x-axis limits of the LTS fit line. If None, lts_xlim uses the min & max of the
        xdata.
      lts_color :: str (optional)
        The color of the LTS-fitted line(s).
      fig_xlim, fig_ylim :: 2-tuple of floats or dict of kwargs (optional)
        The x- and y-axis limits of the plot. If None, fig_xlim and fig_ylim use the
        default values determined by matplotlib
      fig_xlabel, fig_ylabel :: str (optional)
        The x- and y-axis labels
      fig_suptitle :: str (optional)
        The suptitle of the joint plot
      fig_savename :: str (optional)
        If not None, saves the figure to the directory/filename specified by fig_savename.
      contour_cmap :: str or `matplotlib.colors.ListedColormap` object (optional)
        The colour map to use for the contour plot.
      margin_color :: str (optional)
        The colour to use for the marginal plots' histograms and KDEs.
      margin_bins :: int (optional)
        The number of bins to use for the marginal plots' histograms.
      plot_scatter :: bool (optional)
        If True, plots a scatter plot of the data without errorbars.
      scatter_kwargs :: dict (optional)
        The keyword arguments to pass to the scatter plot. If None, sets the marker style
        to points ("."), marker size to 4, and marker colour to black.

    Returns: None
    """
    grid = mpl.gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 4])
    fig = plt.figure()
    #
    # Primary plot
    #
    ax = plt.subplot(grid[1, 0])
    # Plot KDE contours
    sns.kdeplot(
        ax=ax, x=xdata, y=ydata, cmap=contour_cmap, fill=True,
    )
    # Plot LTS line (slope, yint, rms, clip, and pivot from LTS fit)
    if plot_lts:
        if lts_xlim is not None:
            xvals = np.linspace(*lts_xlim, 100)
        else:
            xvals = np.linspace(np.min(xdata), np.max(xdata), 100)
        if lts_slope is not None and lts_yint is not None and lts_pivot is not None:
            # Plot best-fit line
            linevals = lts_slope * (xvals - lts_pivot) + lts_yint
            ax.plot(xvals, linevals, color=lts_color)
        else:
            raise ValueError("LTS slope and y-int must be provided")
        if lts_rms is not None:
            # Plot lines enclosing 68% of data
            ax.plot(xvals, linevals - lts_rms, color=lts_color, ls="--")
            ax.plot(xvals, linevals + lts_rms, color=lts_color, ls="--")
            if lts_clip is not None:
                # If lts_clip = 2.6 sigma, lines enclose 99% of data
                ax.plot(xvals, linevals - lts_clip * lts_rms, color=lts_color, ls=":")
                ax.plot(xvals, linevals + lts_clip * lts_rms, color=lts_color, ls=":")
    # Plot scatter plot
    if plot_scatter:
        if scatter_kwargs is None:
            scatter_kwargs = {"c": "k", "s": 4, "marker": "."}
        ax.scatter(xdata, ydata, **scatter_kwargs)
    if isinstance(fig_xlim, tuple):
        ax.set_xlim(*fig_xlim)
    elif isinstance(fig_xlim, dict):
        ax.set_xlim(**fig_xlim)
    if isinstance(fig_ylim, tuple):
        ax.set_ylim(*fig_ylim)
    elif isinstance(fig_ylim, dict):
        ax.set_ylim(**fig_ylim)
    ax.set_xlabel(fig_xlabel) if fig_xlabel is not None else None
    ax.set_ylabel(fig_ylabel) if fig_ylabel is not None else None
    ax.grid(True)
    #
    # Right marginal plot
    #
    ax_r = plt.subplot(grid[1, 1], frameon=False, sharey=ax, xticks=[])
    ax_r.hist(ydata, bins=margin_bins, orientation="horizontal", density=True)
    sns.kdeplot(y=ydata, ax=ax_r, fill=False, color=margin_color)
    ax_r.yaxis.set_ticks_position("none")
    plt.setp(ax_r.get_yticklabels(), visible=False)
    ax_r.grid(False)
    #
    # Top marginal plot
    #
    ax_t = plt.subplot(grid[0, 0], frameon=False, sharex=ax, yticks=[])
    ax_t.hist(xdata, bins=margin_bins, orientation="vertical", density=True)
    sns.kdeplot(x=xdata, ax=ax_t, fill=False, color=margin_color)
    ax_t.xaxis.set_ticks_position("none")
    plt.setp(ax_t.get_xticklabels(), visible=False)
    ax_t.grid(False)
    #
    # Other plot params
    #
    fig.suptitle(fig_suptitle) if fig_suptitle is not None else None
    fig.tight_layout()
    plt.subplots_adjust(wspace=2e-3, hspace=4e-3)
    fig.savefig(fig_savename, bbox_inches="tight") if fig_savename is not None else None
    plt.show()


def get_beam_size(header):
    """
    Assumes pixel is square.

    TODO: finish docstring
    """
    beam = Beam.from_fits_header(header)
    # Assume square pixels
    deg_per_px = proj_plane_pixel_scales(WCS(header).celestial)[0] * u.deg
    #
    beam_major = (beam.major.to(u.deg) / deg_per_px).to(u.dimensionless_unscaled).value
    beam_minor = (beam.minor.to(u.deg) / deg_per_px).to(u.dimensionless_unscaled).value
    beam_pa = (beam.pa + 90.0 * u.deg).to(u.deg).value  # PA starts at N and increases CCW
    return beam_major, beam_minor, beam_pa
