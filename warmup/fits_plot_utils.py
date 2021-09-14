"""
fits_plot_utils.py

Utilities for generating plots (e.g., imshow, contour, contourf) from FITS files.
Originally developed for NGVS and VERTICO data processing.

Isaac Cheng - September 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from astropy.wcs.utils import proj_plane_pixel_scales
import copy
import warnings

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

def cutout(data, center, shape, wcs=None, header=None):
    """
    Convenience wrapper for astropy's Cutout2D class.

    Parameters:
      data :: 2D array
        The data from which to extract the cutout
      center :: 2-tuple of int or `astropy.coordinates.SkyCoord`
        The center of the cutout
      shape :: 2-tuple of int or `astropy.units.Quantity`
        The (vertical, horizontal) dimensions of the cutout
      wcs :: `astropy.wcs.WCS` (optional, default: None)
        The WCS of the data. If not None, will return a copy of the updated WCS for the
        cutout data array
      header :: `astropy.io.fits.header.Header` (optional, default: None)
        The header to the .fits file used to extract WCS data. If wcs and header provided,
        wcs parameter will take precedence

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
    cutout = Cutout2D(data, center, shape, wcs=wcs)
    data_cut = cutout.data
    wcs_cut = cutout.wcs
    return data_cut, wcs_cut
