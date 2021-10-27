"""
plot_utils.py

Miscellaneous utilities for plotting with matplotlib. Mostly for my own use.

Isaac Cheng - October 2021
"""

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# My own packages
# import radial_profile_utils as rpu


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
    frameon=False,
    fontsize=None,
    **kwargs,
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
      label :: str (optional)
        The scale bar label
      color :: str (optional)
        Colour of the scale bar and label
      loc :: str or int (optional)
        The location of the scale bar and label
      size_vertical :: float (optional)
        Vertical length of the scale bar (in ax units)
      pad :: float or int (optional)
        Padding around scale bar and label (in fractions of the font size)
      frameon :: bool (optional)
        If True, draw a box around the scale bar and label
      fontsize :: str or int (optional)
        The font size for the label. If None, use the default font size determined by
        matplotlib
      **kwargs :: dict (optional)
        Keyworded arguments to pass to `matplotlib.offsetbox.AnchoredOffsetbox`

    Returns: ax.add_artist(scalebar)
      ax.add_artist(scalebar) :: `mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar`
        Adds the scale bar to the given ax
    """
    # pylint: disable=no-member
    from astropy.wcs.utils import proj_plane_pixel_scales
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    arcsec_per_px = (proj_plane_pixel_scales(imgwcs.celestial)[0] * u.deg).to(u.arcsec)
    # Still deciding whether to use arctan for the next line. Small angle approximation
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
        frameon=frameon,
        size_vertical=size_vertical,
        **kwargs,
    )
    return ax.add_artist(scalebar)


def add_beam(ax, header, xy=(0, 0), **kwargs):
    """
    Adds an ellipse to the given ax to show the radio beam size derived from the FITS
    header.

    Requires my radial_profile_utils package.

    Parameters:
      ax :: `matplotlib.axes._subplots.AxesSubplot` object
        The axis on which to add the beam
      header :: `astropy.io.fits.Header` object
        The header of the image
      xy :: 2-tuple of ints (optional)
        The (x, y) pixel coordinates of the centre of the ellipse/beam
      **kwargs :: dict (optional)
        Keyworded arguments to pass to `matplotlib.patches.Ellipse`. If empty, will set
        the following properties: {"ls": "-", "edgecolor": "k", "fc": "None", "lw": 1,
        "zorder": 2}

    Returns: ax.add_artist(beam)
        ax.add_artist(beam) :: `matplotlib.patches.Ellipse`
          The beam object added on the given axis
    """
    # Check inputs
    if kwargs == {}:
        kwargs = {"ls": "-", "edgecolor": "k", "fc": "None", "lw": 1, "zorder": 2}
    elif "width" in kwargs or "height" in kwargs or "angle" in kwargs:
        raise ValueError(
            "Cannot specify width, height, or angle. These are automatically added!"
        )
    from matplotlib.patches import Ellipse
    from radial_profile_utils import get_beam_size_px

    beam_major, beam_minor, beam_pa = get_beam_size_px(header)
    # Add beam
    beam = Ellipse(
        xy=xy,
        width=beam_major,
        height=beam_minor,
        angle=(beam_pa - 90) % 360.0,  # PA starts at North & increases CCW by convention
        **kwargs,
    )
    return ax.add_artist(beam)


def add_annuli(ax, annuli, **kwargs):
    """
    Adds annuli to the given ax.

    Parameters:
      ax :: `matplotlib.axes._subplots.AxesSubplot` object
        The axis on which to add the beam
      annuli :: array-like containing `photutils.aperture.EllipticalAperture` and/or
                `photutils.aperture.EllipticalAnnulus` objects
      kwargs :: dict (optional)
        Keyworded arguments to pass to `matplotlib.patches.Ellipse`. If empty, will set
        the following properties: {"ls": "-", "edgecolor": "tab:cyan", "fc": "None",
        "lw": 1, "zorder": 2}

    Returns: None
    """
    from matplotlib.patches import Ellipse

    if kwargs == {}:
        kwargs = {"ls": "-", "edgecolor": "tab:cyan", "fc": "None", "lw": 1, "zorder": 2}
    for annulus in annuli[::-1]:  # plot annuli ouside-in
        try:
            # EllipticalAnnulus attributes
            width = annulus.b_out
            height = annulus.a_out
        except AttributeError:
            # EllipticalAperture attributes
            width = annulus.b
            height = annulus.a
        ellipse = Ellipse(
            xy=annulus.positions,
            width=width * 2,  # full major/minor axis
            height=height * 2,  # full major/minor axis
            # PA is 0 deg at North & increases CCW by convention
            angle=(np.rad2deg(annulus.theta) - 90) % 360.0,
            **kwargs,
        )
        ax.add_patch(ellipse)
    return None


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
      a :: float (optional)
        The scaling factor for the log transform. Must be greater than 0.
      norm_factor :: float (optional)
        The normalization factor for the median

    Returns: rgb_data
      rgb_data :: 3D array of shape (N, M, 3)
        The transformed data to be mapped to red (rgb_data[:,:,0]),
        green (rgb_data[:,:,1]), and blue (rgb_data[:,:,2]).
    """
    from astropy.visualization import LogStretch

    rgb_data = []
    for image in (r_data, g_data, b_data):
        median = np.median(image.data)
        # Define transformation
        T = LogStretch(a=a)
        # Normalize by median and apply transformation
        image = image / median / norm_factor
        rgb_data.append(T(image))
    return np.dstack(rgb_data)


def add_lts_line(
    ax,
    lts_slope,
    lts_int,
    lts_pivot,
    lts_rms=None,
    lts_clip=None,
    lts_xlim=None,
    xdata=None,
    lts_color=sns.color_palette("colorblind")[5],
    **kwargs,
):
    """
    Add line(s) to the given ax showing the least-trimmed-squares (LTS) line of best fit.

    Parameters:
      lts_slope, lts_int, lts_pivot :: float (optional)
        The slope, intercept, and pivot of the LTS best-fit line. Required if plot_lts is
        True.
      lts_rms, lts_clip :: float (optional)
        The RMS uncertainty and clipping value of the LTS fit.
      lts_xlim :: 2-tuple of floats (optional)
        The x-axis limits of the LTS fit line. If None, lts_xlim uses the min & max of the
        xdata.
      x-data :: array (optional)
        The x-axis data. Required if lts_xlim is None
      lts_color :: str (optional)
        The color of the LTS-fitted line(s)
      kwargs :: dict (optional)
        Keyworded arguments to pass to `matplotlib.pyplot.plot`

    Returns: None
    """
    if lts_xlim is not None:
        xvals = np.linspace(*lts_xlim, 100)
    elif lts_xlim is None and xdata is not None:
        xvals = np.linspace(np.nanmin(xdata), np.nanmax(xdata), 100)
    else:
        raise ValueError("xdata is required if lts_xlim is None")
    if lts_slope is not None and lts_int is not None and lts_pivot:
        # Plot best-fit line
        linevals = lts_slope * (xvals - lts_pivot) + lts_int
        ax.plot(xvals, linevals, color=lts_color, **kwargs)
    else:
        raise ValueError("LTS slope, y-int, and pivot must be provided")
    if lts_rms is not None:
        # Plot lines enclosing 68% of data
        ax.plot(xvals, linevals - lts_rms, color=lts_color, ls="--", **kwargs)
        ax.plot(xvals, linevals + lts_rms, color=lts_color, ls="--", **kwargs)
        if lts_clip is not None:
            # If lts_clip = 2.6 sigma, lines enclose 99% of data
            ax.plot(
                xvals, linevals - lts_clip * lts_rms, color=lts_color, ls=":", **kwargs
            )
            ax.plot(
                xvals, linevals + lts_clip * lts_rms, color=lts_color, ls=":", **kwargs
            )
    return None


def joint_contour_plot(
    xdata,
    ydata,
    plot_lts=True,
    lts_slope=None,
    lts_int=None,
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
    # ax_aspect=None,  # do not use
    plt_show=True,
):
    """
    Creates a joint contour plot of the data with marginal plots showing the histogram and
    KDE of each axis' data. Optionally plots a line showing the least-trimmed squares
    (LTS) fit and/or a scatter plot superimposed over the contours.

    Parameters:
      xdata, ydata :: 1D array
        The x- and y-axis data.
      plot_lts :: bool (optional)
        If True, plots the LTS best-fit line and requires at least the lts_slope,
        lts_int, and lts_pivot. If False, do not plot the LTS line.
      lts_slope, lts_int, lts_pivot :: float (optional)
        The slope, intercept, and pivot of the LTS best-fit line. Required if plot_lts
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
      ax_aspect :: "auto", "equal", or float (optional)
        DO NOT USE! The aspect of the main plot axes. If None, use the default value
        determined by matplotlib. DO NOT USE!
      plt_show :: bool (optional)
        If True, call plt.show() to draw the plot and return None. If False, do not draw
        the plot (allow adding to figure/axes), do not save the figure, and return figure
        and axes

    Returns: None or (fig, ax, ax_r, ax_t)
      If plt_show is True, returns None. If plt_show is False, returns a tuple containing
      the figure and axes. fig=Figure, ax=main axis, ax_r=right marginal plot axis,
      ax_t=top marginal plot axis
    """
    # pylint: disable=expression-not-assigned
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
        add_lts_line(
            ax,
            lts_slope,
            lts_int,
            lts_pivot,
            lts_rms,
            lts_clip,
            lts_xlim,
            xdata,
            lts_color,
        )
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
    # if ax_aspect is not None:
    #     print("WARNING: Manually setting the axes' aspect is very buggy at the moment")
    #     ax.set_aspect(ax_aspect)
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
    if plt_show:
        if fig_savename is not None:
            fig.savefig(fig_savename, bbox_inches="tight")
            print(f"Saved plot to {fig_savename}!")
        plt.show()
        return None
    else:
        if fig_savename is not None:
            print("Not saving plot. Returning figure and axes instead!")
        return (fig, ax, ax_r, ax_t)


# ------------------------ MISCELLANEOUS FUNCTIONS FOR MY OWN USE ------------------------


def _plot_regBin_results(
    arr,
    wcs,
    contour_arr,
    countour_wcs,
    contour_levels=[0, 5, 10],
    vmin=None,
    vmax=None,
    title=None,
):
    """
    Just to check results make sense (at least visually)
    """
    fig, ax = plt.subplots(subplot_kw={"projection": wcs})
    img = ax.imshow(arr, cmap="plasma", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(img)
    cbar.ax.tick_params(which="both", direction="out")
    ax.contour(
        contour_arr,
        transform=ax.get_transform(countour_wcs),
        levels=contour_levels,
        colors="w",
    )
    ax.set_title(f"{title} Binned") if title is not None else None
    ax.set_xlabel("RA (J2000)")
    ax.set_ylabel("Dec (J2000)")
    ax.grid(False)
    ax.set_aspect("equal")
    plt.show()


# ---------------------- END MISCELLANEOUS FUNCTIONS FOR MY OWN USE ----------------------
