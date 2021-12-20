# NGVS Stellar Mass Pipeline

Isaac Cheng - December 2021

## Overview

This folder is a short walkthrough of how to go from NGVS u-, g-, i-, z-band data to
stellar mass density radial profiles. It also covers how we do regular binning, Voronoi
binning, generating a colour-colour to mass-to-light ratio lookup table, and creating the
radial profiles themselves.

## Requirements

These scripts require the files in the [`packages` folder](../packages/).

---

## Procedure

Following are the steps to create your very own stellar mass density radial profile! I
will use the galaxy IC 3392 as an example.

### 1. Bin the NGVS data

Please see [`regbin_tutorial.ipynb`](regbin_tutorial.ipynb) for an example of binning
using a regular block size (e.g., 20x20 binning). See
[`vorbin_tutorial.ipynb`](vorbin_tutorial.ipynb) for an example of adaptive binning using
the [Voronoi binning algorithm](https://pypi.org/project/vorbin/) developed by [Cappellari
& Copin (2003)](https://ui.adsabs.harvard.edu/abs/2003MNRAS.342..345C). The latter
requires my [`fast_vorbin`](../packages/fast_vorbin.py) package.

### 2. Create colour-colour to mass-to-light ratio calibration

See [`lookup_table_tutorial.ipynb`](lookup_table_tutorial.ipynb) for how to create a
lookup table to efficiently assign mass-to-light ratios to some data given their u-i and
g-z colours. This lookup table should be generated using Voronoi binned data to ensure
each bin meets some minimum SNR!

### 3. Use the calibration to assign mass-to-light ratios and stellar mass densities to data

See [`assign_MLi_tutorial.ipynb`](assign_MLi_tutorial.ipynb) for how to use the lookup
table and for how to calculate stellar mass densities.

N.B. When using the lookup table, all data should be extinction-corrected and any
adaptively binned data should be normalized by the number of pixels in each bin (this is
to make sure the stellar mass densities are correct). The functions in
[`assign_MLi_tutorial.ipynb`](assign_MLi_tutorial.ipynb) already do this, but contact me
if you're confused as to why this only applies to Voronoi-binned data. Short answer: we
are calculating densities based on pixel sizes, and Voronoi-binned data keep their native
pixel sizes but just have different number of pixels in each bin. In regularly binned
data, the pixel sizes themselves change so each pixel is one bin.

### 4. Generate radial profiles

See [`radial_profile_tutorial.ipynb`](radial_profile_tutorial.ipynb) for how we generate
high-resolution radial profiles for all NGVS-VERTICO galaxies. If you've seen
[`radial_profile_example.ipynb`](../packages/radial_profile_example.ipynb) in the
[`packages` folder](../packages/), then you may not even need this. In this file, I just
show how I choose the number of annuli to use and how I set their minimum widths.
