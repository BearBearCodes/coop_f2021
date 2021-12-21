# coop_f2021

Isaac Cheng's repo for his 2B work term with Drs. Toby Brown, Joel Roediger, and Matt
Taylor.

If you just need to use some code that I wrote, it can probably be found in my
[`astro-utils` repo](https://github.com/BearBearCodes/astro-utils).

## Table of Contents
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Quick Start](#quick-startquick_start)
- [`warmup` Folder](#warmup-folderwarmup)
- [`packages` Folder](#packages-folderpackages)
  - [Main Dependencies](#main-dependencies)
- [`stellar_mass_pipeline` Folder](#stellar_mass_pipeline-folderstellar_mass_pipeline)
- [`galaxies` Folder](#galaxies-foldergalaxies)
- [`stylesheets` Folder](#stylesheets-folderstylesheets)
- [`compiled_results` Folder](#compiled_results-foldercompiled_results)

<!-- /code_chunk_output -->

## [Quick Start](quick_start/)

I do _not_ recommend cloning this entire repository since it is very large! Instead, use
[`svn`](https://subversion.apache.org/) to download just the folder(s) or file(s) you
need. For example:

```bash
svn export https://github.com/BearBearCodes/coop_f2021/trunk/galaxies/make_rgb.ipynb
svn export https://github.com/BearBearCodes/coop_f2021/trunk/packages/
svn export https://github.com/BearBearCodes/coop_f2021/trunk/stellar_mass_pipeline/
```

Notice the `tree/master` portion of the url has been replaced with `trunk`.

If you require version control, you can `svn checkout` instead of `svn export`. See this
[StackOverflow
page](https://stackoverflow.com/questions/419467/difference-between-checkout-and-export-in-svn)
for more details.

## [`warmup` Folder](warmup/)

Warm-up exercises just to become familiarized with the CANFAR system.

## [`packages` Folder](packages/)

Contains some useful packages. Compiled primarily for my own use, but if you are not
dissuaded by their (lack of) organization, feel free to use these as well. Also see my
[`astro-utils` repo](https://github.com/BearBearCodes/astro-utils).

### Main Dependencies

Most of these should be installed in an `AstroConda` environment by default. Note that I
use f-strings in my code, so a Python version >= 3.6 is required.

- `numpy`
- `astropy`
- `matplotlib`
- `scipy`
- `reproject` (_IMPORTANT_: I use the latest `reproject v0.8`. Do not use `reproject
  v0.4`, which may be the version installed via `conda`. I recommend using `pip` to
  install this package.)
- `numba` and `multiprocessing` (not mandatory, but I use it in my notebooks to speed up
  some steps. `numba` is required for
  [`fast_vorbin`](https://github.com/BearBearCodes/coop_f2021/blob/master/packages/fast_vorbin.py))
- `photutils` (if you are using my radial profile code)
- `radio_beam` (if you are using my radial profile code)
- `seaborn` (if you are using my [`plot_utils`](packages/plot_utils.py) code)
- [`numba-kdtree`](https://github.com/mortacious/numba-kdtree) (if you are using
  [`fast_vorbin`](https://github.com/BearBearCodes/coop_f2021/blob/master/packages/fast_vorbin.py))

Note that `add_beam()` and `add_scalebeam()` from the
[`plot_utils`](packages/plot_utils.py) package use functions from
[`radial_profile_utils`](packages/radial_profile_utils.py).

## [`stellar_mass_pipeline` Folder](stellar_mass_pipeline/)

**A short walkthrough of how to go from NGVS u-, g-, i-, z-band data to stellar mass
density radial profiles**. Includes steps for regular binning, Voronoi binning,
colour-colour to mass-to-light ratio lookup table generation, and radial profile creation.

More details in the
[`README_stellar_mass_pipeline.md`](stellar_mass_pipeline/README_stellar_mass_pipeline.md)
file.

## [`galaxies` Folder](galaxies/)

**Contains the scripts and results for all the galaxies we've looked at**. All galaxies
have been regularly binned to VERTICO 2 arcsec and 4 arcsec pixel resolutions as well as
adaptively binned using Voronoi binning.

There are various README and .txt files in this folder and its subfolders. Please read
them as they contain important information.

N.B. I did not upload _any_ pickle files! Please contact me if you need them (they contain
the data I use to generate my plots, lookup table, etc).

## [`stylesheets` Folder](stylesheets/)

Just has some matplotlib stylesheets that I use for my beamer presentations.

## [`compiled_results` Folder](compiled_results/)

Contains a selection of results from the [`galaxies` folder](galaxies/) as well as some
sample code used to create these data products. Like the [`galaxies` folder](galaxies/),
no pickle files are included so contact me if you need the data products.

More details in the
[`README_compiled_results.md`](compiled_results/README_compiled_results.md)
file.
