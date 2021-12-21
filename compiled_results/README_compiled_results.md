# Some compiled results

Isaac Cheng - December 2021

This folder contains _some_ results that may be relevant for the NGVS-VERTICO
collaboration. The full selection of results (except pickle files) are available in the
[`galaxies`](../galaxies/) folder. If you can't find the location of some result or don't
know where to look for the code used to generate these results (e.g., pickle files, plots,
functions or packages I mention, etc.), _please_ reach out to me! I would rather spend a
few minutes or even a few hours explaining how everything works as opposed to having
someone struggle to navigate this labyrinth...

# [Vorbin Radial Profiles](vorbin_radial_profiles/)

Contains high resolution radial profiles of stellar mass densities
([`sigstar`](vorbin_radial_profiles/sigstar/)), mass-to-light ratios
([`MLi`](vorbin_radial_profiles/MLi/)), and extinction-corrected u-g colours
([`u-g`](vorbin_radial_profiles/u-g/)). These data are from the
[coop_f2021/galaxies/vorbin_radial_profiles/](../galaxies/vorbin_radial_profiles/) folder.

Please see the [`stellar_mass_pipeline` folder](../stellar_mass_pipeline/) for a
walkthrough of how we go from NGVS data to all of these radial profiles. This folder
includes steps for fast(er) Voronoi binning.

# [Colour-Colour to Mass-to-Light Ratio Lookup Table](lookup_table/)

Contains a 4 panel plot showing our colour-colour to mass-to-light ratio lookup table.
Also contains each galaxy's colour-colour space overlaid on the sample-wide colour-colour
space. The steps and scripts to generate these plots and the lookup table, as well as how
to use the lookup table, are located in the [`stellar_mass_pipeline`
folder](../stellar_mass_pipeline/) (i.e., see
[`lookup_table_tutorial.ipynb`](../stellar_mass_pipeline/lookup_table_tutorial.ipynb)).

# [Molecular Gas Main Sequence](MGMS/)

Contains each galaxy's resolved molecular gas main sequence (rMGMS) as well as a global,
unresolved MGMS with points colour-coded by their rMGMS slope, and a few plots showing the
regions above/below/along the MGMS for NGC 4380. See
[`MGMS_tutorial.ipynb`](MGMS_tutorial.ipynb) for how to generate these data products and
plots. Requires the NGVS data to be binned to VERTICO's Nyquist-sampled resolution (i.e. 4
arcsec pixels). See
[`regbin_tutorial.ipynb`](../stellar_mass_pipeline/regbin_tutorial.ipynb)
for how to bin NGVS data to VERTICO's various resolutions.

# [Gas Fraction Radial Profiles](gas_fraction_profiles/)

Contains each galaxy's gas fraction radial profile and their uncertainties. I exclude
masked regions from all calculations since the gas fraction is very sensitive to zeros
(i.e., including masked regions as zeros here would skew the radial profile too much,
since we are using a log-scale). Also contains several gas fraction mosaics and mosaics of
their uncertainties. Please read the note in
[`gas_fraction_tutorial.ipynb`](gas_fraction_profiles/gas_fraction_tutorial.ipynb) for how
to interpret the ordering of the mosaic (short answer: don't take the ordering too
seriously). The notebook linked above also shows how to generate all the gas fraction
profiles and their plots.
