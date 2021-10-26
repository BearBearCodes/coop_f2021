#!/bin/bash
#
# Code to merge plots together. Requires pdfunite.
# Isaac Cheng - October 2021
#
# USAGE:
# ./pdfunite_script.sh <galaxy_name> <optional_suffix.pdf>
#
# Assumes tree is organized like this:
# .
# ├── NGC4380
# │   ├── NGC4380_file1.pdf
# │   ├── NGC4380_file2.pdf
# │   ├── NGC4380_file3.pdf
# │   ...
# └── pdfunite_script.sh


GALAXY=$1  # directory of galaxy (e.g., NGC4380)
FILE_SUFFIX=${2:-"compiled.pdf"}  # optional suffix for the output file

PREFIX=$GALAXY/$GALAXY
OUTFILE="$PREFIX"_"$FILE_SUFFIX"

echo "saving to: $OUTFILE"

pdfunite "$PREFIX"_Kkms.pdf "$PREFIX"_gasDensity_uncorr.pdf "$PREFIX"_gasDensity_corr.pdf \
"$PREFIX"_rgb.pdf "$PREFIX"_gi_profile.pdf "$OUTFILE"
