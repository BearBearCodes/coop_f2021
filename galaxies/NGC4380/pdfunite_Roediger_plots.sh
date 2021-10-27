GALAXY=$1  # directory of galaxy (e.g., NGC4380)
FILE_SUFFIX=${2:-"forJoel.pdf"}  # optional suffix for the output file

PREFIX="$GALAXY"
OUTFILE="$PREFIX"_"$FILE_SUFFIX"

echo "saving to: $OUTFILE"

pdfunite \
"$PREFIX"_TaylorMLi_vs_SED_iter03_scatter.pdf \
"$PREFIX"_TaylorMLi_vs_SED_iter03_kde.pdf \
"$PREFIX"_RoedigerMLi_vs_SED_iter03_scatter.pdf \
"$PREFIX"_RoedigerMLi_vs_SED_iter03_kde.pdf \
"$PREFIX"_MAP_SED_iter02_vs_iter03.pdf \
"$PREFIX"_P50age_SED_iter02_vs_iter03_scatter.pdf \
"$PREFIX"_P50age_SED_iter02_vs_iter03_kde.pdf \
"$OUTFILE"