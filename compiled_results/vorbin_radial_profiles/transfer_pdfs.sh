#!/bin/bash

PARENT_PATH="/arc/home/IsaacCheng/coop_f2021/galaxies/vorbin_radial_profiles/"
TARGET_PATH="/arc/home/IsaacCheng/coop_f2021/compiled_results/stellar_mass_radial_profiles/"

for galaxy in "IC3392" "NGC4216" "NGC4254" "NGC4298" "NGC4302" "NGC4330" "NGC4380" "NGC4388" "NGC4402" "NGC4419" "NGC4450" "NGC4522" "NGC4535" "NGC4567" "NGC4569" "NGC4580"  "NGC4651" "NGC4689" "NGC4192" "NGC4222" "NGC4294" "NGC4299" "NGC4321" "NGC4351" "NGC4383" "NGC4396" "NGC4405" "NGC4424" "NGC4501" "NGC4532" "NGC4548" "NGC4568" "NGC4579" "NGC4607"  "NGC4654" "NGC4694"
do
    # cp -nv "$PARENT_PATH""$galaxy/""$galaxy"_vorbin_SNR50_M_density_radProf_fromLookupTable_noSNRmask_i_corr.pdf "$TARGET_PATH""$galaxy/""$galaxy"_vorbin_SNR50_M_density_radProf_fromLookupTable_noSNRmask_i_corr.pdf
    # cp -nv "$PARENT_PATH""$galaxy/""$galaxy"_vorbin_SNR50_MLi_radProf_fromLookupTable_noSNRmask.pdf "$TARGET_PATH""$galaxy/""$galaxy"_vorbin_SNR50_MLi_radProf_fromLookupTable_noSNRmask.pdf
    # cp -nv "$PARENT_PATH""$galaxy/""$galaxy"_vorbin_SNR50_u-g_radProf_fromLookupTable_noSNRmask.pdf "$TARGET_PATH""$galaxy/""$galaxy"_vorbin_SNR50_u-g_radProf_fromLookupTable_noSNRmask.pdf
    # cp -nv "$PARENT_PATH""$galaxy/""$galaxy"_vorbin_SNR50_M_density_unc_radProf_fromLookupTable_noSNRmask_i_corr.pdf "$TARGET_PATH""$galaxy/""$galaxy"_vorbin_SNR50_M_density_unc_radProf_fromLookupTable_noSNRmask_i_corr.pdf
    # cp -nv "$PARENT_PATH""$galaxy/""$galaxy"_vorbin_SNR50_MLi_unc_radProf_fromLookupTable_noSNRmask.pdf "$TARGET_PATH""$galaxy/""$galaxy"_vorbin_SNR50_MLi_unc_radProf_fromLookupTable_noSNRmask.pdf
    cp -nv  "$PARENT_PATH""$galaxy/""$galaxy"_vorbin_SNR50_u-g_unc_radProf_fromLookupTable_noSNRmask.pdf "$TARGET_PATH""$galaxy/""$galaxy"_vorbin_SNR50_u-g_unc_radProf_fromLookupTable_noSNRmask.pdf
done