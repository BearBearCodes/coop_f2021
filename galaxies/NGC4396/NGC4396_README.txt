There are several vorbin pickled results because I tried to increase the amount of pixels above an SNR of 30
by varying the SNR target and/or reference band (i.e., u-band or z-band).

Just use the Voronoi binned results with an SNR target of 50 in the z-band.
No appreciable difference between bands/SNR target value and the percentage of _bins_ below an SNR of 30 is negligible.
It is the number of bins, not pixels, below an SNR of 30 that matters.

Finally, to make the names consistent, I've duplicated some pickle files:
"NGC4396_vorbin_SNR50_In+Out_zband.pkl" == "NGC4396_vorbin_SNR50_In+Out.pkl"
"NGC4396_vorbin_SNR50_ugizBinned_zband.pkl" == "NGC4396_vorbin_SNR50_ugizBinned.pkl"
