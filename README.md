# penetration_rate_cptu
Code and data to analyze and present the effect of varying the rate of penetration of piezocone tests.  In all 63 CPTus are considered in this study.

The source data was collected in 3 sessions at 3 different sites in central Norway, Tiller-Flotten and Øysand near Trondheim city, and Halsen in Stjørdal city.

## Data models
Considerable efforts were made to ensure high quality data in the analysis.  This includes development of a classes for soil stresses,  cpt data interpretation (another for raw-cpt data), and for detailed analysis of results.
![](https://raw.githubusercontent.com/siggimar/penetration_rate_cptu/refs/heads/main/data_vs_raw_40.png)
These efforts uncovered issues with standard dataformat at faster rates (this test at v=40mm/s)

## Rate effects
The results from this study are visualized using the (messy!) cptu_rate_plotter.py script, here you will find functions used to generate most of the figures used in the related chapter in the thesis.
![](https://raw.githubusercontent.com/siggimar/penetration_rate_cptu/refs/heads/main/k_model_bq.png)


## PCC_warp
This study included the development of PCC_warp.  A class to warp depths in soundings to best match curve characteristics in another.
![](https://raw.githubusercontent.com/siggimar/penetration_rate_cptu/refs/heads/main/PCC_warp%20example_HALS02.gif)
PCC_warp was used in place of DTW and DDTW (examples found herein) as these are found too liberal in stretching and squishing intervals for the best match.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14838136.svg)](https://doi.org/10.5281/zenodo.14838136)
